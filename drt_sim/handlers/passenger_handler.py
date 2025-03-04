from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Dict
import traceback

from drt_sim.core.state.manager import StateManager
from drt_sim.core.simulation.context import SimulationContext
from drt_sim.core.monitoring.types.metrics import MetricName
from drt_sim.models.event import Event, EventType, EventPriority
from drt_sim.models.passenger import PassengerStatus, PassengerState
from drt_sim.models.request import Request
from drt_sim.config.config import ParameterSet
from drt_sim.models.matching import Assignment
from drt_sim.models.stop import StopAssignment
from drt_sim.core.coordination.stop_coordinator import StopCoordinator
import logging
logger = logging.getLogger(__name__)

class PassengerHandler:
    """
    Handles passenger journey events and states in the DRT system.
    Manages the complete passenger journey from initial walking to final destination.
    """
    
    def __init__(
        self,
        config: ParameterSet,
        context: SimulationContext,
        state_manager: StateManager,
        stop_coordinator: StopCoordinator
    ):
        self.config = config
        self.context = context
        self.state_manager = state_manager
        self.stop_coordinator = stop_coordinator
        self.service_thresholds = self._setup_service_thresholds()
        self.initialized = True
        
    def _setup_service_thresholds(self) -> Dict[str, float]:
        """Initialize service quality thresholds from config"""
        return {
            'max_wait_time': self.config.service.max_wait_time,
            'max_ride_time': self.config.service.max_ride_time,
            'max_walking_distance': self.config.service.max_walking_distance,
            'max_total_journey_time': self.config.service.max_journey_time
        }

    def handle_start_passenger_journey(self, event: Event) -> None:
        """Initial handler when a request is assigned to a vehicle"""
        try:
            logger.debug(f"Entering handle_start_passenger_journey for event {event.id}, "
             f"data: {event.data}")
            self.state_manager.begin_transaction()
            if 'assignment' not in event.data:
                logger.error(f"No assignment found in event data for event {event.id}. "
                            f"Event data keys: {list(event.data.keys())}")
                raise ValueError(f"Missing 'assignment' in event data for event {event.id}")
            assignment: Assignment = event.data['assignment']
            stop_assignment = self.state_manager.stop_assignment_worker.get_assignment(assignment.stop_assignment_id)
            request = self.state_manager.request_worker.get_request(assignment.request_id)
            if request.id == "R7" and request.passenger_id == "P8":
                logger.info(f"Found problematic request R7/passenger P8! Assignment details: "
                            f"stop_assignment_id={assignment.stop_assignment_id}, "
                            f"vehicle_id={assignment.vehicle_id}")
                
                # Verify the stop_assignment exists
                stop_assignment_exists = self.state_manager.stop_assignment_worker.get_assignment(
                    assignment.stop_assignment_id) is not None
                logger.info(f"Stop assignment {assignment.stop_assignment_id} exists: {stop_assignment_exists}")
                
                # Check transaction status
                logger.info(f"Transaction active: {self.state_manager.is_transaction_active()}")
            if not request:
                raise ValueError(f"Request {assignment.request_id} not found")
            
            # Initialize passenger journey
            passenger_state = PassengerState(
                id=request.passenger_id,
                request_id=request.id,
                status=PassengerStatus.WALKING_TO_PICKUP,
                current_location=request.origin,
                assigned_origin_stop=stop_assignment.origin_stop,
                assigned_destination_stop=stop_assignment.destination_stop,
                assigned_vehicle_id=assignment.vehicle_id,
                estimated_pickup_time=assignment.estimated_pickup_time,
                estimated_dropoff_time=assignment.estimated_dropoff_time,
                walking_to_pickup_start_time=self.context.current_time,
            )
            
            # Create initial passenger state
            self.state_manager.passenger_worker.create_passenger_state(passenger_state)
            
            # Create walking to pickup event
            self._create_walking_to_pickup_event(request, stop_assignment)
            logger.debug(f"Successfully created passenger state for passenger_id={request.passenger_id}, "
             f"request_id={request.id}, stop_assignment_id={assignment.stop_assignment_id}")
            self.state_manager.commit_transaction()
            
        except Exception as e:
            self.state_manager.rollback_transaction()
            logger.error(f"Error handling request assignment: {str(e)}\n{traceback.format_exc()}")
            self._handle_passenger_error(event, str(e))

    def handle_passenger_walking_to_pickup(self, event: Event) -> None:
        """Handle passenger starting walk to pickup location"""
        try:
            self.state_manager.begin_transaction()
            request_id = event.request_id
            request = self.state_manager.request_worker.get_request(request_id)
            if not request:
                raise ValueError(f"Request {request_id} not found")
            
            # Schedule arrival at pickup based on walking time
            walking_time = event.data.get('estimated_walking_time')
            arrival_time = self.context.current_time + timedelta(seconds=walking_time)
            
            # Create arrival event
            self._create_passenger_arrived_pickup_event(
                request.passenger_id,
                request.id,
                arrival_time,
                walking_time
            )
            
            self.state_manager.commit_transaction()
            
        except Exception as e:
            self.state_manager.rollback_transaction()
            logger.error(f"Error handling walking to pickup: {str(e)}\n{traceback.format_exc()}")
            self._handle_passenger_error(event, str(e))

    def handle_passenger_arrived_pickup(self, event: Event) -> None:
        """Handle passenger arrival at pickup location"""
        try:
            self.state_manager.begin_transaction()
            
            # Update passenger state
            passenger_state = self.state_manager.passenger_worker.update_passenger_status(
                event.passenger_id,
                PassengerStatus.ARRIVED_AT_PICKUP,
                self.context.current_time,
                {
                    'walking_to_pickup_end_time': self.context.current_time,
                    'current_location': event.data.get('location')
                }
            )

            # Register arrival with coordinator
            self.stop_coordinator.register_passenger_arrival(
                stop_id=passenger_state.assigned_origin_stop.id,
                passenger_id=passenger_state.id,
                request_id=passenger_state.request_id,
                arrival_time=self.context.current_time,
                location=event.data.get('location')
            )
            
            self.state_manager.commit_transaction()
            
        except Exception as e:
            self.state_manager.rollback_transaction()
            logger.error(f"Error handling arrival at pickup: {traceback.format_exc()}")
            self._handle_passenger_error(event, str(e))

    def handle_boarding_completed(self, event: Event) -> None:
        """Handle passenger boarding completed event."""
        try:
            self.state_manager.begin_transaction()
            
            passenger_id = event.passenger_id
            request_id = event.request_id
            vehicle_id = event.vehicle_id
            
            logger.info(f"Processing boarding completion for passenger {passenger_id} on vehicle {vehicle_id}")

            passenger_state = self.state_manager.passenger_worker.get_passenger(passenger_id)
            if not passenger_state:
                raise ValueError(f"No state found for passenger {passenger_id}")
                
            # Update passenger status
            self.state_manager.passenger_worker.update_passenger_status(
                passenger_id,
                PassengerStatus.IN_VEHICLE,
                self.context.current_time,
                {
                    'boarding_end_time': self.context.current_time,
                    'in_vehicle_start_time': self.context.current_time
                }
            )
            
            self.state_manager.commit_transaction()
            
        except Exception as e:
            self.state_manager.rollback_transaction()
            logger.error(f"Error handling boarding: {traceback.format_exc()}")
            self._handle_passenger_error(event, str(e))

    def handle_alighting_completed(self, event: Event) -> None:
        """Handle complete alighting process from in-vehicle to walking"""
        if not self.initialized:
            raise RuntimeError("Handler not initialized")
            
        try:
            self.state_manager.begin_transaction()
            passenger_id = event.passenger_id
            vehicle_id = event.vehicle_id
            
            passenger_state = self.state_manager.passenger_worker.get_passenger(passenger_id)
            if not passenger_state:
                raise ValueError(f"No state found for passenger {passenger_id}")
                
            # Calculate and log ride time
            if self.context.metrics_collector:
                ride_time = (
                    self.context.current_time - passenger_state.in_vehicle_start_time
                ).total_seconds() if passenger_state.in_vehicle_start_time else 0
                
                self.context.metrics_collector.log(
                    MetricName.PASSENGER_RIDE_TIME,
                    ride_time,
                    self.context.current_time,
                    {
                        'passenger_id': passenger_id,
                        'vehicle_id': vehicle_id,
                    }
                )
            
            alighting_start_time = event.data.get('alighting_start_time')
            actual_dropoff_time = event.data.get('actual_dropoff_time')
            # Calculate and check ride time
            ride_time = (self.context.current_time - passenger_state.in_vehicle_start_time).total_seconds()
            if ride_time > self.service_thresholds['max_ride_time']:
                self._create_service_violation_event(
                    passenger_id,
                    vehicle_id,
                    'ride_time',
                    ride_time
                )
            alighting_end_time = self.context.current_time
            # Update passenger state directly to walking
            self.state_manager.passenger_worker.update_passenger_status(
                event.passenger_id,
                PassengerStatus.WALKING_TO_DESTINATION,
                self.context.current_time,
                {
                    'alighting_start_time': alighting_start_time,
                    'alighting_end_time': alighting_end_time,
                    'in_vehicle_end_time': alighting_end_time,
                    'walking_to_destination_start_time': alighting_end_time,
                    'actual_dropoff_time': actual_dropoff_time
                }
            )

            stop_assignment = self.state_manager.stop_assignment_worker.get_assignment_for_request(event.request_id)
            if not stop_assignment:
                raise ValueError(f"Stop assignment for request {event.request_id} not found")
            
            walking_time = stop_assignment.walking_time_destination
            arrival_time = self.context.current_time + timedelta(seconds=walking_time)
            
            self._create_passenger_arrived_destination_event(
                event.passenger_id,
                event.request_id,
                arrival_time,
                walking_time
            )
            
            self.state_manager.commit_transaction()
            
        except Exception as e:
            self.state_manager.rollback_transaction()
            logger.error(f"Error handling alighting: {str(e)}")
            self._handle_passenger_error(event, str(e))

    def handle_passenger_arrived_destination(self, event: Event) -> None:
        """Handle passenger arrival at final destination"""
        try:
            self.state_manager.begin_transaction()
            
            # Update final state
            passenger_state = self.state_manager.passenger_worker.update_passenger_status(
                event.passenger_id,
                PassengerStatus.ARRIVED_AT_DESTINATION,
                self.context.current_time,
                {
                    'walking_to_destination_end_time': self.context.current_time,
                }
            )

            request_id = passenger_state.request_id
            request = self.state_manager.request_worker.get_request(request_id)

            self.context.metrics_collector.log(
                    MetricName.PASSENGER_WALK_TIME_FROM_DESTINATION_STOP,
                    event.data.get('walking_time'),
                    self.context.current_time,
                    {
                        'passenger_id': event.passenger_id,
                        'destination': request.destination.to_dict(),
                        'destination_stop': passenger_state.assigned_destination_stop.location.to_dict(),
                    }
                )
            
            self.state_manager.commit_transaction()
            
        except Exception as e:
            self.state_manager.rollback_transaction()
            logger.error(f"Error handling arrival at destination: {str(e)}")
            self._handle_passenger_error(event, str(e))

    def handle_passenger_no_show(self, event: Event) -> None:
        """Handle passenger no-show"""
        try:
            self.state_manager.begin_transaction()
            
            # Update status to cancelled
            self.state_manager.passenger_worker.update_passenger_status(
                event.passenger_id,
                PassengerStatus.CANCELLED,
                self.context.current_time,
                {
                    'cancellation_reason': 'no_show',
                    'cancellation_time': self.context.current_time
                }
            )
            self.context.metrics_collector.log(
                MetricName.PASSENGER_NO_SHOW,
                1,
                self.context.current_time,
                {
                    'passenger_id': event.passenger_id,
                }
            )
            
            self.state_manager.commit_transaction()
            
        except Exception as e:
            self.state_manager.rollback_transaction()
            logger.error(f"Error handling no-show: {str(e)}")
            self._handle_passenger_error(event, str(e))
    
    def handle_service_level_violation(self, event: Event) -> None:
        """Handle service level violation events"""
        try:
            self.state_manager.begin_transaction()
            
            violation_type = event.data.get('violation_type')
            actual_value = event.data.get('actual_value')
            violation_time = event.data.get('violation_time')
            
            # Record violation using the worker's method
            self.state_manager.passenger_worker.record_service_violation(
                event.passenger_id,
                violation_type,
                actual_value,
                violation_time
            )
            self.context.metrics_collector.log(
                MetricName.SERVICE_VIOLATIONS,
                1,
                self.context.current_time,
                {
                    'passenger_id': event.passenger_id,
                    'vehicle_id': event.vehicle_id,
                    'violation_type': violation_type,
                    'measured_value': actual_value,
                    'threshold': self.service_thresholds[f'max_{violation_type}'],
                    'timestamp': violation_time.isoformat()
                }
            )
            self.state_manager.commit_transaction()
        except Exception as e:
            self.state_manager.rollback_transaction()
            logger.error(f"Error handling service violation: {str(e)}")
            self._handle_passenger_error(event, str(e))

    def _create_walking_to_pickup_event(self, request: Request, stop_assignment: StopAssignment) -> None:
        """Create event for passenger walking to pickup"""
        event = Event(
            event_type=EventType.PASSENGER_WALKING_TO_PICKUP,
            priority=EventPriority.HIGH,
            timestamp=self.context.current_time,
            passenger_id=request.passenger_id,
            request_id=request.id,
            data={
                'origin': request.origin,
                'destination': stop_assignment.origin_stop,
                'estimated_walking_time': stop_assignment.walking_time_origin,
                'estimated_walking_distance': stop_assignment.walking_distance_origin
            }
        )
        self.context.event_manager.publish_event(event)

    def _create_boarding_event(self, passenger_id: str, request_id: str, vehicle_id: str) -> None:
        """Create event for passenger boarding"""
        event = Event(
            event_type=EventType.PASSENGER_BOARDING_COMPLETED,
            priority=EventPriority.HIGH,
            timestamp=self.context.current_time + timedelta(seconds=self.config.vehicle.boarding_time),
            passenger_id=passenger_id,
            request_id=request_id,
            vehicle_id=vehicle_id,
            data={'boarding_start_time': self.context.current_time, 'actual_pickup_time': self.context.current_time}
        )
        self.context.event_manager.publish_event(event)
    
    def _create_alighting_event(self, passenger_id: str, request_id: str) -> None:
        """Create event for passenger alighting"""
        event = Event(
            event_type=EventType.PASSENGER_ALIGHTING_COMPLETED,
            priority=EventPriority.HIGH,
            timestamp=self.context.current_time + timedelta(seconds=self.config.vehicle.alighting_time),
            passenger_id=passenger_id,
            request_id=request_id,
            data={'alighting_start_time': self.context.current_time, 'actual_dropoff_time': self.context.current_time}
        )
        self.context.event_manager.publish_event(event)

    def _create_service_violation_event(
        self,
        passenger_id: str,
        vehicle_id: str,
        violation_type: str,
        actual_value: float
    ) -> None:
        """Create event for service level violation"""
        event = Event(
            event_type=EventType.SERVICE_LEVEL_VIOLATION,
            priority=EventPriority.HIGH,
            timestamp=self.context.current_time,
            passenger_id=passenger_id,
            vehicle_id=vehicle_id,
            data={
                'violation_type': violation_type,
                'actual_value': actual_value,
                'threshold': self.service_thresholds[f'max_{violation_type}'],
                'violation_time': self.context.current_time
            }
        )
        self.context.event_manager.publish_event(event)

    def _handle_passenger_error(self, event: Event, error_msg: str) -> None:
        """Handle errors in passenger event processing"""
        logger.error(f"Error processing passenger event {event.id}: {error_msg}")
        error_event = Event(
            event_type=EventType.SIMULATION_ERROR,
            priority=EventPriority.CRITICAL,
            timestamp=self.context.current_time,
            passenger_id=event.passenger_id if hasattr(event, 'passenger_id') else None,
            request_id=event.request_id if hasattr(event, 'request_id') else None,
            data={
                'error': error_msg,
                'original_event': event.to_dict(),
                'error_type': 'passenger_processing_error'
            }
        )
        self.context.event_manager.publish_event(error_event)

    def _create_passenger_arrived_pickup_event(
        self,
        passenger_id: str,
        request_id: str,
        arrival_time: datetime,
        walking_time: float
    ) -> None:
        """Create event for passenger arrival at pickup location"""
        passenger_state = self.state_manager.passenger_worker.get_passenger(passenger_id)
        
        event = Event(
            event_type=EventType.PASSENGER_ARRIVED_PICKUP,
            priority=EventPriority.HIGH,
            timestamp=arrival_time,
            passenger_id=passenger_id,
            request_id=request_id,
            data={
                'location': passenger_state.assigned_origin_stop,
                'walking_time': walking_time,
                'assigned_vehicle_id': passenger_state.assigned_vehicle_id
            }
        )
        self.context.event_manager.publish_event(event)

    def _create_passenger_arrived_destination_event(
        self,
        passenger_id: str,
        request_id: str,
        arrival_time: datetime,
        walking_time: float
    ) -> None:
        """Create event for passenger arrival at final destination"""
        request = self.state_manager.request_worker.get_request(request_id)
        
        event = Event(
            event_type=EventType.PASSENGER_ARRIVED_DESTINATION,
            priority=EventPriority.HIGH,
            timestamp=arrival_time,
            passenger_id=passenger_id,
            request_id=request_id,
            data={
                'location': request.destination,
                'walking_time': walking_time
            }
        )
        self.context.event_manager.publish_event(event)