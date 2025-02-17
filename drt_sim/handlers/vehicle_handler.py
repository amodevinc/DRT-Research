from datetime import datetime, timedelta
from typing import Dict, Any
from drt_sim.core.state.manager import StateManager
from drt_sim.core.simulation.context import SimulationContext
from drt_sim.models.event import Event, EventType, EventPriority
from drt_sim.models.vehicle import VehicleStatus
from drt_sim.config.config import ParameterSet
from drt_sim.models.route import RouteSegment, RouteStatus, RouteStop, Route
from drt_sim.models.stop import StopPurpose
from drt_sim.models.passenger import PassengerStatus
from drt_sim.core.monitoring.types.metrics import MetricName
import traceback
import logging
logger = logging.getLogger(__name__)

class VehicleHandler:
    """
    Handles vehicle operations in the DRT system.
    Manages vehicle states, movements, capacity, and maintenance.
    """
    
    def __init__(
        self,
        config: ParameterSet,
        context: SimulationContext,
        state_manager: StateManager
    ):
        self.config = config
        self.context = context
        self.state_manager = state_manager
        self.vehicle_thresholds = self._setup_vehicle_thresholds()
        
    def _setup_vehicle_thresholds(self) -> Dict[str, Any]:
        """Initialize vehicle operation thresholds from config"""
        return {
            'max_pickup_delay': self.config.vehicle.max_pickup_delay,
            'max_dropoff_delay': self.config.vehicle.max_dropoff_delay
        }
    
    def handle_vehicle_active_route_id_update(self, event: Event) -> None:
        """Handle vehicle active route update"""
        try:
            vehicle_id = event.vehicle_id
            route_id = event.data['route_id']
            logger.info(f"Updating vehicle {vehicle_id} active route to {route_id}")
            return self.state_manager.vehicle_worker.update_vehicle_active_route_id(
                vehicle_id,
                route_id
            )
        except Exception as e:
            logger.error(f"Error handling vehicle active route update: {traceback.format_exc()}")
            self._handle_vehicle_error(event, str(e))

    def handle_vehicle_dispatch_request(self, event: Event) -> None:
        """Handle initial vehicle dispatch request."""
        try:
            logger.info(f"Handling vehicle dispatch request: {event}")
            self.state_manager.begin_transaction()
            
            vehicle_id = event.vehicle_id
            route_id = event.data['route_id']
            self.state_manager.vehicle_worker.update_vehicle_active_route_id(
                vehicle_id,
                route_id
            )
            vehicle = self.state_manager.vehicle_worker.get_vehicle(vehicle_id)
            if not vehicle:
                raise ValueError(f"Vehicle {vehicle_id} not found")
            
            route = self.state_manager.route_worker.get_route(vehicle.get_active_route_id())
            if not route:
                raise ValueError(f"No active route found for vehicle {vehicle_id}")
            
            logger.info(f"Route: {route}")
                
            # Get first/current segment
            current_segment = route.get_current_segment()
            logger.info(f"Current segment: {current_segment}")
            if not current_segment:
                raise ValueError(f"No current segment found in route for vehicle {vehicle_id}")
            
            # Update route status if it's just starting
            if route.status == RouteStatus.CREATED or route.status == RouteStatus.PLANNED:
                route.status = RouteStatus.ACTIVE
                route.actual_start_time = self.context.current_time
                self.state_manager.route_worker.update_route(route)
            
            # Update vehicle status to in service
            if vehicle.current_state.status != VehicleStatus.IN_SERVICE:
                logger.info(f"Updating vehicle {vehicle_id} status to IN_SERVICE")
                self.state_manager.vehicle_worker.update_vehicle_status(
                    vehicle_id,
                    VehicleStatus.IN_SERVICE,
                    self.context.current_time
                )
            
            # Create movement event for the current segment
            logger.info(f"Creating movement event for vehicle {vehicle_id} on segment {current_segment.id}")
            self._create_vehicle_movement_event(
                vehicle_id=vehicle_id,
                segment=current_segment,
                current_time=self.context.current_time
            )
            
            self.state_manager.commit_transaction()
            
        except Exception as e:
            self.state_manager.rollback_transaction()
            logger.error(f"Error handling vehicle dispatch: {str(e)}")
            self._handle_vehicle_error(event, str(e))

    def handle_vehicle_reroute_request(self, event: Event) -> None:
        """
        Handle rerouting request for vehicle already in service.
        The route has already been updated in state, so we just need to
        trigger the vehicle to start following the updated route.
        """
        try:
            self.state_manager.begin_transaction()
            
            vehicle_id = event.vehicle_id
            vehicle = self.state_manager.vehicle_worker.get_vehicle(vehicle_id)
            route_id = event.data['route_id']
            self.state_manager.vehicle_worker.update_vehicle_active_route_id(
                vehicle_id,
                route_id
            )
            
            if not vehicle:
                raise ValueError(f"Vehicle {vehicle_id} not found")
            
            # Get the updated route from state
            route = self.state_manager.route_worker.get_route(vehicle.get_active_route_id())
            if not route:
                raise ValueError(f"No active route found for vehicle {vehicle_id}")
            
            # Get current segment
            current_segment = route.get_current_segment()
            if not current_segment:
                raise ValueError(f"No current segment found in route for vehicle {vehicle_id}")
                
            if vehicle.current_state.status != VehicleStatus.IN_SERVICE:
                # Update vehicle status with new route information
                self.state_manager.vehicle_worker.update_vehicle_status(
                    vehicle_id,
                    VehicleStatus.IN_SERVICE,
                    self.context.current_time
                )
            
            # Create movement event for current segment
            self._create_vehicle_movement_event(
                vehicle_id=vehicle_id,
                segment=current_segment,
                current_time=self.context.current_time
            )
            
            self.state_manager.commit_transaction()
            
        except Exception as e:
            self.state_manager.rollback_transaction()
            logger.error(f"Error handling vehicle reroute: {str(e)}")
            self._handle_vehicle_error(event, str(e))

    def handle_vehicle_rebalancing_required(self, event: Event) -> None:
        """Handle rebalancing request to depot."""
        try:
            self.state_manager.begin_transaction()
            
            vehicle_id = event.vehicle_id

            logger.info(f"Rebalancing Implementation Required for rebalancing vehicles")
            
            # # Update vehicle status to rebalancing
            # self.state_manager.vehicle_worker.update_vehicle_status(
            #     vehicle_id,
            #     VehicleStatus.REBALANCING,
            #       self.context.current_time
            # )
            
            # # Create movement event to depot
            # estimated_duration = self._estimate_travel_time(
            #     event.data['origin'],
            #     event.data['destination']
            # )
            # estimated_distance = self._estimate_travel_distance(
            #     event.data['origin'],
            #     event.data['destination']
            # )
            
            # # Create movement event
            # movement_data = {
            #     'origin': event.data['origin'],
            #     'destination': event.data['destination'],
            #     'estimated_duration': estimated_duration,
            #     'estimated_distance': estimated_distance,
            #     'movement_type': 'rebalancing',
            #     'depot_id': event.data['depot_id'],
            #     'movement_start_time': self.context.current_time,
            #     'expected_arrival_time': self.context.current_time + timedelta(seconds=estimated_duration)
            # }
            
            # event = Event(
            #     event_type=EventType.VEHICLE_EN_ROUTE,
            #     priority=EventPriority.HIGH,
            #     timestamp=self.context.current_time,
            #     vehicle_id=vehicle_id,
            #     data=movement_data
            # )
            # self.context.event_manager.publish_event(event)
            
            self.state_manager.commit_transaction()
            
        except Exception as e:
            self.state_manager.rollback_transaction()
            logger.error(f"Error handling rebalancing request: {str(e)}")
            self._handle_vehicle_error(event, str(e))

    def handle_vehicle_en_route(self, event: Event) -> None:
        """
        Handles VEHICLE_EN_ROUTE event by scheduling a future VEHICLE_ARRIVED event
        """
        try:
            logger.info(f"Handling vehicle en route event: {event}")
            vehicle_id = event.vehicle_id
            segment_id = event.data['segment_id']
            estimated_duration = event.data['estimated_duration']
            
            # Schedule arrival event for the future
            arrival_time = self.context.current_time + timedelta(seconds=estimated_duration)
            
            arrival_event = Event(
                event_type=EventType.VEHICLE_ARRIVED_STOP,
                priority=EventPriority.HIGH,
                timestamp=arrival_time,  # Future time when vehicle will arrive
                vehicle_id=vehicle_id,
                data={
                    'segment_id': segment_id,
                    'movement_start_time': event.data['movement_start_time'],
                    'origin': event.data['origin'],
                    'destination': event.data['destination'],
                    'actual_duration': estimated_duration,
                    'actual_distance': event.data['estimated_distance']
                }
            )
            
            # Add event to simulation's future event queue
            self.context.event_manager.publish_event(arrival_event)
            
        except Exception as e:
            logger.error(f"Error handling vehicle en route: {str(e)}")
            self._handle_vehicle_error(event, str(e))

    def handle_vehicle_arrived_stop(self, event: Event) -> None:
        """
        Handle vehicle arrival at a stop and manage next steps.
        Handles both regular stop arrivals and rebalancing arrivals.
        """
        try:
            logger.info(f"Handling vehicle arrived stop event: {event}")
            self.state_manager.begin_transaction()
            
            vehicle_id = event.vehicle_id
            vehicle = self.state_manager.vehicle_worker.get_vehicle(vehicle_id)
            if not vehicle:
                raise ValueError(f"Vehicle {vehicle_id} not found")
                
            # Log vehicle metrics at stop arrival
            if self.context.metrics_collector:
                logger.info(f"Logging vehicle metrics at stop arrival")
            
            # Update vehicle location first for all arrival types
            self.state_manager.vehicle_worker.update_vehicle_location(
                vehicle_id,
                event.data['destination']
            )
            # Log distance metric if available
            distance = event.data.get('actual_distance', 0)
            if vehicle.current_state.current_occupancy > 0:
                self.context.metrics_collector.log(
                    MetricName.VEHICLE_OCCUPIED_DISTANCE,
                    distance,
                    self.context.current_time,
                    { 'vehicle_id': vehicle_id }
                )
            else:
                self.context.metrics_collector.log(
                    MetricName.VEHICLE_EMPTY_DISTANCE,
                    distance,
                    self.context.current_time,
                    { 'vehicle_id': vehicle_id }
                )
            
            # Handle rebalancing arrival
            if event.data.get('movement_type') == 'rebalancing':
                self._handle_rebalancing_arrival(vehicle_id, event.data)
                self.state_manager.commit_transaction()
                return
                
            # Handle regular stop arrival
            self._handle_regular_stop_arrival(vehicle_id, event)
            
            self.state_manager.commit_transaction()
            
        except Exception as e:
            self.state_manager.rollback_transaction()
            logger.error(f"Error handling vehicle arrival: {traceback.format_exc()}")
            self._handle_vehicle_error(event, str(e))

    def _handle_rebalancing_arrival(self, vehicle_id: str, event_data: Dict[str, Any]) -> None:
        """Handle vehicle arrival at depot after rebalancing."""
        self.state_manager.vehicle_worker.update_vehicle_status(
            vehicle_id,
            VehicleStatus.IDLE,
            self.context.current_time
        )

    def _handle_regular_stop_arrival(self, vehicle_id: str, event: Event) -> None:
        """Handle vehicle arrival at a regular stop."""
        # Get and validate route
        vehicle = self.state_manager.vehicle_worker.get_vehicle(vehicle_id)
        if not vehicle:
            raise ValueError(f"Vehicle {vehicle_id} not found")
        
        route = self.state_manager.route_worker.get_route(vehicle.get_active_route_id())
        if not route:
            raise ValueError(f"No active route found for vehicle {vehicle_id}")
        
        current_segment = route.get_current_segment()
        if not current_segment:
            raise ValueError("No current segment found in route")
        
        # Update segment completion data
        current_stop = current_segment.destination
        self._update_segment_completion(current_segment, current_stop, event)
        
        # Handle passenger operations at stop
        current_time = self._handle_stop_operations(vehicle_id, current_stop)
        if self.context.metrics_collector:
            self.context.metrics_collector.log(
                MetricName.VEHICLE_STOPS_SERVED,
                1,
                self.context.current_time,
                { 'vehicle_id': vehicle_id }
            )
        
        # Determine next action based on route completion
        if self._is_route_completed(route):
            self._handle_route_completion(vehicle_id, route, current_time)
        else:
            self._handle_route_continuation(vehicle_id, route, current_time)
        
        # Update route in state
        self.state_manager.route_worker.update_route(route)

    def _update_segment_completion(
        self, 
        segment: RouteSegment, 
        stop: RouteStop, 
        event: Event
    ) -> None:
        """Update segment completion metrics and timing."""
        stop.actual_arrival_time = self.context.current_time
        segment.completed = True
        segment.actual_duration = (
            self.context.current_time - event.data['movement_start_time']
        ).total_seconds()
        segment.actual_distance = event.data.get('actual_distance', 
                                            segment.estimated_distance)

    def _handle_stop_operations(self, vehicle_id: str, stop: RouteStop) -> None:
        """Handle passenger operations at the stop."""
        logger.info(f"Handling stop operations for vehicle {vehicle_id} at stop {stop}")
        current_time = self.context.current_time
        if stop.pickup_passengers:
            current_time = self._handle_pickup_arrival(vehicle_id, stop, current_time)
        if stop.dropoff_passengers:
            current_time = self._handle_dropoff_arrival(vehicle_id, stop, current_time)
        return current_time

    def _is_route_completed(self, route: 'Route') -> bool:
        """Check if route is completed."""
        return route.current_segment_index >= len(route.segments) - 1

    def _handle_route_completion(
        self, 
        vehicle_id: str, 
        route: 'Route',
        current_time: datetime
    ) -> None:
        """
        Handle completion of route and initiate rebalancing through event publishing.
        Updates route status and triggers rebalancing request.
        """
        # Update route status
        route.status = RouteStatus.COMPLETED
        route.actual_end_time = current_time

        # Create rebalancing required event
        self._create_rebalancing_event(vehicle_id)

    def _handle_route_continuation(self, vehicle_id: str, route: 'Route', current_time: datetime) -> None:
        """Handle continuation to next segment in route."""
        route.current_segment_index += 1
        next_segment = route.get_current_segment()
        
        # Create movement event for next segment
        self._create_vehicle_movement_event(
            vehicle_id=vehicle_id,
            segment=next_segment,
            current_time=current_time
        )
            
    def _handle_pickup_arrival(self, vehicle_id: str, route_stop: RouteStop, current_time: datetime) -> datetime:
        """Handle vehicle arrival for pickup."""
        logger.info(f"Handling pickup arrival for vehicle {vehicle_id} at stop {route_stop}")
        # Check for late arrival
        if current_time > route_stop.planned_arrival_time:
            delay = (current_time - route_stop.planned_arrival_time).total_seconds()
            self._handle_pickup_delay_violation(
                vehicle_id,
                delay
            )
        
        # Get waiting passengers at this stop
        waiting_passengers = self.state_manager.passenger_worker.get_passengers_at_stop(
            PassengerStatus.WAITING_FOR_VEHICLE,
            route_stop.stop.id,
            StopPurpose.PICKUP
        )
        logger.info(f"Waiting passengers: {waiting_passengers}")
        # Create boarding completed events for waiting passengers
        current_boarding_time = current_time
        for passenger in waiting_passengers:
            if passenger.assigned_vehicle_id == vehicle_id:
                event = Event(
                    event_type=EventType.PASSENGER_BOARDING_COMPLETED,
                    priority=EventPriority.HIGH,
                    timestamp=current_boarding_time + timedelta(seconds=self.config.vehicle.boarding_time),
                    passenger_id=passenger.id,
                    request_id=passenger.request_id,
                    vehicle_id=vehicle_id,
                    data={
                        'boarding_start_time': current_boarding_time,
                        'actual_pickup_time': current_boarding_time
                    }
                )
                self.context.event_manager.publish_event(event)
                # Update for next passenger's boarding time
                current_boarding_time += timedelta(seconds=self.config.vehicle.boarding_time)
                self.state_manager.vehicle_worker.increment_vehicle_occupancy(vehicle_id)
        # Check for missing passengers and create wait timeout if needed
        expected_passengers = route_stop.pickup_passengers
        expected_passenger_ids = self.state_manager.passenger_worker.get_all_passenger_ids_for_request_ids(expected_passengers)
        waiting_passenger_ids = {p.id for p in waiting_passengers}
        missing_passengers = set(expected_passenger_ids) - set(waiting_passenger_ids)
        
        if missing_passengers:
            wait_event = Event(
                event_type=EventType.VEHICLE_WAIT_TIMEOUT,
                priority=EventPriority.HIGH,
                timestamp=current_time + timedelta(seconds=self.config.vehicle.max_dwell_time),
                vehicle_id=vehicle_id,
                data={
                    'stop_id': route_stop.stop.id,
                    'missing_passengers': list(missing_passengers),
                    'wait_start_time': current_time
                }
            )
            self.context.event_manager.publish_event(wait_event)
        return current_boarding_time

    def handle_vehicle_wait_timeout(self, event: Event) -> None:
        """Handle timeout of vehicle waiting for passengers."""
        try:
            self.state_manager.begin_transaction()
            
            stop_id = event.data['stop_id']
            initially_missing_passengers = event.data['missing_passengers']
            vehicle_id = event.vehicle_id
            
            # Check which passengers are still not at the stop
            still_missing_passengers = []
            for passenger_id in initially_missing_passengers:
                passenger_state = self.state_manager.passenger_worker.get_passenger(passenger_id)
                
                # Skip if passenger state not found (shouldn't happen but defensive)
                if not passenger_state:
                    logger.warning(f"Passenger {passenger_id} not found during wait timeout check")
                    continue
                    
                # Skip if passenger is already cancelled
                if passenger_state.status == PassengerStatus.CANCELLED:
                    logger.info(f"Passenger {passenger_id} already cancelled, skipping no-show event")
                    continue
                    
                # Only consider passengers who never made it to the stop
                if passenger_state.status == PassengerStatus.WALKING_TO_PICKUP:
                    still_missing_passengers.append(passenger_id)
            
            # Mark remaining missing passengers as no-show
            for passenger_id in still_missing_passengers:
                no_show_event = Event(
                    event_type=EventType.PASSENGER_NO_SHOW,
                    priority=EventPriority.HIGH,
                    timestamp=self.context.current_time,
                    passenger_id=passenger_id,
                    vehicle_id=vehicle_id,
                    data={
                        'stop_id': stop_id,
                        'wait_duration': (self.context.current_time - event.data['wait_start_time']).total_seconds()
                    }
                )
                self.context.event_manager.publish_event(no_show_event)
            
            vehicle = self.state_manager.vehicle_worker.get_vehicle(vehicle_id)
            if not vehicle:
                raise ValueError(f"Vehicle {vehicle_id} not found")
            
            # Continue with route
            route = self.state_manager.route_worker.get_route(vehicle.get_active_route_id())
            if route:
                self._handle_route_continuation(vehicle_id, route, self.context.current_time)
            
            # Log dwell time metric from wait timeout event
            if self.context.metrics_collector:
                dwell_duration = (self.context.current_time - event.data['wait_start_time']).total_seconds()
                self.context.metrics_collector.log(
                    MetricName.VEHICLE_DWELL_TIME,
                    dwell_duration,
                    self.context.current_time,
                    {
                        'vehicle_id': vehicle_id,
                        'stop_id': event.data['stop_id'],
                        'dwell_start_time': event.data['wait_start_time'],
                        'dwell_end_time': self.context.current_time.isoformat(),
                        'reason': 'wait_timeout'
                    }
                )
            self.state_manager.commit_transaction()
            
        except Exception as e:
            self.state_manager.rollback_transaction()
            logger.error(f"Error handling wait timeout: {traceback.format_exc()}")
            self._handle_vehicle_error(event, str(e))

    def _handle_dropoff_arrival(self, vehicle_id: str, route_stop: RouteStop, current_time: datetime) -> datetime:
        """Handle vehicle arrival for dropoff."""
        # Check for late arrival
        if self.context.current_time > route_stop.planned_arrival_time:
            delay = (self.context.current_time - route_stop.planned_arrival_time).total_seconds()
            self._handle_dropoff_delay_violation(
                vehicle_id,
                delay
            )
        
        # Get passengers to drop off at this stop
        dropoff_passengers = self.state_manager.passenger_worker.get_passengers_at_stop(
            PassengerStatus.IN_VEHICLE,
            route_stop.stop.id,
            StopPurpose.DROPOFF
        )
        
        # Create alighting completed events for each passenger
        current_alighting_time = self.context.current_time
        for passenger in dropoff_passengers:
            if passenger.assigned_vehicle_id == vehicle_id:
                event = Event(
                    event_type=EventType.PASSENGER_ALIGHTING_COMPLETED,
                    priority=EventPriority.HIGH,
                    timestamp=current_alighting_time + timedelta(seconds=self.config.vehicle.alighting_time),
                    passenger_id=passenger.id,
                    request_id=passenger.request_id,
                    vehicle_id=vehicle_id,
                    data={
                        'alighting_start_time': current_alighting_time,
                        'actual_dropoff_time': current_alighting_time
                    }
                )
                self.context.event_manager.publish_event(event)
                self.context.metrics_collector.log(
                    MetricName.VEHICLE_PASSENGERS_SERVED,
                    1,
                    self.context.current_time,
                    {
                        'vehicle_id': vehicle_id,
                        'passenger_id': passenger.id
                    }
                )
                # Update for next passenger's alighting time
                current_alighting_time += timedelta(seconds=self.config.vehicle.alighting_time)
                self.state_manager.vehicle_worker.decrement_vehicle_occupancy(vehicle_id)
        return current_alighting_time

    def _handle_pickup_delay_violation(
        self,
        vehicle_id: str,
        delay: float
    ) -> None:
        """Handle excessive pickup delay."""
        event = Event(
            event_type=EventType.VEHICLE_SERVICE_KPI_VIOLATION,
            priority=EventPriority.NORMAL,
            timestamp=self.context.current_time,
            vehicle_id=vehicle_id,
            data={
                'violation_type': 'pickup_delay',
                'delay': delay,
                'threshold': self.vehicle_thresholds['max_pickup_delay']
            }
        )
        self.context.event_manager.publish_event(event)

    def _handle_dropoff_delay_violation(
        self,
        vehicle_id: str,
        delay: float
    ) -> None:
        """Handle excessive dropoff delay."""
        event = Event(
            event_type=EventType.VEHICLE_SERVICE_KPI_VIOLATION,
            priority=EventPriority.HIGH,
            timestamp=self.context.current_time,
            vehicle_id=vehicle_id,
            data={
                'violation_type': 'dropoff_delay',
                'delay': delay,
                'threshold': self.vehicle_thresholds['max_dropoff_delay']
            }
        )
        self.context.event_manager.publish_event(event)

    def _create_vehicle_movement_event(
        self,
        vehicle_id: str,
        segment: RouteSegment,
        current_time: datetime
    ) -> None:
        """Create event for vehicle movement along a route segment."""
        # Get origin location - either from RouteStop or direct location
        origin_location = (
            segment.origin.stop.location if segment.origin 
            else segment.origin_location
        )
        
        # Get destination location - either from RouteStop or direct location
        destination_location = (
            segment.destination.stop.location if segment.destination 
            else segment.destination_location
        )
        
        # Get planned arrival time if destination is a RouteStop
        planned_arrival_time = (
            segment.destination.planned_arrival_time if segment.destination 
            else None
        )
        
        movement_data = {
            'segment_id': segment.id,
            'origin': origin_location,
            'destination': destination_location,
            'estimated_duration': segment.estimated_duration,
            'estimated_distance': segment.estimated_distance,
            'planned_arrival_time': planned_arrival_time,
            'movement_start_time': current_time,
            'expected_arrival_time': current_time + timedelta(seconds=segment.estimated_duration)
        }
        logger.info(f"Movement data: {movement_data}")
        event = Event(
            event_type=EventType.VEHICLE_EN_ROUTE,
            priority=EventPriority.HIGH,
            timestamp=current_time,
            vehicle_id=vehicle_id,
            data=movement_data
        )
        self.context.event_manager.publish_event(event)

    def _create_rebalancing_event(
        self,
        vehicle_id: str
    ) -> None:
        """Create event to rebalance vehicle to depot."""
        
        event = Event(
            event_type=EventType.VEHICLE_REBALANCING_REQUIRED,
            priority=EventPriority.HIGH,
            timestamp=self.context.current_time,
            vehicle_id=vehicle_id,
            data={
                'rebalancing_start_time': self.context.current_time
            }
        )
        self.context.event_manager.publish_event(event)

    def _handle_vehicle_error(self, event: Event, error_msg: str) -> None:
        """Handle errors in vehicle event processing."""
        logger.error(f"Error processing vehicle event {event.id}: {error_msg}")
        error_event = Event(
            event_type=EventType.SIMULATION_ERROR,
            priority=EventPriority.CRITICAL,
            timestamp=self.context.current_time,
            vehicle_id=event.vehicle_id,
            data={
                'error': error_msg,
                'original_event': event.to_dict(),
                'error_type': 'vehicle_processing_error'
            }
        )
        self.context.event_manager.publish_event(error_event)