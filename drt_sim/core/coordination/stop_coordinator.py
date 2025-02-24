from datetime import datetime, timedelta
from typing import Dict, Set, Optional, List
from drt_sim.models.event import Event, EventType, EventPriority
from drt_sim.models.location import Location
from drt_sim.core.state.manager import StateManager
from drt_sim.core.simulation.context import SimulationContext
from drt_sim.config.config import ParameterSet
from drt_sim.models.passenger import PassengerStatus
from drt_sim.models.route import RouteStop
from drt_sim.models.vehicle import VehicleStatus
from drt_sim.models.stop import StopPurpose
from drt_sim.core.state.workers.stop_coordination_worker import StopCoordinationState
from drt_sim.core.monitoring.types.metrics import MetricName
import logging

logger = logging.getLogger(__name__)

class StopCoordinator:
    def __init__(
        self,
        config: ParameterSet,
        context: SimulationContext,
        state_manager: StateManager
    ):
        self.config = config
        self.context = context
        self.state_manager = state_manager

    def _get_stop_state_with_route_stop_id(self, route_stop_id: str) -> Optional[StopCoordinationState]:
        """Get coordination state from RouteStop"""
        logger.debug(f"Retrieving coordination state for route stop {route_stop_id}")
        state = self.state_manager.stop_coordination_worker.get_stop_state(route_stop_id)
        return state
        
    def _get_stop_state(self, stop_id: str) -> Optional[StopCoordinationState]:
        """Get coordination state from RouteStop"""
        logger.debug(f"Retrieving coordination state for stop {stop_id}")
        
        # Find active route stop for this stop_id
        route_stop = None
        route = None
        # Find active route stop for this stop_id
        for r in self.state_manager.route_worker.get_active_routes():
            for rs in r.stops:
                if rs.stop.id == stop_id and not rs.completed:
                    route_stop = rs
                    route = r
                    break
            if route_stop:
                break
                
        if not route_stop:
            logger.warning(f"No active route stop found for stop {stop_id}")
            return None
            
        # First check if we have existing coordination state
        state = self.state_manager.stop_coordination_worker.get_stop_state(route_stop.id)
        if state:
            return state
            
        # Get vehicle state if vehicle exists
        vehicle_id = route.vehicle_id if route else None
        vehicle_is_at_stop = False
        vehicle_location = None
        
        if vehicle_id:
            vehicle = self.state_manager.vehicle_worker.get_vehicle(vehicle_id)
            if vehicle:
                vehicle_location = vehicle.current_state.current_location
                # Check if vehicle is at this specific stop based on vehicle state
                vehicle_is_at_stop = (
                    vehicle.current_state.status == VehicleStatus.AT_STOP and
                    vehicle.current_state.current_stop_id == stop_id
                )
            
        # Create new coordination state
        state = StopCoordinationState(
            route_stop_id=route_stop.id,
            expected_passenger_request_ids=set(route_stop.pickup_passengers),
            arrived_passenger_request_ids=set(route_stop.boarded_request_ids),
            vehicle_id=vehicle_id,
            vehicle_location=vehicle_location,
            vehicle_is_at_stop=vehicle_is_at_stop,
            wait_start_time=route_stop.wait_start_time,
            wait_timeout_event_id=route_stop.wait_timeout_event_id
        )
        
        # Store the new state
        self.state_manager.stop_coordination_worker.update_stop_state(state)
        logger.debug(f"Retrieved state: {state}")
        return state
        
    def _update_stop_state(self, state: StopCoordinationState) -> None:
        """Update RouteStop with coordination state"""
        logger.debug(f"Updating stop state: {state}")
        
        # Begin transaction
        self.state_manager.begin_transaction()
        try:
            # Update coordination state
            self.state_manager.stop_coordination_worker.update_stop_state(state)
            
            # Find and update route stop
            route_stop = None
            route = None
            
            # Find active route stop
            for r in self.state_manager.route_worker.get_active_routes():
                for rs in r.stops:
                    if rs.id == state.route_stop_id and not rs.completed:
                        route_stop = rs
                        route = r
                        break
                if route_stop:
                    break
                    
            if not route_stop or not route:
                logger.error(f"Failed to find route stop for state update: {state}")
                self.state_manager.rollback_transaction()
                return
                
            # Update route stop state
            route_stop.boarded_request_ids = list(state.arrived_passenger_request_ids)
            route_stop.wait_start_time = state.wait_start_time
            route_stop.wait_timeout_event_id = state.wait_timeout_event_id
            
            # Update route in state manager
            self.state_manager.route_worker.update_route(route)
            
            # Commit transaction
            self.state_manager.commit_transaction()
            logger.debug(f"Successfully updated route stop state for stop {state.route_stop_id}")
            
        except Exception as e:
            logger.error(f"Error updating stop state: {e}")
            self.state_manager.rollback_transaction()
            raise

    def _cleanup_stop_state(self, state: StopCoordinationState) -> None:
        """Clean up stop coordination state after vehicle completes all operations at stop"""
        logger.info(f"Cleaning up stop state for route stop {state.route_stop_id}")
        
        # Remove the stop state from the worker
        self.state_manager.stop_coordination_worker.remove_stop_state(state.route_stop_id)

    def register_passenger_arrival(
        self,
        stop_id: str,
        passenger_id: str,
        request_id: str,
        arrival_time: datetime,
        location: Location
    ) -> None:
        """Register passenger arrival at stop and check boarding conditions"""
        
        logger.info(f"Registering passenger arrival - Stop: {stop_id}, Passenger: {passenger_id}, Request: {request_id}")
        
        state = self._get_stop_state(stop_id)
        logger.debug(f"Stop state retrieved for stop {stop_id}, passenger_id {passenger_id}, request_id {request_id}")
        if not state:
            logger.error(f"Cannot register passenger arrival - stop {stop_id} not found in active routes")
            return
            
        state.arrived_passenger_request_ids.add(request_id)
        
        # Get passenger and stop assignment for walking metrics
        passenger = self.state_manager.passenger_worker.get_passenger(passenger_id)
        stop_assignment = self.state_manager.stop_assignment_worker.get_assignment_for_request(request_id)
        request = self.state_manager.request_worker.get_request(request_id)
        
        if passenger and stop_assignment:
            # Log walking time to origin stop
            self.context.metrics_collector.log(
                MetricName.PASSENGER_WALK_TIME_TO_ORIGIN_STOP,
                stop_assignment.walking_time_origin,
                arrival_time,
                {
                    'passenger_id': passenger_id,
                    'request_id': request_id,
                    'origin': request.origin.to_dict(),
                    'origin_stop': stop_assignment.origin_stop.location.to_dict()
                }
            )
        
        # Check if vehicle is already at stop and waiting
        vehicle_present = self._is_vehicle_at_stop(state)
        logger.debug(f"Vehicle present at stop: {vehicle_present}")
        
        # Update passenger state to WAITING_FOR_VEHICLE if vehicle not present
        if not vehicle_present:
            logger.info(f"Vehicle not present, updating passenger {passenger_id} status to WAITING_FOR_VEHICLE")
            self.state_manager.passenger_worker.update_passenger_status(
                passenger_id,
                PassengerStatus.WAITING_FOR_VEHICLE,
                arrival_time,
                {
                    'waiting_start_time': arrival_time
                }
            )

            
        # Update state in route stop
        self._update_stop_state(state)
        
        # Check if we can start boarding
        self._check_and_trigger_boarding(state)

    def register_vehicle_arrival(
        self,
        stop_id: str,
        vehicle_id: str,
        arrival_time: datetime,
        location: Location,
        event: Event,
        segment_id: Optional[str] = None
    ) -> None:
        """Register vehicle arrival at stop and check boarding conditions"""
        logger.info(f"Registering vehicle arrival - Stop: {stop_id}, Vehicle: {vehicle_id}")
        
        state = self._get_stop_state(stop_id)
        if not state:
            logger.error(f"Cannot register vehicle arrival - stop {stop_id} not found in active routes")
            return
        
        movement_start_time = event.data.get('movement_start_time')
        actual_distance = event.data.get('actual_distance')
        state.movement_start_time = movement_start_time
        state.actual_distance = actual_distance
        state.segment_id = segment_id
        # Get route stop to check planned arrival time
        route_stop = None
        for route in self.state_manager.route_worker.get_active_routes():
            for rs in route.stops:
                if rs.id == state.route_stop_id and not rs.completed:
                    route_stop = rs
                    break
            if route_stop:
                break
                
        # Check for delay violations based on stop purpose
        if route_stop and route_stop.planned_arrival_time:
            delay = (arrival_time - route_stop.planned_arrival_time).total_seconds()
            
            # Check for pickup delay if this is a pickup stop
            if route_stop.pickup_passengers:
                if delay > self.config.vehicle.max_pickup_delay:
                    logger.warning(f"Pickup delay violation: Vehicle {vehicle_id} arrived {delay:.1f}s late at stop {stop_id}")
                    self._create_pickup_delay_violation_event(vehicle_id, delay)
                    
            # Check for dropoff delay if this is a dropoff stop
            if route_stop.dropoff_passengers:
                if delay > self.config.vehicle.max_dropoff_delay:
                    logger.warning(f"Dropoff delay violation: Vehicle {vehicle_id} arrived {delay:.1f}s late at stop {stop_id}")
                    self._create_dropoff_delay_violation_event(vehicle_id, delay)
            
        state.vehicle_id = vehicle_id
        state.vehicle_location = location
        state.vehicle_is_at_stop = True

        # Handle alighting if this is a dropoff stop
        if route_stop and route_stop.dropoff_passengers:
            self._handle_passenger_alighting(state, route_stop)

        vehicle_route = self.state_manager.vehicle_worker.get_vehicle_active_route_id(state.vehicle_id)
        route = self.state_manager.route_worker.get_route(vehicle_route)
        rs = None
        for r in route.stops:
            if r.id == state.route_stop_id:
                rs = r
                break
        if rs:
            state.expected_passenger_request_ids = set(rs.pickup_passengers)
        
        # Start wait timer if passengers are missing for pickup
        if len(state.arrived_passenger_request_ids) < len(state.expected_passenger_request_ids):
            logger.info(f"Starting wait timer for {len(state.expected_passenger_request_ids) - len(state.arrived_passenger_request_ids)} missing passengers")
            self._start_wait_timer(state)
        
        # Check if we can start boarding
        self._check_and_trigger_boarding(state)
        # Update state in route stop
        self._update_stop_state(state)

    def _is_vehicle_at_stop(self, state: StopCoordinationState) -> bool:
        """Check if vehicle is at stop based on state"""
        if not state.vehicle_id:
            logger.debug(f"Vehicle check failed - no vehicle assigned to stop {state.route_stop_id}")
            return False
            
        # Get current vehicle state
        vehicle = self.state_manager.vehicle_worker.get_vehicle(state.vehicle_id)
        if not vehicle:
            logger.error(f"Vehicle {state.vehicle_id} not found")
            return False
            
        # Get route stop to check physical stop ID
        route_stop = None
        for route in self.state_manager.route_worker.get_active_routes():
            for rs in route.stops:
                if rs.id == state.route_stop_id and not rs.completed:
                    route_stop = rs
                    break
            if route_stop:
                break
                
        if not route_stop:
            logger.error(f"Route stop {state.route_stop_id} not found")
            return False
            
        # Check if vehicle is at this specific stop
        is_at_stop = (
            vehicle.current_state.status == VehicleStatus.AT_STOP and
            vehicle.current_state.current_stop_id == route_stop.stop.id
        )
        
        if not is_at_stop:
            logger.debug(f"Vehicle {state.vehicle_id} is not at stop {state.route_stop_id}")
            
        return is_at_stop

    def _check_and_trigger_boarding(self, state: StopCoordinationState) -> None:
        """Check if conditions are met to start boarding process"""
        logger.debug(f"Checking boarding conditions for stop {state.route_stop_id}")
        
        # Must have vehicle and at least one passenger
        if not state.vehicle_id or not state.arrived_passenger_request_ids:
            logger.debug("Boarding check failed - missing vehicle or no arrived passengers")
            return
            
        # Verify vehicle is at stop
        if not self._is_vehicle_at_stop(state):
            logger.debug("Boarding check failed - vehicle not at stop")
            return
            
        # Cancel any existing wait timeout
        if state.wait_timeout_event_id:
            logger.info(f"Cancelling wait timeout event {state.wait_timeout_event_id}")
            self.context.event_manager.cancel_event(state.wait_timeout_event_id)
            state.wait_timeout_event_id = None
            state.wait_start_time = None
            self._update_stop_state(state)
        
        # Create boarding events for present passengers
        current_time = self.context.current_time
        last_boarding_time = current_time
        for request_id in state.arrived_passenger_request_ids:
            passengers = self.state_manager.passenger_worker.get_all_passenger_ids_for_request_ids([request_id])
            logger.info(f"Creating boarding events for request {request_id} with {len(passengers)} passengers")
            for passenger_id in passengers:
                # Log passenger wait time
                passenger = self.state_manager.passenger_worker.get_passenger(passenger_id)
                if passenger and passenger.waiting_start_time:
                    wait_time = (current_time - passenger.waiting_start_time).total_seconds()
                    self.context.metrics_collector.log(
                        MetricName.PASSENGER_WAIT_TIME,
                        wait_time,
                        current_time,
                        {
                            'passenger_id': passenger_id,
                            'vehicle_id': state.vehicle_id,
                            'request_id': request_id,
                            'stop_id': state.route_stop_id
                        }
                    )
                
                # Increment vehicle occupancy for each boarding passenger
                self.state_manager.vehicle_worker.increment_vehicle_occupancy(state.vehicle_id)
                boarding_time = current_time + timedelta(seconds=self.config.vehicle.boarding_time)
                last_boarding_time = max(last_boarding_time, boarding_time)
                self._create_boarding_event(
                    passenger_id,
                    state.vehicle_id,
                    current_time
                )
        
        # Log vehicle wait time if applicable
        if state.wait_start_time:
            vehicle_wait_time = (current_time - state.wait_start_time).total_seconds()
            self.context.metrics_collector.log(
                MetricName.VEHICLE_WAIT_TIME,
                vehicle_wait_time,
                current_time,
                {
                    'vehicle_id': state.vehicle_id,
                    'stop_id': state.route_stop_id
                }
            )

        vehicle_route = self.state_manager.vehicle_worker.get_vehicle_active_route_id(state.vehicle_id)
        route = self.state_manager.route_worker.get_route(vehicle_route)
        rs = None
        for r in route.stops:
            if r.id == state.route_stop_id:
                rs = r
                break
        if rs:
            state.expected_passenger_request_ids = set(rs.pickup_passengers)
        
        # Clear vehicle waiting state if all expected passengers are present
        if len(state.arrived_passenger_request_ids) == len(state.expected_passenger_request_ids):
            logger.info(f"All passengers present, clearing waiting state for vehicle {state.vehicle_id}")
            # First update the vehicle status
            self.state_manager.vehicle_worker.update_vehicle_status(
                state.vehicle_id,
                VehicleStatus.IN_SERVICE,
                self.context.current_time
            )
            # Then update the vehicle state
            vehicle = self.state_manager.vehicle_worker.get_vehicle(state.vehicle_id)
            if vehicle:
                vehicle.current_state.waiting_for_passengers = False
                vehicle.current_state.current_stop_id = state.route_stop_id  # Keep the stop ID until we actually leave
                self.state_manager.vehicle_worker.update_vehicle(vehicle)

                # Get route information for completion event
                active_route_id = self.state_manager.vehicle_worker.get_vehicle_active_route_id(vehicle.id)
                route = self.state_manager.route_worker.get_route(active_route_id)
                if route:
                    current_segment = route.get_current_segment()
                    if current_segment:
                        # Create stop operations completion event after last boarding
                        completion_event = Event(
                            event_type=EventType.VEHICLE_STOP_OPERATIONS_COMPLETED,
                            priority=EventPriority.HIGH,
                            timestamp=last_boarding_time,
                            vehicle_id=state.vehicle_id,
                            data={
                                'route_id': active_route_id,
                                'segment_id': state.segment_id,
                                'stop_id': state.route_stop_id,
                                'operation_type': 'boarding',
                                'movement_start_time': state.movement_start_time,
                                'actual_distance': state.actual_distance
                            }
                        )
                        self.context.event_manager.publish_event(completion_event)
                        self._cleanup_stop_state(state)
        return state

    def _start_wait_timer(self, state: StopCoordinationState) -> None:
        """Start wait timer for remaining passengers"""
        # Don't start new timer if one exists
        if state.wait_timeout_event_id:
            logger.debug(f"Wait timer already exists for stop {state.route_stop_id}")
            return
            
        # Only start timer if vehicle is present
        if not self._is_vehicle_at_stop(state):
            logger.debug(f"Cannot start wait timer - vehicle not at stop {state.route_stop_id}")
            return
            
        wait_duration = self.config.vehicle.max_dwell_time
        logger.info(f"Starting {wait_duration}s wait timer for stop {state.route_stop_id}")
        
        wait_event = Event(
            event_type=EventType.VEHICLE_WAIT_TIMEOUT,
            priority=EventPriority.HIGH,
            timestamp=self.context.current_time + timedelta(seconds=wait_duration),
            data={
                'route_stop_id': state.route_stop_id,
                'missing_requests': list(
                    state.expected_passenger_request_ids - state.arrived_passenger_request_ids
                ),
                'wait_start_time': self.context.current_time
            }
        )
        
        state.wait_timeout_event_id = wait_event.id
        state.wait_start_time = self.context.current_time
        self._update_stop_state(state)
        
        logger.debug(f"Publishing wait timeout event {wait_event.id} for route stop {state.route_stop_id} at {self.context.current_time}")
        self.context.event_manager.publish_event(wait_event)

    def handle_wait_timeout(self, event: Event) -> None:
        """Handle wait timeout for missing passengers"""
        route_stop_id = event.data['route_stop_id']
        logger.info(f"Handling wait timeout for route stop {route_stop_id} at {self.context.current_time}")
        
        state = self._get_stop_state_with_route_stop_id(route_stop_id)
        if not state:
            logger.error(f"Cannot handle wait timeout - route stop {route_stop_id} not found")
            return
            
        # Create no-show events for missing passengers
        missing_requests = state.expected_passenger_request_ids - state.arrived_passenger_request_ids
        logger.info(f"Creating no-show events for {len(missing_requests)} missing requests")
        
        for request_id in missing_requests:
            # Get passenger_id from request
            passengers = self.state_manager.passenger_worker.get_all_passenger_ids_for_request_ids([request_id])
            for passenger_id in passengers:
                logger.debug(f"Creating no-show event for passenger {passenger_id}")
                self._create_no_show_event(passenger_id, route_stop_id)
            
        # Clear vehicle waiting state
        if state.vehicle_id:
            logger.info(f"Clearing waiting state for vehicle {state.vehicle_id}")
            # First update the vehicle status
            self.state_manager.vehicle_worker.update_vehicle_status(
                state.vehicle_id,
                VehicleStatus.IN_SERVICE,
                self.context.current_time
            )
            # Then update the vehicle state
            vehicle = self.state_manager.vehicle_worker.get_vehicle(state.vehicle_id)
            if vehicle:
                vehicle.current_state.waiting_for_passengers = False
                vehicle.current_state.current_stop_id = state.route_stop_id  # Keep the stop ID until we actually leave
                self.state_manager.vehicle_worker.update_vehicle(vehicle)
                
                # Get vehicle's current route and create stop operations completion event
                active_route_id = self.state_manager.vehicle_worker.get_vehicle_active_route_id(vehicle.id)
                if active_route_id:
                    completion_event = Event(
                        event_type=EventType.VEHICLE_STOP_OPERATIONS_COMPLETED,
                        priority=EventPriority.HIGH,
                        timestamp=self.context.current_time,
                        vehicle_id=state.vehicle_id,
                        data={
                            'route_id': active_route_id,
                            'segment_id': state.segment_id,
                            'stop_id': state.route_stop_id,
                            'operation_type': 'wait_timeout',
                            'movement_start_time': state.movement_start_time,
                            'actual_distance': state.actual_distance
                        }
                    )
                    self.context.event_manager.publish_event(completion_event)

    def _create_boarding_event(
        self,
        passenger_id: str,
        vehicle_id: str,
        current_time: datetime
    ) -> None:
        """Create boarding event for passenger"""
        logger.debug(f"Creating boarding event - Passenger: {passenger_id}, Vehicle: {vehicle_id}")
        
        passenger = self.state_manager.passenger_worker.get_passenger(passenger_id)
        if not passenger:
            logger.error(f"Cannot create boarding event - passenger {passenger_id} not found")
            return
            
        event = Event(
            event_type=EventType.PASSENGER_BOARDING_COMPLETED,
            priority=EventPriority.HIGH,
            timestamp=current_time + timedelta(
                seconds=self.config.vehicle.boarding_time
            ),
            passenger_id=passenger_id,
            request_id=passenger.request_id,
            vehicle_id=vehicle_id,
            data={
                'boarding_start_time': current_time,
                'actual_pickup_time': current_time
            }
        )
        logger.debug(f"Publishing boarding event {event.id}")
        self.context.event_manager.publish_event(event)

    def _create_no_show_event(self, passenger_id: str, stop_id: str) -> None:
        """Create no-show event for passenger"""
        logger.info(f"Creating no-show event - Passenger: {passenger_id}, Stop: {stop_id}")
        event = Event(
            event_type=EventType.PASSENGER_NO_SHOW,
            priority=EventPriority.HIGH,
            timestamp=self.context.current_time,
            passenger_id=passenger_id,
            data={'stop_id': stop_id}
        )
        logger.debug(f"Publishing no-show event {event.id}")
        self.context.event_manager.publish_event(event)

    def _create_pickup_delay_violation_event(
        self,
        vehicle_id: str,
        delay: float
    ) -> None:
        """Create event for pickup delay violation"""
        event = Event(
            event_type=EventType.VEHICLE_SERVICE_KPI_VIOLATION,
            priority=EventPriority.HIGH,
            timestamp=self.context.current_time,
            vehicle_id=vehicle_id,
            data={
                'violation_type': 'pickup_delay',
                'delay': delay,
                'threshold': self.config.vehicle.max_pickup_delay
            }
        )
        logger.debug(f"Publishing pickup delay violation event for vehicle {vehicle_id}")
        self.context.event_manager.publish_event(event)

    def _create_dropoff_delay_violation_event(
        self,
        vehicle_id: str,
        delay: float
    ) -> None:
        """Create event for dropoff delay violation"""
        event = Event(
            event_type=EventType.VEHICLE_SERVICE_KPI_VIOLATION,
            priority=EventPriority.HIGH,
            timestamp=self.context.current_time,
            vehicle_id=vehicle_id,
            data={
                'violation_type': 'dropoff_delay',
                'delay': delay,
                'threshold': self.config.vehicle.max_dropoff_delay
            }
        )
        logger.debug(f"Publishing dropoff delay violation event for vehicle {vehicle_id}")
        self.context.event_manager.publish_event(event)

    def _handle_passenger_alighting(self, state: StopCoordinationState, route_stop: RouteStop) -> None:
        """Handle alighting process for passengers at stop"""
        logger.debug(f"Processing alighting at stop {state.route_stop_id}")

        # Get passengers to alight at this stop
        alighting_passengers = self.state_manager.passenger_worker.get_passengers_at_stop(
            PassengerStatus.IN_VEHICLE,
            route_stop.stop.id,
            StopPurpose.DROPOFF
        )

        if not alighting_passengers:
            logger.debug(f"No passengers to alight at stop {state.route_stop_id}")
            # If this is a dropoff-only stop with no passengers to alight, trigger next operation
            if not route_stop.pickup_passengers:
                # Get vehicle's current route
                vehicle = self.state_manager.vehicle_worker.get_vehicle(state.vehicle_id)
                if vehicle:
                    active_route_id = self.state_manager.vehicle_worker.get_vehicle_active_route_id(vehicle.id)
                    if active_route_id:
                        # Create stop operations completion event immediately since no alighting needed
                        completion_event = Event(
                            event_type=EventType.VEHICLE_STOP_OPERATIONS_COMPLETED,
                            priority=EventPriority.HIGH,
                            timestamp=self.context.current_time,
                            vehicle_id=state.vehicle_id,
                            data={
                                'route_id': active_route_id,
                                'segment_id': state.segment_id,
                                'stop_id': state.route_stop_id,
                                'operation_type': 'alighting',
                                'movement_start_time': state.movement_start_time,
                                'actual_distance': state.actual_distance
                            }
                        )
                        self.context.event_manager.publish_event(completion_event)
            return

        logger.info(f"Processing alighting for {len(alighting_passengers)} passengers at stop {state.route_stop_id}")

        # Create alighting events for each passenger
        current_time = self.context.current_time
        last_alighting_time = current_time
        for passenger in alighting_passengers:
            if passenger.assigned_vehicle_id == state.vehicle_id:
                # Create alighting event
                alighting_time = current_time + timedelta(seconds=self.config.vehicle.alighting_time)
                last_alighting_time = max(last_alighting_time, alighting_time)
                event = Event(
                    event_type=EventType.PASSENGER_ALIGHTING_COMPLETED,
                    priority=EventPriority.HIGH,
                    timestamp=alighting_time,
                    passenger_id=passenger.id,
                    request_id=passenger.request_id,
                    vehicle_id=state.vehicle_id,
                    data={
                        'alighting_start_time': current_time,
                        'actual_dropoff_time': current_time
                    }
                )
                self.context.event_manager.publish_event(event)
                
                # Update metrics
                self.context.metrics_collector.log(
                    MetricName.VEHICLE_PASSENGERS_SERVED,
                    1,
                    current_time,
                    {
                        'vehicle_id': state.vehicle_id,
                        'passenger_id': passenger.id,
                        'stop_id': state.route_stop_id
                    }
                )

                # Decrement vehicle occupancy
                self.state_manager.vehicle_worker.decrement_vehicle_occupancy(state.vehicle_id)
                
                # Update for next passenger's alighting time
                current_time += timedelta(seconds=self.config.vehicle.alighting_time)

        # If this is a dropoff-only stop, create completion event after last alighting
        if not route_stop.pickup_passengers:
            # Get vehicle's current route
            vehicle = self.state_manager.vehicle_worker.get_vehicle(state.vehicle_id)
            if vehicle:
                active_route_id = self.state_manager.vehicle_worker.get_vehicle_active_route_id(vehicle.id)
                if active_route_id:
                    # Create stop operations completion event after last alighting
                    completion_event = Event(
                        event_type=EventType.VEHICLE_STOP_OPERATIONS_COMPLETED,
                        priority=EventPriority.HIGH,
                        timestamp=last_alighting_time,
                        vehicle_id=state.vehicle_id,
                        data={
                            'route_id': active_route_id,
                            'segment_id': state.segment_id,
                            'stop_id': state.route_stop_id,
                            'operation_type': 'alighting',
                            'movement_start_time': state.movement_start_time,
                            'actual_distance': state.actual_distance
                        }
                    )
                    self.context.event_manager.publish_event(completion_event)