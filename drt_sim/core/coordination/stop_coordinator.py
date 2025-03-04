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
        # Add route stop lookup cache
        self._route_stop_cache = {}  # Maps route_stop_id to (route, route_stop) tuples
        
    def _find_route_stop(self, route_stop_id: str) -> Optional[tuple]:
        """Find route and route stop by route_stop_id, using cache when available."""
        # Check cache first
        if route_stop_id in self._route_stop_cache:
            route, route_stop = self._route_stop_cache[route_stop_id]
            # Verify the cached route stop is still active
            if not route_stop.completed:
                return route, route_stop
                
        # Cache miss or stale cache, look up route stop
        for route in self.state_manager.route_worker.get_active_routes():
            for rs in route.stops:
                if rs.id == route_stop_id and not rs.completed:
                    # Update cache and return
                    self._route_stop_cache[route_stop_id] = (route, rs)
                    return route, rs
                    
        return None, None
        
    def _find_route_stop_by_stop_id(self, stop_id: str) -> Optional[tuple]:
        """Find route and route stop by physical stop_id."""
        for route in self.state_manager.route_worker.get_active_routes():
            for rs in route.stops:
                if rs.stop.id == stop_id and not rs.completed:
                    # Update route stop cache
                    self._route_stop_cache[rs.id] = (route, rs)
                    return route, rs
                    
        return None, None
        
    def _find_route_stop_by_stop_and_vehicle_id(self, stop_id: str, vehicle_id: str) -> Optional[tuple]:
        """Find route and route stop by physical stop_id and vehicle_id."""
        for route in self.state_manager.route_worker.get_active_routes():
            # Only check routes for the specified vehicle
            if route.vehicle_id == vehicle_id:
                for rs in route.stops:
                    if rs.stop.id == stop_id and not rs.completed:
                        # Update route stop cache
                        self._route_stop_cache[rs.id] = (route, rs)
                        return route, rs
                        
        return None, None
        
    def _find_route_stop_for_request(self, stop_id: str, request_id: str) -> Optional[tuple]:
        """Find route and route stop that expects a specific request at a specific stop."""
        for route in self.state_manager.route_worker.get_active_routes():
            for rs in route.stops:
                if rs.stop.id == stop_id and not rs.completed and request_id in rs.pickup_passengers:
                    # Update route stop cache
                    self._route_stop_cache[rs.id] = (route, rs)
                    return route, rs
                    
        return None, None

    def _get_or_create_stop_state(self, route_stop_id: str) -> Optional[StopCoordinationState]:
        """Get existing stop state or create a new one if needed."""
        # Try to get existing state
        state = self.state_manager.stop_coordination_worker.get_stop_state(route_stop_id)
        if state:
            return self._sync_state_with_route_stop(state)
            
        # State doesn't exist, create new state
        route, route_stop = self._find_route_stop(route_stop_id)
        if not route_stop:
            logger.error(f"Cannot create stop state: Route stop {route_stop_id} not found")
            return None
            
        # Determine vehicle state if applicable
        vehicle_id = route.vehicle_id if route else None
        vehicle_is_at_stop = False
        vehicle_location = None
        
        if vehicle_id:
            vehicle = self.state_manager.vehicle_worker.get_vehicle(vehicle_id)
            if vehicle:
                vehicle_location = vehicle.current_state.current_location
                vehicle_is_at_stop = (
                    vehicle.current_state.status == VehicleStatus.AT_STOP and
                    vehicle.current_state.current_stop_id == route_stop.stop.id
                )
            
        # Create new coordination state
        state = StopCoordinationState(
            route_stop_id=route_stop.id,
            expected_pickup_request_ids=set(route_stop.pickup_passengers),
            expected_dropoff_request_ids=set(route_stop.dropoff_passengers),
            arrived_pickup_request_ids=set(),
            boarded_request_ids=set(),
            completed_dropoff_request_ids=set(),
            vehicle_id=vehicle_id,
            vehicle_location=vehicle_location,
            vehicle_is_at_stop=vehicle_is_at_stop,
            wait_start_time=route_stop.wait_start_time,
            wait_timeout_event_id=route_stop.wait_timeout_event_id
        )
        
        # Store and return new state
        self.state_manager.begin_transaction()
        try:
            self.state_manager.stop_coordination_worker.update_stop_state(state)
            self.state_manager.commit_transaction()
            logger.debug(f"Created new stop coordination state: {state}")
            return state
        except Exception as e:
            logger.error(f"Error creating stop state: {e}")
            self.state_manager.rollback_transaction()
            return None
            
    def _sync_state_with_route_stop(self, state: StopCoordinationState) -> StopCoordinationState:
        """Sync stop coordination state with current route stop state."""
        route, route_stop = self._find_route_stop(state.route_stop_id)
        if not route_stop:
            logger.warning(f"Cannot sync state: Route stop {state.route_stop_id} not found")
            return state
            
        # Update expected request IDs
        state.expected_pickup_request_ids = set(route_stop.pickup_passengers)
        state.expected_dropoff_request_ids = set(route_stop.dropoff_passengers)
        
        # Update boarded request IDs from route_stop
        if route_stop.boarded_request_ids:
            # Check for any discrepancies
            route_boarded = set(route_stop.boarded_request_ids)
            state_boarded = state.boarded_request_ids
            
            # Identify discrepancies
            missing_in_state = route_boarded - state_boarded
            missing_in_route = state_boarded - route_boarded
            
            # Log and fix discrepancies
            if missing_in_state:
                logger.warning(f"Found {len(missing_in_state)} request(s) boarded in route but not in state - fixing")
                state.boarded_request_ids.update(missing_in_state)
            
            if missing_in_route:
                logger.warning(f"Found {len(missing_in_route)} request(s) boarded in state but not in route - fixing")
                for request_id in missing_in_route:
                    if request_id not in route_stop.boarded_request_ids:
                        route_stop.boarded_request_ids.append(request_id)
                # Update the route
                self.state_manager.route_worker.update_route(route)
        
        # Update vehicle info if needed
        if route.vehicle_id and state.vehicle_id != route.vehicle_id:
            state.vehicle_id = route.vehicle_id
            
        # Check if the stop is completed in the route but not in the state
        if route_stop.completed:
            # If the stop is completed, all expected pickups should be boarded
            # and all expected dropoffs should be completed
            state.boarded_request_ids.update(state.expected_pickup_request_ids)
            state.completed_dropoff_request_ids.update(state.expected_dropoff_request_ids)
            
        return state

    def _update_stop_state_with_transaction(self, state: StopCoordinationState) -> bool:
        """Update stop state and related route data within a transaction."""
        self.state_manager.begin_transaction()
        try:
            # Find relevant route and route stop
            route, route_stop = self._find_route_stop(state.route_stop_id)
            if not route_stop:
                logger.error(f"Cannot update state: Route stop {state.route_stop_id} not found")
                self.state_manager.rollback_transaction()
                return False
                
            # Sync state with route stop
            state = self._sync_state_with_route_stop(state)
            
            # Update the coordination state
            self.state_manager.stop_coordination_worker.update_stop_state(state)
                
            # Update the route stop state
            route_stop.boarded_request_ids = list(state.boarded_request_ids)
            route_stop.wait_start_time = state.wait_start_time
            route_stop.wait_timeout_event_id = state.wait_timeout_event_id
            
            self.state_manager.route_worker.update_route(route)
            self.state_manager.commit_transaction()
            logger.debug(
                f"Updated state for stop {state.route_stop_id}. "
                f"Expected pickups: {state.expected_pickup_request_ids}, "
                f"Expected dropoffs: {state.expected_dropoff_request_ids}, "
                f"Arrived: {state.arrived_pickup_request_ids}, "
                f"Boarded: {state.boarded_request_ids}"
            )
            return True
            
        except Exception as e:
            logger.error(f"Error updating state for route_stop_id {state.route_stop_id}: {e}")
            self.state_manager.rollback_transaction()
            return False

    def register_passenger_arrival(
        self,
        stop_id: str,
        passenger_id: str,
        request_id: str,
        arrival_time: datetime,
        location: Location
    ) -> None:
        logger.info(f"Passenger {passenger_id} (Request {request_id}) arrived at stop {stop_id} at {arrival_time}")
        
        # First verify the stop assignment
        stop_assignment = self.state_manager.stop_assignment_worker.get_assignment_for_request(request_id)
        if not stop_assignment or stop_assignment.origin_stop.id != stop_id:
            logger.error(
                f"Invalid stop assignment for request {request_id}. "
                f"Assigned stop: {stop_assignment.origin_stop.id if stop_assignment else 'None'}, "
                f"Actual arrival stop: {stop_id}"
            )
            return
            
        # Find the specific route stop that expects this request
        route, route_stop = self._find_route_stop_for_request(stop_id, request_id)
                
        if not route_stop:
            logger.error(f"No route stop found expecting request {request_id} at stop {stop_id}")
            return
            
        # Get or create coordination state for this route stop
        state = self._get_or_create_stop_state(route_stop.id)
        if not state:
            logger.error(f"Failed to get or create coordination state for route stop {route_stop.id}")
            return
            
        # Only register arrival if passenger is expected and hasn't already boarded
        if request_id not in state.expected_pickup_request_ids:
            logger.warning(f"Unexpected arrival - Passenger {passenger_id} (Request {request_id}) not expected at stop {stop_id}")
            
            # Check if this request should be expected at this stop
            if request_id in route_stop.pickup_passengers:
                logger.warning(f"Request {request_id} is in route_stop.pickup_passengers but not in state.expected_pickup_request_ids - fixing")
                state.expected_pickup_request_ids.add(request_id)
                self._update_stop_state_with_transaction(state)
            else:
                return
            
        if request_id in state.boarded_request_ids:
            logger.warning(f"Passenger {passenger_id} (Request {request_id}) has already boarded at stop {stop_id}")
            return
            
        # Update arrival state
        state.arrived_pickup_request_ids.add(request_id)
        if not self._update_stop_state_with_transaction(state):
            return
            
        # Log metrics
        passenger = self.state_manager.passenger_worker.get_passenger(passenger_id)
        request = self.state_manager.request_worker.get_request(request_id)
        
        if passenger and stop_assignment:
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
        
        # Check if vehicle is present for immediate boarding
        vehicle_present = self._is_vehicle_at_stop(state)
        logger.debug(f"Vehicle present at stop {stop_id}: {vehicle_present}")
        
        if vehicle_present:
            self._check_and_trigger_boarding(state)
        else:
            logger.info(f"Setting passenger {passenger_id} status to WAITING_FOR_VEHICLE at stop {stop_id}")
            self.state_manager.passenger_worker.update_passenger_status(
                passenger_id,
                PassengerStatus.WAITING_FOR_VEHICLE,
                arrival_time,
                {'waiting_start_time': arrival_time}
            )

    def register_vehicle_arrival(
        self,
        stop_id: str,
        vehicle_id: str,
        arrival_time: datetime,
        location: Location,
        event: Event,
        segment_id: Optional[str] = None
    ) -> None:
        logger.info(f"Vehicle {vehicle_id} arrived at stop {stop_id} at {arrival_time}")
        
        # Find the route stop for this vehicle at this stop
        route, route_stop = self._find_route_stop_by_stop_and_vehicle_id(stop_id, vehicle_id)
        if not route_stop:
            logger.error(f"No active route stop found for vehicle {vehicle_id} at stop_id: {stop_id}")
            return
            
        # Get or create coordination state
        state = self._get_or_create_stop_state(route_stop.id)
        if not state:
            logger.error(f"Failed to get or create coordination state for route stop {route_stop.id}")
            return

        if state.last_vehicle_arrival_time and state.last_vehicle_arrival_time == arrival_time:
            logger.warning(f"Duplicate arrival event ignored for vehicle {vehicle_id} at stop {stop_id}")
            return
            
        # Update state with vehicle information
        state.last_vehicle_arrival_time = arrival_time
        state.movement_start_time = event.data.get('movement_start_time')
        state.actual_distance = event.data.get('actual_distance')
        state.segment_id = segment_id
        state.vehicle_id = vehicle_id
        state.vehicle_location = location
        state.vehicle_is_at_stop = True
        
        # Check for delay violations
        if route_stop.planned_arrival_time:
            delay = (arrival_time - route_stop.planned_arrival_time).total_seconds()
            if route_stop.pickup_passengers and delay > self.config.vehicle.max_pickup_delay:
                logger.warning(f"Pickup delay violation: Vehicle {vehicle_id} delayed by {delay:.1f}s at stop {stop_id}")
                self._create_pickup_delay_violation_event(vehicle_id, delay)
            if route_stop.dropoff_passengers and delay > self.config.vehicle.max_dropoff_delay:
                logger.warning(f"Dropoff delay violation: Vehicle {vehicle_id} delayed by {delay:.1f}s at stop {stop_id}")
                self._create_dropoff_delay_violation_event(vehicle_id, delay)
        
        # Update state
        if not self._update_stop_state_with_transaction(state):
            return
            
        # Handle passenger alighting first
        if route_stop.dropoff_passengers:
            self._handle_passenger_alighting(state, route_stop)

        # Start wait timer if needed and trigger boarding
        missing_pickups = len(state.expected_pickup_request_ids) - len(state.arrived_pickup_request_ids)
        if missing_pickups > 0:
            logger.info(f"Starting wait timer for {missing_pickups} missing passenger(s) at stop {stop_id}")
            self._start_wait_timer(state)
        
        self._check_and_trigger_boarding(state)

    def _is_vehicle_at_stop(self, state: StopCoordinationState) -> bool:
        """Check if vehicle is currently at the stop."""
        if not state.vehicle_id:
            return False
            
        vehicle = self.state_manager.vehicle_worker.get_vehicle(state.vehicle_id)
        if not vehicle:
            return False
            
        route, route_stop = self._find_route_stop(state.route_stop_id)
        if not route_stop:
            return False
            
        return (
            vehicle.current_state.status == VehicleStatus.AT_STOP and
            vehicle.current_state.current_stop_id == route_stop.stop.id
        )

    def _check_and_trigger_boarding(self, state: StopCoordinationState) -> None:
        """Check if conditions are met for passenger boarding and trigger boarding events."""
        if not state.vehicle_id:
            return
            
        pending_arrivals = state.arrived_pickup_request_ids - state.boarded_request_ids
        if not pending_arrivals:
            return
            
        if not self._is_vehicle_at_stop(state):
            return
            
        # Get the route associated with this stop to ensure proper synchronization
        route, route_stop = self._find_route_stop(state.route_stop_id)
        if not route or not route_stop:
            logger.error(f"Could not find route or route_stop for route_stop_id {state.route_stop_id}")
            return
            
        # Cancel wait timer if active
        if state.wait_timeout_event_id:
            logger.info(f"Cancelling wait timeout event {state.wait_timeout_event_id}")
            self.context.event_manager.cancel_event(state.wait_timeout_event_id)
            state.wait_timeout_event_id = None
            state.wait_start_time = None
            self._update_stop_state_with_transaction(state)
        
        current_time = self.context.current_time
        last_boarding_time = current_time
        
        # Process boarding for each pending arrival
        for request_id in pending_arrivals:
            # Verify this request is actually expected to be picked up at this stop
            if request_id not in route_stop.pickup_passengers:
                logger.warning(f"Request {request_id} is not in pickup_passengers list for stop {route_stop.id} - adding it")
                route_stop.pickup_passengers.append(request_id)
                # Update the route to reflect this change
                self.state_manager.route_worker.update_route(route)
            
            passengers = self.state_manager.passenger_worker.get_all_passenger_ids_for_request_ids([request_id])
            logger.info(f"Processing boarding for request {request_id} with {len(passengers)} passenger(s)")
            
            for passenger_id in passengers:
                # Log wait time metrics
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
                
                # Update vehicle occupancy and create boarding event
                self.state_manager.vehicle_worker.increment_vehicle_occupancy(state.vehicle_id)
                boarding_time = current_time + timedelta(seconds=self.config.vehicle.boarding_time)
                last_boarding_time = max(last_boarding_time, boarding_time)
                self._create_boarding_event(passenger_id, state.vehicle_id, current_time)
            
            state.boarded_request_ids.add(request_id)
            
            # Update the route_stop's boarded_request_ids to maintain consistency
            if request_id not in route_stop.boarded_request_ids:
                route_stop.boarded_request_ids.append(request_id)
                # Update the route to reflect this change
                self.state_manager.route_worker.update_route(route)
                logger.debug(f"Updated route {route.id} to reflect boarding of request {request_id}")
        
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

        # Update state and check if all pickups are complete
        self._update_stop_state_with_transaction(state)
        
        # If all expected pickups have boarded, complete the stop operation
        pending_pickups = state.expected_pickup_request_ids - state.boarded_request_ids
        if not pending_pickups:
            logger.info(f"All expected passengers have boarded at stop {state.route_stop_id}")
            self._complete_stop_operation(state, last_boarding_time, 'boarding')

    def _start_wait_timer(self, state: StopCoordinationState) -> None:
        """Start a wait timer for missing passengers."""
        if state.wait_timeout_event_id:
            return
            
        if not self._is_vehicle_at_stop(state):
            return
            
        wait_duration = self.config.vehicle.max_dwell_time
        logger.info(f"Starting wait timer ({wait_duration}s) at stop {state.route_stop_id}")
        
        # Create wait timeout event
        wait_event = Event(
            event_type=EventType.VEHICLE_WAIT_TIMEOUT,
            priority=EventPriority.HIGH,
            timestamp=self.context.current_time + timedelta(seconds=wait_duration),
            data={
                'route_stop_id': state.route_stop_id,
                'missing_requests': list(state.expected_pickup_request_ids - state.arrived_pickup_request_ids),
                'wait_start_time': self.context.current_time
            }
        )
        
        # Update state and publish event
        state.wait_timeout_event_id = wait_event.id
        state.wait_start_time = self.context.current_time
        self._update_stop_state_with_transaction(state)
        self.context.event_manager.publish_event(wait_event)

    def handle_wait_timeout(self, event: Event) -> None:
        """Handle timeout for vehicle waiting for passengers."""
        route_stop_id = event.data['route_stop_id']
        logger.info(f"Handling wait timeout for stop {route_stop_id}")
        
        # Get coordination state
        state = self.state_manager.stop_coordination_worker.get_stop_state(route_stop_id)
        if not state:
            logger.error(f"No state found for route_stop_id: {route_stop_id}")
            return
            
        # Create no-show events for missing passengers
        missing_requests = state.expected_pickup_request_ids - state.arrived_pickup_request_ids
        logger.info(f"Creating no-show events for {len(missing_requests)} missing request(s)")
        
        for request_id in missing_requests:
            passengers = self.state_manager.passenger_worker.get_all_passenger_ids_for_request_ids([request_id])
            for passenger_id in passengers:
                logger.debug(f"Creating no-show event for passenger {passenger_id}")
                self._create_no_show_event(passenger_id, route_stop_id)
            
        # Complete stop operation after timeout
        if state.vehicle_id:
            self._complete_stop_operation(state, self.context.current_time, 'wait_timeout')

    def _handle_passenger_alighting(self, state: StopCoordinationState, route_stop: RouteStop) -> None:
        """Handle passenger alighting at a stop."""
        # Get passengers who are in the vehicle and expected to alight at this stop
        pending_dropoffs = state.expected_dropoff_request_ids - state.completed_dropoff_request_ids
        if not pending_dropoffs:
            if not state.expected_pickup_request_ids:
                self._complete_stop_operation(state, self.context.current_time, 'alighting')
            return

        # Get the route associated with this stop to ensure proper synchronization
        route, _ = self._find_route_stop(state.route_stop_id)
        if not route:
            logger.error(f"Could not find route for route_stop_id {state.route_stop_id}")
            return

        alighting_passengers = self.state_manager.passenger_worker.get_passengers_at_stop(
            PassengerStatus.IN_VEHICLE,
            route_stop.stop.id,
            StopPurpose.DROPOFF
        )

        if not alighting_passengers:
            # Double-check if there are any passengers that should be dropped off
            # but aren't properly tracked in the passenger state
            if pending_dropoffs:
                logger.warning(f"Expected {len(pending_dropoffs)} dropoffs at stop {route_stop.stop.id}, but no passengers found in IN_VEHICLE state")
                # Try to find these passengers in the system
                for request_id in pending_dropoffs:
                    passengers = self.state_manager.passenger_worker.get_all_passenger_ids_for_request_ids([request_id])
                    for passenger_id in passengers:
                        passenger = self.state_manager.passenger_worker.get_passenger(passenger_id)
                        if passenger:
                            logger.warning(f"Passenger {passenger_id} (request {request_id}) is in state {passenger.status} instead of IN_VEHICLE")
            
            if not state.expected_pickup_request_ids:
                self._complete_stop_operation(state, self.context.current_time, 'alighting')
            return

        logger.info(f"Processing alighting for {len(alighting_passengers)} passenger(s)")
        current_time = self.context.current_time
        last_alighting_time = current_time
        
        # Process each alighting passenger
        for passenger in alighting_passengers:
            if passenger.assigned_vehicle_id == state.vehicle_id and passenger.request_id in pending_dropoffs:
                alighting_time = current_time + timedelta(seconds=self.config.vehicle.alighting_time)
                last_alighting_time = max(last_alighting_time, alighting_time)
                
                # Create alighting event
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
                
                # Log metrics
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
                
                # Update vehicle occupancy and state
                self.state_manager.vehicle_worker.decrement_vehicle_occupancy(state.vehicle_id)
                state.completed_dropoff_request_ids.add(passenger.request_id)
                
                # Ensure the route is properly updated to reflect this dropoff
                # This is critical for maintaining consistency between the route and stop coordination state
                if route:
                    # Mark the passenger as dropped off in the route's boarded_request_ids
                    for stop in route.stops:
                        if passenger.request_id in stop.boarded_request_ids and passenger.request_id not in stop.dropoff_passengers:
                            logger.warning(f"Passenger {passenger.id} (request {passenger.request_id}) was boarded but not marked for dropoff - fixing")
                            stop.dropoff_passengers.append(passenger.request_id)
                    
                    # Update the route in the state manager
                    self.state_manager.route_worker.update_route(route)
                    logger.debug(f"Updated route {route.id} to reflect dropoff of passenger {passenger.id} (request {passenger.request_id})")

        # Update state
        self._update_stop_state_with_transaction(state)
        
        # Complete stop operation if no pickups expected
        if not state.expected_pickup_request_ids:
            self._complete_stop_operation(state, last_alighting_time, 'alighting')

    def _complete_stop_operation(self, state: StopCoordinationState, completion_time: datetime, operation_type: str) -> None:
        """
        Complete a stop operation (boarding or alighting) and prepare for vehicle departure.
        
        Args:
            state: The stop coordination state
            completion_time: When the operation completes
            operation_type: Type of operation ('boarding' or 'alighting')
        """
        # Get the route and route stop
        route, route_stop = self._find_route_stop(state.route_stop_id)
        if not route or not route_stop:
            logger.error(f"Could not find route or route stop for {state.route_stop_id}")
            return
            
        logger.info(f"Completing {operation_type} operation at stop {route_stop.stop.id}")
        
        # Mark the stop as completed in the route
        route_stop.completed = True
        route_stop.actual_departure_time = completion_time
        
        # Ensure passenger lists are consistent between route_stop and state
        # This is critical for maintaining consistency
        if operation_type == 'boarding':
            # All expected pickups should be in boarded_request_ids
            for request_id in state.expected_pickup_request_ids:
                if request_id not in state.boarded_request_ids:
                    logger.warning(f"Expected pickup {request_id} not in boarded_request_ids - marking as no-show")
                    # Handle no-show case
                    passengers = self.state_manager.passenger_worker.get_all_passenger_ids_for_request_ids([request_id])
                    for passenger_id in passengers:
                        self._create_no_show_event(passenger_id, route_stop.stop.id)
                
                # Ensure route_stop.boarded_request_ids is consistent with state.boarded_request_ids
                if request_id in state.boarded_request_ids and request_id not in route_stop.boarded_request_ids:
                    logger.warning(f"Request {request_id} in state.boarded_request_ids but not in route_stop.boarded_request_ids - fixing")
                    route_stop.boarded_request_ids.append(request_id)
        
        elif operation_type == 'alighting':
            # All expected dropoffs should be in completed_dropoff_request_ids
            for request_id in state.expected_dropoff_request_ids:
                if request_id not in state.completed_dropoff_request_ids:
                    logger.warning(f"Expected dropoff {request_id} not in completed_dropoff_request_ids - marking as completed anyway")
                    state.completed_dropoff_request_ids.add(request_id)
        
        # Update the route in the state manager
        self.state_manager.route_worker.update_route(route)
        
        # Update vehicle status
        self.state_manager.vehicle_worker.update_vehicle_status(
            state.vehicle_id,
            VehicleStatus.IN_SERVICE,
            self.context.current_time
        )
        
        # Update vehicle details
        vehicle = self.state_manager.vehicle_worker.get_vehicle(state.vehicle_id)
        if vehicle:
            vehicle.current_state.waiting_for_passengers = False
            vehicle.current_state.current_stop_id = route_stop.stop.id
            self.state_manager.vehicle_worker.update_vehicle(vehicle)
            
            # Get active route and create completion event
            active_route_id = self.state_manager.vehicle_worker.get_vehicle_active_route_id(vehicle.id)
            if active_route_id:
                completion_event = Event(
                    event_type=EventType.VEHICLE_STOP_OPERATIONS_COMPLETED,
                    priority=EventPriority.HIGH,
                    timestamp=completion_time,
                    vehicle_id=state.vehicle_id,
                    data={
                        'route_id': active_route_id,
                        'segment_id': state.segment_id,
                        'stop_id': state.route_stop_id,
                        'operation_type': operation_type,
                        'movement_start_time': state.movement_start_time,
                        'actual_distance': state.actual_distance
                    }
                )
                self.context.event_manager.publish_event(completion_event)
        
        # Clean up the stop state
        self._cleanup_stop_state(state)

    # Helper event creation methods remain mostly the same
    def _create_boarding_event(self, passenger_id: str, vehicle_id: str, current_time: datetime) -> None:
        logger.debug(f"[BoardingEvent] Creating boarding event for passenger {passenger_id} on vehicle {vehicle_id}")
        passenger = self.state_manager.passenger_worker.get_passenger(passenger_id)
        if not passenger:
            logger.error(f"[BoardingEvent] Cannot create event: Passenger {passenger_id} not found")
            return
        event = Event(
            event_type=EventType.PASSENGER_BOARDING_COMPLETED,
            priority=EventPriority.HIGH,
            timestamp=current_time + timedelta(seconds=self.config.vehicle.boarding_time),
            passenger_id=passenger_id,
            request_id=passenger.request_id,
            vehicle_id=vehicle_id,
            data={
                'boarding_start_time': current_time,
                'actual_pickup_time': current_time
            }
        )
        logger.debug(f"[BoardingEvent] Publishing boarding event {event.id} for passenger {passenger_id}")
        self.context.event_manager.publish_event(event)

    def _create_no_show_event(self, passenger_id: str, stop_id: str) -> None:
        logger.info(f"[NoShowEvent] Creating no-show event for passenger {passenger_id} at stop {stop_id}")
        event = Event(
            event_type=EventType.PASSENGER_NO_SHOW,
            priority=EventPriority.HIGH,
            timestamp=self.context.current_time,
            passenger_id=passenger_id,
            data={'stop_id': stop_id}
        )
        logger.debug(f"[NoShowEvent] Publishing no-show event {event.id} for passenger {passenger_id}")
        self.context.event_manager.publish_event(event)

    def _create_pickup_delay_violation_event(self, vehicle_id: str, delay: float) -> None:
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
        logger.debug(f"[PickupDelay] Publishing pickup delay violation event for vehicle {vehicle_id}")
        self.context.event_manager.publish_event(event)

    def _create_dropoff_delay_violation_event(self, vehicle_id: str, delay: float) -> None:
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
        logger.debug(f"[DropoffDelay] Publishing dropoff delay violation event for vehicle {vehicle_id}")
        self.context.event_manager.publish_event(event)

    def _cleanup_stop_state(self, state: StopCoordinationState) -> None:
        logger.info(f"[CleanupStopState] Cleaning up state for route_stop_id: {state.route_stop_id}")
        self.state_manager.stop_coordination_worker.remove_stop_state(state.route_stop_id)
