from datetime import datetime, timedelta
from typing import Dict, Any, Optional
from drt_sim.core.state.manager import StateManager
from drt_sim.core.simulation.context import SimulationContext
from drt_sim.models.event import Event, EventType, EventPriority
from drt_sim.models.vehicle import VehicleStatus
from drt_sim.models.location import Location
from drt_sim.config.config import ParameterSet
from drt_sim.models.route import RouteStatus, RouteStop, Route
from drt_sim.core.monitoring.types.metrics import MetricName
import traceback
import logging
logger = logging.getLogger(__name__)

class VehicleHandler:
    """
    Handles vehicle operations in the DRT system.
    Simplified to work directly with route stops without segments.
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
        # Add semaphore for critical operations
        self.operation_locks = {}  # Vehicle ID -> Lock

    def _setup_vehicle_thresholds(self) -> Dict[str, Any]:
        """Initialize vehicle operation thresholds from config"""
        return {
            'max_pickup_delay': self.config.vehicle.max_pickup_delay,
            'max_dropoff_delay': self.config.vehicle.max_dropoff_delay
        }

    def _get_vehicle_lock(self, vehicle_id: str):
        """Get or create a lock for a specific vehicle"""
        import asyncio
        if vehicle_id not in self.operation_locks:
            self.operation_locks[vehicle_id] = asyncio.Lock()
        return self.operation_locks[vehicle_id]

    def _get_active_route_id(self, vehicle_id: str) -> Optional[str]:
        """
        Get the active route ID for a vehicle using the state_manager
        
        Args:
            vehicle_id: ID of the vehicle
            
        Returns:
            Active route ID or None if no active route
        """
        return self.state_manager.vehicle_worker.get_vehicle_active_route_id(vehicle_id)

    async def handle_vehicle_dispatch_request(self, event: Event) -> None:
        """
        Handle vehicle dispatch request with simplified route handling.
        """
        vehicle_id = event.vehicle_id
        
        # Acquire lock for this vehicle to prevent concurrent operations
        async with self._get_vehicle_lock(vehicle_id):
            try:
                logger.debug(f"=== START VEHICLE DISPATCH HANDLING ===")
                logger.debug(f"Vehicle {vehicle_id} - Processing dispatch request at time: {self.context.current_time}")
                logger.debug(f"Event data: {event.data}")
                
                # Begin transaction
                self.state_manager.begin_transaction()
                
                route_id = event.data.get('route_id')
                event_route_version = event.data.get('route_version')
                
                vehicle = self.state_manager.vehicle_worker.get_vehicle(vehicle_id)
                if not vehicle:
                    raise ValueError(f"Vehicle {vehicle_id} not found")
                
                logger.debug(f"Vehicle state before dispatch: Status={vehicle.current_state.status}, "
                        f"Location=({vehicle.current_state.current_location.lat}, {vehicle.current_state.current_location.lon}), "
                        f"RouteID={vehicle.current_state.current_route_id}, "
                        f"Occupancy={vehicle.current_state.current_occupancy}")
                
                route = self.state_manager.route_worker.get_route(route_id)
                if not route:
                    raise ValueError(f"No route found with ID {route_id}")
                
                # Check if the route version in the event matches the current route version
                if event_route_version is not None and route.version != event_route_version:
                    logger.info(f"Skipping dispatch for vehicle {vehicle_id}: route {route_id} version mismatch " 
                             f"(event version: {event_route_version}, current version: {route.version})")
                    self.state_manager.rollback_transaction()
                    return
                
                logger.debug(f"Route details: ID={route.id}, Status={route.status}, StopCount={len(route.stops)}, Version={route.version}")
                
                # Cancel any pending events for this vehicle before creating new ones
                self._cancel_pending_vehicle_events(vehicle_id)
                
                # Update vehicle status to IN_SERVICE
                self.state_manager.vehicle_worker.update_vehicle_status(
                    vehicle_id,
                    VehicleStatus.IN_SERVICE,
                    self.context.current_time
                )
                
                logger.debug(f"Updated vehicle status to IN_SERVICE")
                
                # Validate route state
                if not route.stops:
                    raise ValueError(f"Route {route_id} has no stops")
                
                # For initial dispatch, ensure the current stop index is set correctly
                route.recalc_current_stop_index()
                
                # Get current stop
                current_stop = route.get_current_stop()
                if not current_stop:
                    raise ValueError(f"No current stop found in route {route_id}")
                
                logger.debug(f"Current stop: ID={current_stop.id}, Sequence={current_stop.sequence}, "
                        f"Location=({current_stop.stop.location.lat}, {current_stop.stop.location.lon})")
                
                # Update route status if it's just starting
                if route.status in [RouteStatus.CREATED, RouteStatus.PLANNED]:
                    route.status = RouteStatus.ACTIVE
                    route.actual_start_time = self.context.current_time
                    logger.debug(f"Updated route status to ACTIVE, ActualStartTime={self.context.current_time}")
                    
                # Mark current stop as in progress
                current_stop.mark_in_progress(vehicle.current_state.current_location)
                route.set_active_stop(current_stop.id, vehicle.current_state.current_location)
                
                logger.debug(f"Marked stop {current_stop.id} as in progress, set as active stop in route")
                
                # Update route state
                self.state_manager.route_worker.update_route(route)
                
                # Update vehicle's active route
                logger.info(f"Updating vehicle {vehicle_id} active route to {route_id}")
                self.state_manager.vehicle_worker.update_vehicle_active_route_id(
                    vehicle_id,
                    route_id
                )
                
                # Create movement event for the current stop
                logger.info(f"Creating movement event for vehicle {vehicle_id} toward stop {current_stop.id}")
                self._create_vehicle_movement_event(
                    vehicle_id=vehicle_id,
                    route_stop=current_stop,
                    current_time=self.context.current_time
                )
                
                self.state_manager.commit_transaction()
                logger.info(f"Successfully completed dispatch for vehicle {vehicle_id} with route {route_id}")
                logger.debug(f"=== END VEHICLE DISPATCH HANDLING ===")
                
            except Exception as e:
                self.state_manager.rollback_transaction()
                logger.error(f"Error handling vehicle dispatch: {traceback.format_exc()}")
                self._handle_vehicle_error(event, str(e))

    async def handle_vehicle_reroute_request(self, event: Event) -> None:
        """
        Handle vehicle reroute request.
        This is called when a vehicle needs to be rerouted to a new route.
        """
        vehicle_id = event.vehicle_id
        
        # Acquire lock for this vehicle to prevent concurrent operations
        async with self._get_vehicle_lock(vehicle_id):
            try:
                logger.debug(f"=== START VEHICLE REROUTE HANDLING ===")
                logger.debug(f"Vehicle {vehicle_id} - Processing reroute request at time: {self.context.current_time}")
                logger.debug(f"Event data: {event.data}")
                
                # Begin transaction
                self.state_manager.begin_transaction()
                
                route_id = event.data.get('route_id')
                event_route_version = event.data.get('route_version')
                
                if not route_id:
                    raise ValueError("No route ID provided in reroute request")
                
                # Get vehicle
                vehicle = self.state_manager.vehicle_worker.get_vehicle(vehicle_id)
                if not vehicle:
                    raise ValueError(f"Vehicle {vehicle_id} not found")
                
                # Get current route ID
                current_route_id = vehicle.current_state.current_route_id
                
                # Get new route
                new_route = self.state_manager.route_worker.get_route(route_id)
                if not new_route:
                    raise ValueError(f"Route {route_id} not found")
                
                # Check if the route version in the event matches the current route version
                if event_route_version is not None and new_route.version != event_route_version:
                    logger.info(f"Skipping reroute for vehicle {vehicle_id}: route {route_id} version mismatch " 
                             f"(event version: {event_route_version}, current version: {new_route.version})")
                    self.state_manager.rollback_transaction()
                    return
                
                logger.debug(f"New route details: ID={new_route.id}, Status={new_route.status}, StopCount={len(new_route.stops)}, Version={new_route.version}")
                
                # Get current route for later comparison
                current_route = None
                if current_route_id and current_route_id != route_id:
                    current_route = self.state_manager.route_worker.get_route(current_route_id)
                    logger.info(f"Vehicle {vehicle_id} is being rerouted from route {current_route_id} to route {route_id}")
                
                # Check if vehicle is currently at a stop waiting for passengers
                is_at_stop_waiting = (vehicle.current_state.status == VehicleStatus.AT_STOP)

                logger.debug(f"DIAGNOSTIC: Reroute request for vehicle {vehicle_id} with status {vehicle.current_state.status}")
                logger.debug(f"DIAGNOSTIC: Current route ID: {current_route_id}, New route ID: {route_id}")
                logger.debug(f"DIAGNOSTIC: Vehicle at stop: {is_at_stop_waiting}, Current stop ID: {vehicle.current_state.current_stop_id}")
                
                # Cancel any pending movement events for this vehicle
                self._cancel_pending_vehicle_events(vehicle_id)
                
                # Special handling for vehicles at a stop
                if is_at_stop_waiting:
                    # Get current stop ID and route stop
                    current_stop_id = vehicle.current_state.current_stop_id
                    
                    if not current_stop_id:
                        logger.warning(f"Vehicle {vehicle_id} is at stop but has no current_stop_id")
                        # Treat as normal rerouting
                        is_at_stop_waiting = False
                    else:
                        # Get the current route stop from the current route
                        current_route_stop = None
                        
                        # First try to find the stop in the current route
                        if current_route:
                            for stop in current_route.stops:
                                if stop.id == current_stop_id:
                                    current_route_stop = stop
                                    break
                        
                        # If not found in current route, it might be that the vehicle's current_stop_id
                        # has already been updated to a stop in the new route
                        if not current_route_stop and new_route:
                            logger.debug(f"Could not find route stop {current_stop_id} in current route {current_route_id}, checking new route {route_id}")
                            for stop in new_route.stops:
                                if stop.id == current_stop_id:
                                    # We found the stop in the new route - this means the vehicle's state
                                    # has already been partially updated to the new route
                                    logger.info(f"Found route stop {current_stop_id} in new route {route_id} - vehicle state already partially updated")
                                    current_route_stop = stop
                                    
                                    # Since we're already at this stop in the new route, we just need to
                                    # update the vehicle's route ID and continue waiting
                                    self.state_manager.vehicle_worker.update_vehicle_active_route_id(vehicle_id, route_id)
                                    current_route_stop.vehicle_is_present = True
                                    self.state_manager.route_worker.update_route(new_route)
                                    self.state_manager.commit_transaction()
                                    logger.debug(f"=== END VEHICLE REROUTE HANDLING (ALREADY AT STOP IN NEW ROUTE) ===")
                                    return
                                    
                        if not current_route_stop:
                            logger.warning(f"Could not find route stop {current_stop_id} in either current route {current_route_id} or new route {route_id}")
                            # Treat as normal rerouting
                            is_at_stop_waiting = False
                
                if is_at_stop_waiting and current_route_stop:
                    # SPECIAL CASE: Vehicle is at a stop waiting for passengers
                    
                    # 1. Determine if the current stop exists in the new route
                    # We need to find a stop in the new route that has the same physical stop ID
                    current_stop_in_new_route = False
                    new_route_stop = None
                    physical_stop_id = current_route_stop.stop.id  # Get the physical stop ID
                    
                    # First check if there's a direct match on route stop ID
                    for stop in new_route.stops:
                        if stop.id == current_route_stop.id:  # Direct match on route stop ID
                            current_stop_in_new_route = True
                            new_route_stop = stop
                            logger.debug(f"Found direct match on route stop ID {stop.id} in new route")
                            break
                    
                    # If no direct match on route stop ID, try to match on physical stop ID
                    if not current_stop_in_new_route:
                        for stop in new_route.stops:
                            if stop.stop.id == physical_stop_id:  # Match on physical stop ID
                                current_stop_in_new_route = True
                                new_route_stop = stop
                                logger.debug(f"Found match on physical stop ID {physical_stop_id} in new route (route stop ID: {stop.id})")
                                break
                    
                    if not current_stop_in_new_route:
                        logger.info(f"Current physical stop {physical_stop_id} not found in new route {route_id}")
                        
                    # 2. Check if we need to continue waiting at this stop
                    need_to_continue_waiting = False
                    
                    if current_stop_in_new_route:
                        # Check if there are passengers we're waiting for that are still in the new route
                        waiting_for_passengers = set(current_route_stop.pickup_passengers) - current_route_stop.arrived_pickup_request_ids
                        
                        if waiting_for_passengers:
                            # Check if these passengers are still in the new route stop
                            for request_id in waiting_for_passengers:
                                if request_id in new_route_stop.pickup_passengers:
                                    need_to_continue_waiting = True
                                    break
                    
                    # 3. Handle the two cases: continue waiting or move on
                    if need_to_continue_waiting:
                        # We need to continue waiting at this stop
                        logger.info(f"Vehicle {vehicle_id} will continue waiting at stop {current_stop_id} after reroute to {route_id}")
                        
                        # Update vehicle's active route ID
                        self.state_manager.vehicle_worker.update_vehicle_active_route_id(vehicle_id, route_id)
                        
                        # Transfer the waiting state to the new route stop
                        new_route_stop.vehicle_is_present = True
                        new_route_stop.last_vehicle_arrival_time = current_route_stop.last_vehicle_arrival_time
                        
                        # Transfer wait timer if it exists
                        if current_route_stop.wait_timeout_event_id:
                            new_route_stop.wait_timeout_event_id = current_route_stop.wait_timeout_event_id
                            new_route_stop.wait_start_time = current_route_stop.wait_start_time
                        
                        # Transfer arrived passengers information
                        for request_id in current_route_stop.arrived_pickup_request_ids:
                            if request_id in new_route_stop.pickup_passengers:
                                new_route_stop.register_passenger_arrival(request_id, self.context.current_time)
                        
                        # Update route in state
                        self.state_manager.route_worker.update_route(new_route)
                        
                        # Update vehicle's current_stop_id to the new route stop ID
                        vehicle.current_state.current_stop_id = new_route_stop.id
                        self.state_manager.vehicle_worker.update_vehicle(vehicle)
                        
                        # Keep vehicle status as AT_STOP
                        # No need to change vehicle status or create new events
                        
                        self.state_manager.commit_transaction()
                        logger.debug(f"=== END VEHICLE REROUTE HANDLING (CONTINUING AT STOP) ===")
                        return
                    else:
                        # We don't need to continue waiting, move on to next stop in new route
                        logger.info(f"Vehicle {vehicle_id} will stop waiting at {current_stop_id} and proceed with new route {route_id}")
                        
                        # Cancel any wait timer
                        if current_route_stop.wait_timeout_event_id:
                            self._cancel_wait_timer(current_route_stop)
                        
                        # Update vehicle status to IN_SERVICE
                        self.state_manager.vehicle_worker.update_vehicle_status(
                            vehicle_id,
                            VehicleStatus.IN_SERVICE,
                            self.context.current_time
                        )
                        
                        # Clear stop-related state
                        vehicle.current_state.current_stop_id = None
                        vehicle.current_state.waiting_for_passengers = False
                        
                        # Update vehicle's active route ID
                        self.state_manager.vehicle_worker.update_vehicle_active_route_id(vehicle_id, route_id)
                        self.state_manager.vehicle_worker.update_vehicle(vehicle)
                        
                        # Find first uncompleted stop in new route
                        first_uncompleted_stop = None
                        for stop in new_route.stops:
                            if not stop.completed:
                                first_uncompleted_stop = stop
                                break
                        
                        if first_uncompleted_stop:
                            # Create movement event to first uncompleted stop
                            self._create_vehicle_movement_event(
                                vehicle_id=vehicle_id,
                                route_stop=first_uncompleted_stop,
                                current_time=self.context.current_time
                            )
                        else:
                            logger.warning(f"No uncompleted stops found in new route {route_id}")
                        
                        # Update route in state
                        self.state_manager.route_worker.update_route(new_route)
                        
                        self.state_manager.commit_transaction()
                        logger.debug(f"=== END VEHICLE REROUTE HANDLING (LEAVING STOP) ===")
                        return
                
                # For vehicles not at a stop, proceed with normal rerouting
                
                # Get current vehicle location
                current_location = vehicle.current_state.current_location

                logger.info(f"Vehicle New Route Stops: {[str(stop) for stop in new_route.stops]}")
                
                # Instead of using get_active_stop, find the first uncompleted stop
                first_uncompleted_stop = None
                for stop in new_route.stops:
                    if not stop.completed:
                        first_uncompleted_stop = stop
                        break
                
                logger.debug(f"Rerouting from current location: ({current_location.lat}, {current_location.lon})")
                if first_uncompleted_stop:
                    logger.debug(f"Found first uncompleted stop: ID={first_uncompleted_stop.id}, Sequence={first_uncompleted_stop.sequence}")
                else:
                    logger.debug(f"No uncompleted stops found, will recalculate")
                
                # If no uncompleted stop could be determined, recalculate
                if not first_uncompleted_stop:
                    new_route.recalc_current_stop_index()
                    first_uncompleted_stop = new_route.get_current_stop()
                    if first_uncompleted_stop:
                        logger.debug(f"Recalculated first uncompleted stop: ID={first_uncompleted_stop.id}, Sequence={first_uncompleted_stop.sequence}")
                        first_uncompleted_stop.mark_in_progress(current_location)
                        new_route.set_active_stop(first_uncompleted_stop.id, current_location)
                
                # Validate that we have a valid stop
                if not first_uncompleted_stop:
                    raise ValueError(f"No valid uncompleted stop found in route {route_id}")
                
                # Update vehicle's active route
                self.state_manager.vehicle_worker.update_vehicle_active_route_id(vehicle_id, route_id)
                
                # Update route in state
                self.state_manager.route_worker.update_route(new_route)
                
                # Create movement event for current stop
                logger.debug(f"Creating movement event to next stop: ID={first_uncompleted_stop.id}, Sequence={first_uncompleted_stop.sequence}")
                self._create_vehicle_movement_event(
                    vehicle_id=vehicle_id,
                    route_stop=first_uncompleted_stop,
                    current_time=self.context.current_time
                )
                
                self.state_manager.commit_transaction()
                logger.info(f"Successfully rerouted vehicle {vehicle_id} to route {route_id}")
                logger.debug(f"=== END VEHICLE REROUTE HANDLING ===")
                
            except Exception as e:
                self.state_manager.rollback_transaction()
                logger.error(f"Error handling vehicle reroute: {traceback.format_exc()}")
                self._handle_vehicle_error(event, str(e))

    async def handle_vehicle_position_update(self, event: Event) -> None:
        """
        Handle intermediate position updates as vehicle moves between stops.
        """
        vehicle_id = event.vehicle_id
        
        # We don't need a lock for position updates as they're non-critical
        try:
            logger.debug(f"=== START POSITION UPDATE HANDLING ===")
            logger.debug(f"Vehicle {vehicle_id} - Processing position update at time: {self.context.current_time}")
            logger.debug(f"Event data: {event.data}")
            
            self.state_manager.begin_transaction()
            
            vehicle = self.state_manager.vehicle_worker.get_vehicle(vehicle_id)
            if not vehicle:
                logger.warning(f"Vehicle {vehicle_id} not found for position update")
                self.state_manager.rollback_transaction()
                return
            
            logger.debug(f"Current vehicle state: Status={vehicle.current_state.status}, "
                        f"Location=({vehicle.current_state.current_location.lat}, {vehicle.current_state.current_location.lon}), "
                        f"Occupancy={vehicle.current_state.current_occupancy}")
                
            # Get current route
            route_id = self.state_manager.vehicle_worker.get_vehicle_active_route_id(vehicle_id)
            if not route_id:
                logger.warning(f"Vehicle {vehicle_id} has no active route for position update")
                self.state_manager.rollback_transaction()
                return
                
            route = self.state_manager.route_worker.get_route(route_id)
            if not route:
                logger.warning(f"Route {route_id} not found for position update")
                self.state_manager.rollback_transaction()
                return
            
            # Extract update information
            new_location = event.data['location']
            distance_covered = event.data.get('distance_covered', 0)
            progress_percentage = event.data.get('progress_percentage', 0)
            stop_id = event.data.get('stop_id')
            
            logger.debug(f"Updating vehicle location to: ({new_location.lat}, {new_location.lon}), "
                        f"Distance covered: {distance_covered}m, Progress: {progress_percentage:.1f}%")
            
            # Update vehicle's current location
            self.state_manager.vehicle_worker.update_vehicle_location(
                vehicle_id,
                new_location,
                distance_covered
            )
            
            # Update stop progress if stop_id is provided
            if stop_id:
                logger.debug(f"Updating progress for stop ID: {stop_id}")
                
                # Find the active stop
                active_stop = None
                for stop in route.stops:
                    if stop.id == stop_id:
                        active_stop = stop
                        break
                
                if active_stop:
                    logger.debug(f"Found active stop: ID={active_stop.id}, Sequence={active_stop.sequence}")
                    
                    # Update progress
                    previous_progress = active_stop.progress_percentage if hasattr(active_stop, 'progress_percentage') else 0
                    active_stop.update_progress(progress_percentage, new_location)
                    
                    logger.debug(f"Stop progress updated: {previous_progress:.1f}% -> {active_stop.progress_percentage:.1f}%")
                    
                    # Ensure this stop is marked as the active one in the route
                    if route.active_stop_id != stop_id:
                        logger.debug(f"Setting active stop in route: Previous={route.active_stop_id}, New={stop_id}")
                        route.set_active_stop(stop_id, new_location)
                else:
                    logger.warning(f"Active stop {stop_id} not found in route {route_id}")
                        
            # Update route in state
            self.state_manager.route_worker.update_route(route)
            
            self.state_manager.commit_transaction()
            logger.debug(f"=== END POSITION UPDATE HANDLING ===")
            
        except Exception as e:
            self.state_manager.rollback_transaction()
            logger.error(f"Error handling vehicle position update: {traceback.format_exc()}")
            self._handle_vehicle_error(event, str(e))

    async def handle_vehicle_arrived_stop(self, event: Event) -> None:
        """
        Handle vehicle arrival at a stop, including all processing of passenger boarding and alighting.
        
        This consolidated method handles:
        1. Vehicle location and status updates
        2. Route stop state updates
        3. Passenger alighting (dropoffs)
        4. Passenger boarding (pickups)
        5. Wait timer management
        6. Metrics collection
        
        Args:
            event: The VEHICLE_ARRIVED_STOP event containing all necessary data
        """
        vehicle_id = event.vehicle_id
        
        # Acquire lock for this vehicle to prevent concurrent operations
        async with self._get_vehicle_lock(vehicle_id):
            try:
                logger.debug(f"=== START VEHICLE ARRIVAL HANDLING ===")
                logger.debug(f"Vehicle {vehicle_id} - Processing arrival at stop, time: {self.context.current_time}")
                logger.debug(f"Event data: {event.data}")
                
                # Begin transaction
                self.state_manager.begin_transaction()
                
                # Get vehicle
                vehicle = self.state_manager.vehicle_worker.get_vehicle(vehicle_id)
                if not vehicle:
                    raise ValueError(f"Vehicle {vehicle_id} not found")
                
                logger.debug(f"Vehicle state before arrival: Status={vehicle.current_state.status}, "
                        f"Location=({vehicle.current_state.current_location.lat}, {vehicle.current_state.current_location.lon}), "
                        f"RouteID={vehicle.current_state.current_route_id}, "
                        f"StopID={vehicle.current_state.current_stop_id}, "
                        f"Occupancy={vehicle.current_state.current_occupancy}, "
                        f"Waiting={vehicle.current_state.waiting_for_passengers}")
                
                # Check if this is a rebalancing arrival
                if event.data.get('is_rebalancing', False):
                    logger.debug(f"Handling rebalancing arrival for vehicle {vehicle_id}")
                    # Handle rebalancing arrival
                    self._handle_rebalancing_arrival(vehicle_id, event)
                    self.state_manager.commit_transaction()
                    return
                
                # Get current route
                current_route_id = vehicle.current_state.current_route_id
                if not current_route_id:
                    logger.warning(f"Vehicle {vehicle_id} has no active route during arrival")
                    self.state_manager.rollback_transaction()
                    return
                    
                route = self.state_manager.route_worker.get_route(current_route_id)
                if not route:
                    logger.warning(f"Route {current_route_id} not found during arrival")
                    self.state_manager.rollback_transaction()
                    return
                
                logger.debug(f"Route details: ID={route.id}, Status={route.status}, ActiveStopID={route.active_stop_id}, "
                        f"TotalStops={len(route.stops)}, StartTime={route.actual_start_time}")
                
                # Get route stop from event data
                route_stop: RouteStop = event.data.get('route_stop')
                if not route_stop:
                    logger.warning(f"Route stop not found in event data")
                    self.state_manager.rollback_transaction()
                    return
                
                logger.debug(f"Arrival at stop: ID={route_stop.id}, Sequence={route_stop.sequence}, "
                        f"Location=({route_stop.stop.location.lat}, {route_stop.stop.location.lon}), "
                        f"PickupCount={len(route_stop.pickup_passengers)}, "
                        f"DropoffCount={len(route_stop.dropoff_passengers)}")
                
                # Check if the vehicle is already at a stop - this could indicate a stale event
                if vehicle.current_state.status == VehicleStatus.AT_STOP:
                    logger.warning(f"[Stop Arrival] Vehicle {vehicle_id} is already at stop {vehicle.current_state.current_stop_id} - This may be a stale event")
                    self.state_manager.rollback_transaction()
                    return
                
                # 1. Update vehicle location
                destination_location = route_stop.stop.location
                logger.debug(f"Updating vehicle location to stop location: ({destination_location.lat}, {destination_location.lon})")
                self.state_manager.vehicle_worker.update_vehicle_location(
                    vehicle_id,
                    destination_location
                )
                
                # 2. Log distance metrics
                distance = event.data.get('actual_distance', 0)
                if vehicle.current_state.current_occupancy > 0:
                    logger.debug(f"Logging occupied distance: {distance}m, Occupancy={vehicle.current_state.current_occupancy}")
                    self.context.metrics_collector.log(
                        MetricName.VEHICLE_OCCUPIED_DISTANCE,
                        distance,
                        self.context.current_time,
                        { 'vehicle_id': vehicle_id }
                    )
                else:
                    logger.debug(f"Logging empty distance: {distance}m")
                    self.context.metrics_collector.log(
                        MetricName.VEHICLE_EMPTY_DISTANCE,
                        distance,
                        self.context.current_time,
                        { 'vehicle_id': vehicle_id }
                    )
                
                # 3. Update route stop with arrival information
                movement_start_time = event.data.get('movement_start_time')
                actual_duration = None
                if movement_start_time:
                    actual_duration = (self.context.current_time - movement_start_time).total_seconds()
                    logger.debug(f"Movement duration: {actual_duration}s, Started at: {movement_start_time}")
                
                route_stop.actual_arrival_time = self.context.current_time
                route_stop.actual_distance_to_stop = distance
                route_stop.actual_duration_to_stop = actual_duration
                
                logger.debug(f"Setting stop arrival data: ArrivalTime={self.context.current_time}, "
                        f"Distance={distance}m, Duration={actual_duration}s")
                
                # 4. Mark stop as completed in route
                route.mark_stop_completed(
                    stop_id=route_stop.id, 
                    actual_arrival_time=self.context.current_time,
                    actual_duration=actual_duration, 
                    actual_distance=distance
                )
                
                # 5. Update vehicle status to AT_STOP
                logger.debug(f"Updating vehicle status to AT_STOP")
                self.state_manager.vehicle_worker.update_vehicle_status(
                    vehicle_id,
                    VehicleStatus.AT_STOP,
                    self.context.current_time
                )
                
                # 6. Update additional vehicle state
                vehicle.current_state.current_stop_id = route_stop.id
                vehicle.current_state.waiting_for_passengers = True
                self.state_manager.vehicle_worker.update_vehicle(vehicle)
                
                logger.debug(f"Updated vehicle state: Status=AT_STOP, StopID={route_stop.id}, Waiting=True")
                
                # 7. Register vehicle arrival in route stop and get ready-for-boarding passengers
                ready_for_boarding = route_stop.register_vehicle_arrival(self.context.current_time)
                if ready_for_boarding:
                    logger.debug(f"Passengers ready for boarding: {ready_for_boarding}")
                else:
                    logger.debug(f"No passengers ready for boarding")
                
                # 8. Check for delay violations
                self._check_delay_violations(route_stop, vehicle_id, self.context.current_time)
                
                # 9. Handle dropoffs first
                if route_stop.dropoff_passengers:
                    logger.debug(f"Processing dropoffs for {len(route_stop.dropoff_passengers)} passengers")
                    self._handle_passenger_alighting(route_stop, vehicle_id)
                    
                    # Check if this is a dropoff-only stop (no pickups) and complete operations if needed
                    if not route_stop.pickup_passengers and route_stop.is_dropoff_complete():
                        logger.debug(f"Dropoff-only stop with all dropoffs complete, completing stop operations")
                        self._create_stop_operations_completed_event(vehicle_id, route_stop.id, self.context.current_time)
                
                # 10. Start wait timer if needed
                missing_pickups = len(route_stop.pickup_passengers) - len(route_stop.arrived_pickup_request_ids)
                if missing_pickups > 0:
                    logger.debug(f"Starting wait timer for {missing_pickups} missing pickups")
                    self._start_wait_timer(route_stop, vehicle_id)
                
                # 11. Initialize boarding for passengers already at the stop
                if ready_for_boarding:
                    logger.debug(f"Processing boarding for {len(ready_for_boarding)} ready passengers")
                    self._process_passenger_boarding(route_stop, ready_for_boarding, vehicle_id, self.context.current_time)
                
                # 12. Update route to persist state
                self.state_manager.route_worker.update_route(route)
                logger.info(f"Route Updated after step #12 in handle_vehicle_arrived_stop")

                logger.info(f"The Route Stops for the vehicle are: {[str(rs) for rs in route.stops]}")
                
                # 13. Update metrics
                if self.context.metrics_collector:
                    logger.debug(f"Logging stop served metric")
                    self.context.metrics_collector.log(
                        MetricName.VEHICLE_STOPS_SERVED,
                        1,
                        self.context.current_time,
                        { 'vehicle_id': vehicle_id }
                    )
                
                # Commit the transaction
                self.state_manager.commit_transaction()
                logger.debug(f"=== END VEHICLE ARRIVAL HANDLING ===")
                
            except Exception as e:
                self.state_manager.rollback_transaction()
                logger.error(f"Error handling vehicle arrival: {traceback.format_exc()}")
                self._handle_vehicle_error(event, str(e))

    def _handle_rebalancing_arrival(self, vehicle_id: str, event: Event) -> None:
        """Handle vehicle arrival at depot after rebalancing."""
        # Update vehicle location first
        self.state_manager.vehicle_worker.update_vehicle_location(
            vehicle_id,
            event.data.get('destination')
        )
        
        # Log distance metric if available
        distance = event.data.get('actual_distance', 0)
        vehicle = self.state_manager.vehicle_worker.get_vehicle(vehicle_id)
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
        
        # Clear active route for the vehicle
        self.state_manager.vehicle_worker.update_vehicle_active_route_id(vehicle_id, None)
        
        # Update vehicle status to IDLE
        self.state_manager.vehicle_worker.update_vehicle_status(
            vehicle_id,
            VehicleStatus.IDLE,
            self.context.current_time
        )

    def handle_vehicle_wait_timeout(self, event: Event) -> None:
        """
        Handle timeout for vehicle waiting for passengers.
        
        This is called when a vehicle has waited the maximum allowed time for passengers to arrive.
        
        Args:
            event: The VEHICLE_WAIT_TIMEOUT event
        """
        vehicle_id = event.vehicle_id
        route_stop_id = event.data.get('route_stop_id')
        
        if not vehicle_id or not route_stop_id:
            logger.error(f"Missing required data in wait timeout event: {event.id}")
            return
            
        try:
            logger.debug(f"=== START WAIT TIMEOUT HANDLING ===")
            logger.debug(f"Vehicle {vehicle_id} - Processing wait timeout at time: {self.context.current_time}")
            logger.debug(f"Event data: {event.data}")
            
            self.state_manager.begin_transaction()
            
            # Get the vehicle to access its current route ID
            vehicle = self.state_manager.vehicle_worker.get_vehicle(vehicle_id)
            if not vehicle or not vehicle.current_state.current_route_id:
                logger.warning(f"Vehicle {vehicle_id} not found or has no active route")
                self.state_manager.rollback_transaction()
                return
                
            logger.debug(f"Vehicle state during timeout: Status={vehicle.current_state.status}, "
                    f"RouteID={vehicle.current_state.current_route_id}, "
                    f"StopID={vehicle.current_state.current_stop_id}, "
                    f"Waiting={vehicle.current_state.waiting_for_passengers}")
                
            # Get the route using the vehicle's current route ID
            route = self.state_manager.route_worker.get_route(vehicle.current_state.current_route_id)
            if not route:
                logger.warning(f"Route {vehicle.current_state.current_route_id} not found for vehicle {vehicle_id}")
                self.state_manager.rollback_transaction()
                return
            
            logger.debug(f"Route details: ID={route.id}, Status={route.status}, ActiveStopID={route.active_stop_id}")
            
            # Find the route stop
            route_stop = None
            for rs in route.stops:
                if rs.id == route_stop_id:
                    route_stop = rs
                    break
            
            if not route_stop:
                logger.warning(f"Route stop {route_stop_id} not found for wait timeout")
                self.state_manager.rollback_transaction()
                return
                
            logger.debug(f"Route stop details: ID={route_stop.id}, Sequence={route_stop.sequence}, "
                    f"PickupCount={len(route_stop.pickup_passengers)}, "
                    f"DropoffCount={len(route_stop.dropoff_passengers)}, "
                    f"ArrivedPickups={len(route_stop.arrived_pickup_request_ids)}")
                
            # Check if wait timer is still active
            if route_stop.wait_timeout_event_id != event.id:
                logger.info(f"Wait timeout event {event.id} is no longer active for stop {route_stop_id} (current={route_stop.wait_timeout_event_id})")
                self.state_manager.rollback_transaction()
                return
                
            # Process no-shows for passengers who haven't arrived
            missing_passengers = set(route_stop.pickup_passengers) - route_stop.arrived_pickup_request_ids
            
            logger.debug(f"Missing passengers at timeout: {missing_passengers}")
            
            for request_id in missing_passengers:
                # Get passenger ID from request
                request = self.state_manager.request_worker.get_request(request_id)
                if not request:
                    logger.warning(f"Request {request_id} not found for no-show processing")
                    continue
                    
                logger.debug(f"Creating no-show event for passenger {request.passenger_id}, request {request_id}")
                    
                # Create no-show event
                no_show_event = Event(
                    event_type=EventType.PASSENGER_NO_SHOW,
                    priority=EventPriority.HIGH,
                    timestamp=self.context.current_time,
                    vehicle_id=vehicle_id,
                    passenger_id=request.passenger_id,
                    request_id=request_id,
                    data={
                        'route_stop_id': route_stop_id,
                        'stop_id': route_stop.stop.id
                    }
                )
                self.context.event_manager.publish_event(no_show_event)
                logger.info(f"Created no-show event {no_show_event.id} for passenger {request.passenger_id} (request {request_id}) at route stop {route_stop.id}")
            
            # Clear wait timer
            logger.debug(f"Cancelling wait timer for stop {route_stop.id}")
            route_stop.cancel_wait_timer()
            
            # Create stop operations completed event if all operations are complete
            if route_stop.is_dropoff_complete():
                logger.debug(f"All operations at stop {route_stop.id} are complete, creating stop operations completed event")
                self._create_stop_operations_completed_event(vehicle_id, route_stop_id, self.context.current_time)
            else:
                logger.debug(f"Stop operations not yet complete at {route_stop.id}")
            
            # Update route to persist changes
            self.state_manager.route_worker.update_route(route)
            
            self.state_manager.commit_transaction()
            logger.debug(f"=== END WAIT TIMEOUT HANDLING ===")
            
        except Exception as e:
            self.state_manager.rollback_transaction()
            logger.error(f"Error handling wait timeout: {traceback.format_exc()}")
            self._handle_vehicle_error(event, str(e))

    async def handle_stop_operations_completed(self, event: Event) -> None:
        """
        Handle completion of operations at a stop
        
        This method:
        1. Marks the stop as completed
        2. Checks for pending route changes
        3. Creates movement events to the next stop
        
        Args:
            event: The VEHICLE_STOP_OPERATIONS_COMPLETED event
        """
        vehicle_id = event.vehicle_id
        route_stop_id = event.data.get('route_stop_id')
        
        if not vehicle_id or not route_stop_id:
            logger.error(f"Missing required data in stop operations completed event: {event.id}")
            return
            
        # Acquire lock for this vehicle to prevent concurrent operations
        async with self._get_vehicle_lock(vehicle_id):
            try:
                logger.debug(f"=== START STOP OPERATIONS COMPLETED HANDLING ===")
                logger.debug(f"Vehicle {vehicle_id} - Processing stop operations completed at time: {self.context.current_time}")
                logger.debug(f"Event data: {event.data}")
                
                self.state_manager.begin_transaction()
                
                # Get the vehicle to access its current route ID
                vehicle = self.state_manager.vehicle_worker.get_vehicle(vehicle_id)
                if not vehicle or not vehicle.current_state.current_route_id:
                    logger.warning(f"Vehicle {vehicle_id} not found or has no active route")
                    self.state_manager.rollback_transaction()
                    return
                    
                logger.debug(f"Vehicle state: Status={vehicle.current_state.status}, "
                        f"RouteID={vehicle.current_state.current_route_id}, "
                        f"StopID={vehicle.current_state.current_stop_id}, "
                        f"Location=({vehicle.current_state.current_location.lat}, {vehicle.current_state.current_location.lon}), "
                        f"Occupancy={vehicle.current_state.current_occupancy}")
                    
                # Get the route using the vehicle's current route ID
                route = self.state_manager.route_worker.get_route(vehicle.current_state.current_route_id)
                if not route:
                    logger.warning(f"Route {vehicle.current_state.current_route_id} not found for vehicle {vehicle_id}")
                    self.state_manager.rollback_transaction()
                    return
                
                logger.debug(f"Route details: ID={route.id}, Status={route.status}, ActiveStopID={route.active_stop_id}, StopCount={len(route.stops)}")
                
                # Find the route stop
                route_stop = None
                for rs in route.stops:
                    if rs.id == route_stop_id:
                        route_stop = rs
                        break
                
                if not route_stop:
                    logger.warning(f"Route stop {route_stop_id} not found for stop operations completed")
                    self.state_manager.rollback_transaction()
                    return
                    
                logger.debug(f"Route stop details: ID={route_stop.id}, Sequence={route_stop.sequence}, "
                        f"Completed={route_stop.completed}, PendingRouteChange={route_stop.has_pending_route_change()}")
                    
                # Check for pending route change
                if route_stop.has_pending_route_change():
                    logger.debug(f"Route stop {route_stop.id} has pending route change to route {route_stop.new_route_id}")
                    
                    # Complete the current stop
                    route_stop.mark_completed(
                        actual_arrival_time=route_stop.last_vehicle_arrival_time,
                        actual_departure_time=self.context.current_time
                    )
                    
                    logger.debug(f"Marked stop {route_stop.id} as completed: DepartureTime={self.context.current_time}")
                    
                    # Update the current route
                    self.state_manager.route_worker.update_route(route)
                    
                    # Get the new route
                    new_route_id = route_stop.new_route_id
                    new_route = self.state_manager.route_worker.get_route(new_route_id)
                    
                    if new_route:
                        logger.debug(f"New route details: ID={new_route.id}, Status={new_route.status}, StopCount={len(new_route.stops)}")
                        
                        # Update vehicle status
                        self.state_manager.vehicle_worker.update_vehicle_status(
                            vehicle_id,
                            VehicleStatus.IN_SERVICE,
                            self.context.current_time
                        )
                        
                        logger.debug(f"Updated vehicle status to IN_SERVICE with new route {new_route_id}")
                        
                        # Create movement event to first stop of new route
                        if new_route.stops:
                            next_stop = new_route.stops[0]
                            logger.debug(f"Creating movement event to first stop of new route: StopID={next_stop.id}, Sequence={next_stop.sequence}")
                            self._create_vehicle_movement_event(vehicle_id, next_stop, self.context.current_time)
                            
                        logger.info(f"Vehicle {vehicle_id} switched from route {route.id} to {new_route_id}")
                    else:
                        logger.error(f"New route {new_route_id} not found for route change")
                        
                    # Clear route change
                    route_stop.clear_route_change()
                    logger.debug(f"Cleared route change information from stop {route_stop.id}")
                    
                else:
                    # Complete the current stop
                    route_stop.mark_completed(
                        actual_arrival_time=route_stop.last_vehicle_arrival_time,
                        actual_departure_time=self.context.current_time
                    )
                    
                    logger.debug(f"Marked stop {route_stop.id} as completed: DepartureTime={self.context.current_time}")
                    
                    # Update the route
                    self.state_manager.route_worker.update_route(route)
                    
                    # Check if this was the last stop
                    if route_stop.sequence == len(route.stops) - 1:
                        logger.debug(f"This was the last stop (sequence {route_stop.sequence}) in route {route.id}")
                        
                        # This was the last stop, complete the route
                        self.state_manager.route_worker.update_route_status(
                            route.id,
                            RouteStatus.COMPLETED,
                            {
                                'actual_end_time': self.context.current_time
                            }
                        )
                        
                        logger.debug(f"Updated route {route.id} status to COMPLETED, EndTime={self.context.current_time}")
                        
                        # Update vehicle status
                        self.state_manager.vehicle_worker.update_vehicle_status(
                            vehicle_id,
                            VehicleStatus.IDLE,
                            self.context.current_time
                        )
                        self.state_manager.vehicle_worker.clear_vehicle_active_route(vehicle_id)
                        
                        logger.debug(f"Updated vehicle {vehicle_id} status to IDLE, cleared route and stop IDs")
                        logger.info(f"Vehicle {vehicle_id} completed route {route.id}")
                        
                    else:
                        logger.debug(f"Looking for next stop after sequence {route_stop.sequence} in route {route.id}")
                        
                        # Find the next stop
                        next_stop = None
                        for i in range(route_stop.sequence + 1, len(route.stops)):
                            if not route.stops[i].completed:
                                next_stop = route.stops[i]
                                break
                                
                        if next_stop:
                            logger.debug(f"Found next stop: ID={next_stop.id}, Sequence={next_stop.sequence}")
                            self.state_manager.vehicle_worker.update_vehicle_status(
                                vehicle_id,
                                VehicleStatus.IN_SERVICE,
                                self.context.current_time
                            )
                            # Create movement event to next stop
                            self._create_vehicle_movement_event(vehicle_id, next_stop, self.context.current_time)
                        else:
                            logger.warning(f"No next stop found for vehicle {vehicle_id} on route {route.id}")
                
                self.state_manager.commit_transaction()
                logger.debug(f"=== END STOP OPERATIONS COMPLETED HANDLING ===")
                
            except Exception as e:
                self.state_manager.rollback_transaction()
                logger.error(f"Error handling stop operations completed: {traceback.format_exc()}")
                self._handle_vehicle_error(event, str(e))


    def _handle_route_completion(self, vehicle_id: str, route: Route, current_time: datetime) -> Route:
        """
        Handle completion of route and initiate rebalancing through event publishing.
        Updates route status and triggers rebalancing request.
        """
        # Update route status
        route.status = RouteStatus.COMPLETED
        route.actual_end_time = current_time

        # Create rebalancing required event
        self._create_rebalancing_event(vehicle_id)
        return route

    async def _handle_route_continuation(self, vehicle_id: str, route: Route, current_time: datetime) -> Route:
        """
        Handle continuation to next stop in route.
        Simplified to work directly with route stops.
        
        Args:
            vehicle_id: ID of the vehicle
            route: Current route
            current_time: Current time
            
        Returns:
            Updated route
        """
        logger.debug(f"=== START ROUTE CONTINUATION ===")
        logger.debug(f"Vehicle {vehicle_id} - Processing route continuation at time: {current_time}")
        
        # Validate route state before continuation
        if not route.stops:
            logger.error(f"[Route Continuation] Vehicle {vehicle_id} - Route has no stops")
            raise ValueError("Route has no stops")
        
        # Let Route model handle finding the next stop
        route.recalc_current_stop_index()
        
        # Get next stop using Route's method
        next_stop = route.get_current_stop()
        if not next_stop:
            logger.error(f"[Route Continuation] Vehicle {vehicle_id} - No next stop found")
            raise ValueError("No next stop found")
        
        # Get vehicle for current location
        vehicle = self.state_manager.vehicle_worker.get_vehicle(vehicle_id)
        if not vehicle:
            raise ValueError(f"Vehicle {vehicle_id} not found")
        
        # Update vehicle status to IN_SERVICE
        self.state_manager.vehicle_worker.update_vehicle_status(
            vehicle_id,
            VehicleStatus.IN_SERVICE,
            current_time
        )
        
        # Clear stop-related state
        vehicle.current_state.current_stop_id = None
        vehicle.current_state.waiting_for_passengers = False
        self.state_manager.vehicle_worker.update_vehicle(vehicle)
        
        # Mark next stop as in progress
        next_stop.mark_in_progress(vehicle.current_state.current_location)
        route.set_active_stop(next_stop.id, vehicle.current_state.current_location)
        
        # Create movement event for next stop
        self._create_vehicle_movement_event(
            vehicle_id=vehicle_id,
            route_stop=next_stop,
            current_time=current_time
        )
        
        return route

    def _create_vehicle_movement_event(
        self,
        vehicle_id: str,
        route_stop: RouteStop,
        current_time: datetime
    ) -> None:
        """
        Create events for vehicle movement to a stop with waypoint updates.
        Simplified to work directly with route stops.
        
        Args:
            vehicle_id: ID of the vehicle
            route_stop: Stop to create movement events for
            current_time: Current time
        """
        logger.debug(f"=== START MOVEMENT EVENT CREATION ===")
        logger.debug(f"Vehicle {vehicle_id} - Creating movement events at time: {current_time}")
        logger.debug(f"Movement destination: Route Stop ID={route_stop.id}, Location=({route_stop.stop.location.lat}, {route_stop.stop.location.lon})")
        
        if not route_stop:
            logger.error(f"[Movement Event] Vehicle {vehicle_id} - RouteStop is None")
            raise ValueError("Cannot create movement event for None route_stop")
            
        # Get vehicle location (as origin)
        vehicle = self.state_manager.vehicle_worker.get_vehicle(vehicle_id)
        if not vehicle:
            raise ValueError(f"Vehicle {vehicle_id} not found")
        
        origin_location = vehicle.current_state.current_location
        destination_location = route_stop.stop.location
        
        # Calculate total movement duration and distance
        total_duration = route_stop.estimated_duration_to_stop
        total_distance = route_stop.estimated_distance_to_stop
        
        logger.debug(f"Movement details: Origin=({origin_location.lat}, {origin_location.lon}), " 
                    f"Destination=({destination_location.lat}, {destination_location.lon}), "
                    f"EstDuration={total_duration}s, EstDistance={total_distance}m")
        
        # Mark stop as in progress
        route_stop.mark_in_progress(origin_location)
        route_stop.movement_start_time = current_time
        
        # Create intermediate position update events (max 4)
        position_update_events = []
        
        # Determine how many updates to create
        max_updates = 4
        
        # Only create updates for longer movements
        if total_duration > 30:  # Only add position updates for trips longer than 30 seconds
            # Calculate time intervals for updates
            time_between_updates = total_duration / (max_updates + 1)
            distance_between_updates = total_distance / (max_updates + 1)
            
            logger.debug(f"Creating {max_updates} intermediate position updates, interval={time_between_updates}s")
            
            current_distance = 0
            for i in range(max_updates):
                update_time = current_time + timedelta(seconds=(i+1) * time_between_updates)
                current_distance += distance_between_updates
                progress_percentage = ((i+1) / (max_updates + 1)) * 100
                
                intermediate_location = self._interpolate_location(origin_location, destination_location, progress_percentage/100)
                
                logger.debug(f"Position update {i+1}: Time={update_time}, "
                            f"Location=({intermediate_location.lat}, {intermediate_location.lon}), "
                            f"Progress={progress_percentage:.1f}%, Distance={current_distance:.1f}m")
                
                # Create position update event
                update_event = Event(
                    event_type=EventType.VEHICLE_POSITION_UPDATE,
                    priority=EventPriority.HIGH,
                    timestamp=update_time,
                    vehicle_id=vehicle_id,
                    data={
                        'stop_id': route_stop.id,
                        'location': intermediate_location,
                        'progress_percentage': progress_percentage,
                        'distance_covered': current_distance,
                        'movement_start_time': current_time
                    }
                )
                position_update_events.append(update_event)
        else:
            logger.debug(f"Trip too short ({total_duration}s), skipping intermediate position updates")
        
        # Create final arrival event
        arrival_time = current_time + timedelta(seconds=total_duration)
        logger.debug(f"Creating final arrival event: Time={arrival_time}, "
                    f"TotalDuration={total_duration}s, TotalDistance={total_distance}m")
        
        arrival_event = Event(
            event_type=EventType.VEHICLE_ARRIVED_STOP,
            priority=EventPriority.HIGH,
            timestamp=arrival_time,
            vehicle_id=vehicle_id,
            data={
                'route_stop': route_stop,
                'movement_start_time': current_time,
                'origin': origin_location,
                'destination': destination_location,
                'actual_duration': total_duration,
                'actual_distance': total_distance
            }
        )
        
        # Publish events in order
        for i, event in enumerate(position_update_events):
            logger.debug(f"Publishing position update event {i+1}: ID={event.id}, Time={event.timestamp}")
            self.context.event_manager.publish_event(event)
        
        logger.debug(f"Publishing arrival event: ID={arrival_event.id}, Time={arrival_event.timestamp}")
        self.context.event_manager.publish_event(arrival_event)
        logger.debug(f"=== END MOVEMENT EVENT CREATION ===")
        
    def _interpolate_location(self, origin: Location, destination: Location, fraction: float) -> Location:
        """
        Interpolate between two locations based on a fraction (0-1).
        Used for creating intermediate position updates.
        
        Args:
            origin: Origin location
            destination: Destination location
            fraction: How far along the path (0-1)
            
        Returns:
            Interpolated location
        """
        # Simple linear interpolation
        lat = origin.lat + (destination.lat - origin.lat) * fraction
        lon = origin.lon + (destination.lon - origin.lon) * fraction
        
        return Location(
            lat=lat,
            lon=lon,
        )


    def handle_vehicle_rebalancing_required(self, event: Event) -> None:
        """Handle rebalancing request to depot."""
        try:
            self.state_manager.begin_transaction()
            
            vehicle_id = event.vehicle_id
            logger.info(f"Handling rebalancing for vehicle {vehicle_id}")
            
            # This is a placeholder for actual rebalancing logic
            # In a real implementation, you would calculate optimal depot location,
            # create a route to the depot, and dispatch the vehicle
            logger.info(f"Rebalancing implementation required")
            
            self.state_manager.commit_transaction()
            
        except Exception as e:
            self.state_manager.rollback_transaction()
            logger.error(f"Error handling rebalancing request: {traceback.format_exc()}")
            self._handle_vehicle_error(event, str(e))
        
    def _create_rebalancing_event(self, vehicle_id: str) -> None:
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

    async def handle_passenger_ready_for_boarding(self, event: Event) -> None:
        """
        Handle passenger ready for boarding event
        
        This event is triggered when a passenger arrives at a stop where a vehicle is already present.
        
        Args:
            event: The PASSENGER_READY_FOR_BOARDING event
        """
        vehicle_id = event.vehicle_id
        request_id = event.request_id
        route_stop_id = event.data.get('route_stop_id')
        
        if not vehicle_id or not request_id or not route_stop_id:
            logger.error(f"Missing required data in PASSENGER_READY_FOR_BOARDING event: {event.id}")
            return
            
        # Acquire lock for this vehicle to prevent concurrent operations
        async with self._get_vehicle_lock(vehicle_id):
            try:
                self.state_manager.begin_transaction()
                
                # Get the vehicle to access its current route ID
                vehicle = self.state_manager.vehicle_worker.get_vehicle(vehicle_id)
                if not vehicle or not vehicle.current_state.current_route_id:
                    logger.warning(f"Vehicle {vehicle_id} not found or has no active route")
                    self.state_manager.rollback_transaction()
                    return
                    
                # Get the route using the vehicle's current route ID
                route = self.state_manager.route_worker.get_route(vehicle.current_state.current_route_id)
                if not route:
                    logger.warning(f"Route {vehicle.current_state.current_route_id} not found for vehicle {vehicle_id}")
                    self.state_manager.rollback_transaction()
                    return
                
                # Find the route stop
                route_stop = None
                for rs in route.stops:
                    if rs.id == route_stop_id:
                        route_stop = rs
                        break
                
                if not route_stop:
                    logger.warning(f"Route stop {route_stop_id} not found for vehicle {vehicle_id}")
                    self.state_manager.rollback_transaction()
                    return
                    
                # Process boarding for this passenger
                self._process_passenger_boarding(route_stop, {request_id}, vehicle_id, self.context.current_time)
                
                # Update route to persist changes
                self.state_manager.route_worker.update_route(route)
                
                self.state_manager.commit_transaction()
                
            except Exception as e:
                self.state_manager.rollback_transaction()
                logger.error(f"Error handling passenger ready for boarding: {traceback.format_exc()}")
                self._handle_vehicle_error(event, str(e))

    def _process_passenger_boarding(self, route_stop: RouteStop, request_ids: set, vehicle_id: str, current_time: datetime) -> None:
        """
        Process passenger boarding for the given request IDs
        
        Args:
            route_stop: The route stop where boarding is happening
            request_ids: Set of request IDs to process boarding for
            vehicle_id: ID of the vehicle
            current_time: Current simulation time
        """
        if not request_ids:
            logger.debug(f"No request IDs provided for boarding")
            return
            
        logger.debug(f"Processing boarding for {len(request_ids)} passengers at route stop {route_stop.id}")

        # Create boarding events for each passenger
        for request_id in request_ids:
            # Get passenger ID from request
            request = self.state_manager.request_worker.get_request(request_id)
            if not request:
                logger.warning(f"Request {request_id} not found for boarding")
                continue
                
            logger.debug(f"Processing boarding for passenger {request.passenger_id}, request {request_id}")
                
            # Register boarding with route stop
            route_stop.register_boarding(request_id)

            #Update vehicle occupancy
            self.state_manager.vehicle_worker.increment_vehicle_occupancy(vehicle_id)
            
            # Create boarding completed event
            boarding_event = Event(
                event_type=EventType.PASSENGER_BOARDING_COMPLETED,
                priority=EventPriority.HIGH,
                timestamp=current_time,
                vehicle_id=vehicle_id,
                passenger_id=request.passenger_id,
                request_id=request_id,
                data={
                    'route_stop_id': route_stop.id,
                    'stop_id': route_stop.stop.id
                }
            )
            self.context.event_manager.publish_event(boarding_event)
            logger.info(f"Created boarding event {boarding_event.id} for passenger {request.passenger_id} (request {request_id}) at route stop {route_stop.id}")
            
        # Check if all pickups are complete
        pickup_complete = route_stop.is_pickup_complete()
        dropoff_complete = route_stop.is_dropoff_complete()
        
        logger.debug(f"Stop operations status: PickupComplete={pickup_complete}, DropoffComplete={dropoff_complete}, "
                f"ArrivedPickups={len(route_stop.arrived_pickup_request_ids)}/{len(route_stop.pickup_passengers)}, "
                f"CompletedDropoffs={len(route_stop.completed_dropoff_request_ids)}/{len(route_stop.dropoff_passengers)}")
        
        if pickup_complete and dropoff_complete:
            logger.debug(f"All operations complete at route stop {route_stop.id}, cancelling any wait timer")
            
            # If all operations are complete, cancel any wait timer
            if route_stop.wait_timeout_event_id:
                self._cancel_wait_timer(route_stop)
                
            self._create_stop_operations_completed_event(vehicle_id, route_stop.id, current_time)

    def _start_wait_timer(self, route_stop: RouteStop, vehicle_id: str) -> None:
        """
        Start a wait timer for a vehicle at a stop
        
        Args:
            route_stop: The route stop where the vehicle is waiting
            vehicle_id: ID of the vehicle
        """
        # Only start timer if not already waiting
        if route_stop.wait_start_time is not None:
            logger.debug(f"Wait timer already started for vehicle {vehicle_id} at route stop {route_stop.id} at {route_stop.wait_start_time}")
            return
            
        # Calculate wait timeout
        wait_duration = self.config.vehicle.max_dwell_time
        wait_end_time = self.context.current_time + timedelta(seconds=wait_duration)
        
        logger.debug(f"Creating wait timer for vehicle {vehicle_id} at route stop {route_stop.id}")
        logger.debug(f"Wait duration: {wait_duration}s, End time: {wait_end_time}")
        
        # Create wait timeout event
        wait_event = Event(
            event_type=EventType.VEHICLE_WAIT_TIMEOUT,
            priority=EventPriority.HIGH,
            timestamp=wait_end_time,
            vehicle_id=vehicle_id,
            data={
                'route_stop_id': route_stop.id,
                'stop_id': route_stop.stop.id,
                'wait_start_time': self.context.current_time
            }
        )
        
        # Register wait timer with route stop
        route_stop.start_wait_timer(wait_event.id, self.context.current_time)
        
        # Add event to queue
        self.context.event_manager.publish_event(wait_event)
        logger.debug(f"Started wait timer for vehicle {vehicle_id} at stop {route_stop.id}: EventID={wait_event.id}, Until={wait_end_time}")
        
    def _cancel_wait_timer(self, route_stop: RouteStop) -> None:
        """
        Cancel a wait timer for a vehicle at a stop
        
        Args:
            route_stop: The route stop where the vehicle is waiting
        """
        # Get the event ID to cancel
        event_id = route_stop.cancel_wait_timer()
        
        if event_id:
            logger.debug(f"Cancelling wait timer {event_id} at stop {route_stop.id}")
            # Cancel the event in the queue
            success = self.context.event_manager.cancel_event(event_id)
            if success:
                logger.info(f"Successfully cancelled wait timer {event_id} at stop {route_stop.id}")
            else:
                logger.warning(f"Failed to cancel wait timer {event_id} at stop {route_stop.id}")

    def _cancel_pending_vehicle_events(self, vehicle_id: str) -> int:
        """
        Cancel all pending movement-related events for a vehicle.
        
        Args:
            vehicle_id: The ID of the vehicle
            
        Returns:
            int: Number of events canceled
        """
        logger.debug(f"=== START CANCELLING PENDING EVENTS ===")
        logger.debug(f"Looking for pending movement events to cancel for vehicle {vehicle_id}")
        
        # Get all pending events from the event queue
        events = self.context.event_manager.get_all_events()
        
        # Define all event types to cancel
        movement_event_types = [
            EventType.VEHICLE_POSITION_UPDATE,
            EventType.VEHICLE_ARRIVED_STOP,
        ]
        
        # Find events related to this vehicle's movement
        events_to_cancel = []
        for event in events:
            if event.vehicle_id == vehicle_id and event.event_type in movement_event_types:
                events_to_cancel.append(event.id)
                logger.debug(f"Found event to cancel: Type={event.event_type.value}, ID={event.id}, Time={event.timestamp}")
        
        # Cancel each event
        canceled_count = 0
        for event_id in events_to_cancel:
            success = self.context.event_manager.cancel_event(event_id)
            if success:
                logger.debug(f"Successfully canceled event {event_id}")
                canceled_count += 1
            else:
                logger.warning(f"Failed to cancel event {event_id}")
        
        logger.info(f"Canceled {canceled_count} events for vehicle {vehicle_id}")
        logger.debug(f"=== END CANCELLING PENDING EVENTS ===")
        return canceled_count
            
    def _create_stop_operations_completed_event(self, vehicle_id: str, route_stop_id: str, current_time: datetime) -> None:
        """
        Create an event for stop operations completed
        
        Args:
            vehicle_id: ID of the vehicle
            route_stop_id: ID of the route stop
            current_time: Current simulation time
        """
        logger.debug(f"Creating stop operations completed event for vehicle {vehicle_id} at stop {route_stop_id}")
        
        event = Event(
            event_type=EventType.VEHICLE_STOP_OPERATIONS_COMPLETED,
            priority=EventPriority.HIGH,
            timestamp=current_time,
            vehicle_id=vehicle_id,
            data={
                'route_stop_id': route_stop_id
            }
        )
        self.context.event_manager.publish_event(event)
        logger.info(f"Created stop operations completed event {event.id} for vehicle {vehicle_id} at stop {route_stop_id}")

    def _handle_passenger_alighting(self, route_stop: RouteStop, vehicle_id: str) -> None:
        """
        Handle passenger alighting (dropoff) at a stop
        
        Args:
            route_stop: The route stop where alighting is happening
            vehicle_id: ID of the vehicle
        """
        if not route_stop.dropoff_passengers:
            logger.debug(f"No dropoff passengers at route stop {route_stop.id}")
            return
            
        logger.debug(f"=== START PASSENGER ALIGHTING ===")
        logger.debug(f"Processing {len(route_stop.dropoff_passengers)} dropoffs at route stop {route_stop.id}")
            
        # Process each dropoff
        for request_id in route_stop.dropoff_passengers:
            # Get passenger ID from request
            request = self.state_manager.request_worker.get_request(request_id)
            if not request:
                logger.warning(f"Request {request_id} not found for alighting")
                continue
                
            logger.debug(f"Processing alighting for passenger {request.passenger_id}, request {request_id}")
                
            # Register dropoff with route stop
            route_stop.register_dropoff(request_id)
            self.state_manager.vehicle_worker.decrement_vehicle_occupancy(vehicle_id)
            
            # Create alighting completed event
            alighting_event = Event(
                event_type=EventType.PASSENGER_ALIGHTING_COMPLETED,
                priority=EventPriority.HIGH,
                timestamp=self.context.current_time,
                vehicle_id=vehicle_id,
                passenger_id=request.passenger_id,
                request_id=request_id,
                data={
                    'route_stop_id': route_stop.id,
                    'stop_id': route_stop.stop.id,
                    'location': route_stop.stop.location.to_dict()
                }
            )
            self.context.event_manager.publish_event(alighting_event)
            logger.info(f"Created alighting event {alighting_event.id} for passenger {request.passenger_id} (request {request_id}) at route stop {route_stop.id}")
        
        # Check if dropoffs are complete
        dropoff_complete = route_stop.is_dropoff_complete()
        logger.debug(f"Dropoff status: Complete={dropoff_complete}, "
                f"CompletedDropoffs={len(route_stop.completed_dropoff_request_ids)}/{len(route_stop.dropoff_passengers)}")
        logger.debug(f"=== END PASSENGER ALIGHTING ===")

    def _check_delay_violations(self, route_stop: RouteStop, vehicle_id: str, arrival_time: datetime) -> None:
        """
        Check for delay violations at a stop
        
        Args:
            route_stop: The route stop to check
            vehicle_id: ID of the vehicle
            arrival_time: Time of arrival
        """
        # Skip if no planned arrival time
        if not route_stop.planned_arrival_time:
            logger.debug(f"No planned arrival time for route stop {route_stop.id}, skipping delay check")
            return
            
        # Calculate delay
        delay = (arrival_time - route_stop.planned_arrival_time).total_seconds()
        
        logger.debug(f"Checking delay at route stop {route_stop.id}: PlannedArrival={route_stop.planned_arrival_time}, "
                f"ActualArrival={arrival_time}, Delay={delay:.1f}s, MaxThreshold={self.vehicle_thresholds['max_pickup_delay']}s")
        
        # Check if delay exceeds threshold
        if delay > self.vehicle_thresholds['max_pickup_delay']:
            # Log violation
            logger.warning(f"DELAY VIOLATION: Vehicle {vehicle_id} arrived at route stop {route_stop.id} with delay of {delay:.1f} seconds (exceeds threshold of {self.vehicle_thresholds['max_pickup_delay']}s)")
        else:
            logger.debug(f"Delay is within acceptable limits")
            