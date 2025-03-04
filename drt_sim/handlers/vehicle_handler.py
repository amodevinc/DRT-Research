from datetime import datetime, timedelta
from typing import Dict, Any, Optional
from drt_sim.core.state.manager import StateManager
from drt_sim.core.simulation.context import SimulationContext
from drt_sim.models.event import Event, EventType, EventPriority
from drt_sim.models.vehicle import VehicleStatus
from drt_sim.config.config import ParameterSet
from drt_sim.models.route import RouteSegment, RouteStatus, RouteStop, Route
from drt_sim.models.stop import StopPurpose
from drt_sim.models.passenger import PassengerStatus
from drt_sim.core.monitoring.types.metrics import MetricName
from drt_sim.core.coordination.stop_coordinator import StopCoordinator
import traceback
import logging
logger = logging.getLogger(__name__)

class VehicleHandler:
    """
    Handles vehicle operations in the DRT system.
    Enhanced with better state management for route modifications and segment processing.
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
        self.vehicle_thresholds = self._setup_vehicle_thresholds()
        self.stop_coordinator = stop_coordinator
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
        Handle vehicle dispatch request with improved state handling.
        Implements a lock-based approach to prevent race conditions.
        """
        vehicle_id = event.vehicle_id
        
        # Acquire lock for this vehicle to prevent concurrent operations
        async with self._get_vehicle_lock(vehicle_id):
            try:
                logger.debug(f"=== START VEHICLE DISPATCH HANDLING ===")
                logger.debug(f"Vehicle {vehicle_id} - Processing dispatch request at time: {self.context.current_time}")
                
                # Begin transaction
                self.state_manager.begin_transaction()
                
                route_id = event.data.get('route_id')
                
                vehicle = self.state_manager.vehicle_worker.get_vehicle(vehicle_id)
                if not vehicle:
                    raise ValueError(f"Vehicle {vehicle_id} not found")
                
                route = self.state_manager.route_worker.get_route(route_id)
                if not route:
                    raise ValueError(f"No route found with ID {route_id}")
                
                # Cancel any pending events for this vehicle before creating new ones
                self._cancel_pending_vehicle_events(vehicle_id)
                
                # Update vehicle status to IN_SERVICE
                self.state_manager.vehicle_worker.update_vehicle_status(
                    vehicle_id,
                    VehicleStatus.IN_SERVICE,
                    self.context.current_time
                )
                
                # Validate route state
                if not route.segments:
                    raise ValueError(f"Route {route_id} has no segments")
                
                # For initial dispatch, we need to ensure the current segment index is set correctly
                # This is one place where recalc_current_segment_index is truly necessary
                route.recalc_current_segment_index()
                
                # Get current segment using Route's method
                current_segment = route.get_current_segment()
                if not current_segment:
                    raise ValueError(f"No current segment found in route {route_id}")
                
                # Update route status if it's just starting
                if route.status in [RouteStatus.CREATED, RouteStatus.PLANNED]:
                    route.status = RouteStatus.ACTIVE
                    route.actual_start_time = self.context.current_time
                    
                # Use Route's methods to mark segment as active
                current_segment.mark_in_progress(vehicle.current_state.current_location)
                route.set_active_segment(current_segment.id, vehicle.current_state.current_location)
                
                # Update route state
                self.state_manager.route_worker.update_route(route)
                
                # Update vehicle's active route
                logger.info(f"Updating vehicle {vehicle_id} active route to {route_id}")
                self.state_manager.vehicle_worker.update_vehicle_active_route_id(
                    vehicle_id,
                    route_id
                )
                
                # Create movement event for the current segment
                logger.info(f"Creating movement event for vehicle {vehicle_id} on segment {current_segment.id}")
                self._create_vehicle_movement_event(
                    vehicle_id=vehicle_id,
                    segment=current_segment,
                    current_time=self.context.current_time
                )
                
                self.state_manager.commit_transaction()
                logger.info(f"Successfully completed dispatch for vehicle {vehicle_id} with route {route_id}")
                
            except Exception as e:
                self.state_manager.rollback_transaction()
                logger.error(f"Error handling vehicle dispatch: {traceback.format_exc()}")
                self._handle_vehicle_error(event, str(e))

    async def handle_vehicle_position_update(self, event: Event) -> None:
        """
        Handle intermediate position updates as vehicle moves between stops.
        Enhanced to update segment progress information.
        """
        vehicle_id = event.vehicle_id
        
        # We don't need a lock for position updates as they're non-critical
        try:
            self.state_manager.begin_transaction()
            
            vehicle = self.state_manager.vehicle_worker.get_vehicle(vehicle_id)
            if not vehicle:
                logger.warning(f"Vehicle {vehicle_id} not found for position update")
                self.state_manager.rollback_transaction()
                return
                
            # Get current route
            route_id = self._get_active_route_id(vehicle_id)
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
            segment_id = event.data.get('segment_id')
            
            # Update vehicle's current location
            self.state_manager.vehicle_worker.update_vehicle_location(
                vehicle_id,
                new_location,
                distance_covered
            )
            
            # Update segment progress if segment_id is provided
            if segment_id:
                # Find the active segment
                active_segment = None
                for segment in route.segments:
                    if segment.id == segment_id:
                        active_segment = segment
                        break
                
                if active_segment:
                    # Update progress
                    active_segment.update_progress(progress_percentage, new_location)
                    
                    # Ensure this segment is marked as the active one in the route
                    if route.active_segment_id != segment_id:
                        route.set_active_segment(segment_id, new_location)
                        
            # Update route in state
            self.state_manager.route_worker.update_route(route)
            
            self.state_manager.commit_transaction()
            
        except Exception as e:
            self.state_manager.rollback_transaction()
            logger.error(f"Error handling vehicle position update: {traceback.format_exc()}")
            self._handle_vehicle_error(event, str(e))

    async def handle_vehicle_arrived_stop(self, event: Event) -> None:
        """
        Handle vehicle arrival at a stop.
        Enhanced with better event validation and segment state tracking.
        """
        vehicle_id = event.vehicle_id
        
        # Acquire lock for this vehicle to prevent concurrent operations
        async with self._get_vehicle_lock(vehicle_id):
            try:
                logger.info(f"Processing vehicle arrival for event: {event.id}")
                
                # Begin transaction
                self.state_manager.begin_transaction()
                
                vehicle = self.state_manager.vehicle_worker.get_vehicle(vehicle_id)
                
                if not vehicle:
                    raise ValueError(f"Vehicle {vehicle_id} not found")
                
                logger.debug(f"Vehicle {vehicle_id} arrived at stop:")
                logger.debug(f"Current state: status={vehicle.current_state.status}, "
                          f"location={vehicle.current_state.current_location}, "
                          f"occupancy={vehicle.current_state.current_occupancy}")
                logger.debug(f"Arrival details: origin={event.data.get('origin')}, "
                          f"destination={event.data.get('destination')}, "
                          f"movement_start_time={event.data.get('movement_start_time')}")
                
                # Get current route
                current_route_id = self._get_active_route_id(vehicle_id)
                if not current_route_id:
                    logger.warning(f"Vehicle {vehicle_id} has no active route during arrival")
                    self.state_manager.rollback_transaction()
                    return
                    
                route = self.state_manager.route_worker.get_route(current_route_id)
                if not route:
                    logger.warning(f"Route {current_route_id} not found during arrival")
                    self.state_manager.rollback_transaction()
                    return
                
                # Check if this is a rebalancing arrival or a regular stop arrival
                if event.data.get('is_rebalancing', False):
                    self._handle_rebalancing_arrival(vehicle_id, event)
                else:
                    try:
                        await self._handle_regular_stop_arrival(vehicle_id, event, route)
                    except Exception as e:
                        if "already at stop" in str(e):
                            # This is a stale event, log and ignore
                            logger.warning(f"Ignoring stale arrival event for vehicle {vehicle_id}: {str(e)}")
                            self.state_manager.rollback_transaction()
                            return
                        else:
                            # Re-raise other exceptions
                            raise
                
                # Commit the transaction
                self.state_manager.commit_transaction()
                
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

    async def _handle_regular_stop_arrival(self, vehicle_id: str, event: Event, route: Route) -> None:
        """
        Handle regular stop arrival with improved segment state management.
        
        Args:
            vehicle_id: ID of the vehicle
            event: Arrival event
            route: Current route object
        """
        logger.debug(f"=== START STOP ARRIVAL HANDLING ===")
        logger.debug(f"Vehicle {vehicle_id} - Processing stop arrival at time: {self.context.current_time}")
        
        vehicle = self.state_manager.vehicle_worker.get_vehicle(vehicle_id)
        if not vehicle:
            raise ValueError(f"Vehicle {vehicle_id} not found")
            
        # Check if the vehicle is already at a stop - this could indicate a stale event
        if vehicle.current_state.status == VehicleStatus.AT_STOP:
            logger.warning(f"[Stop Arrival] Vehicle {vehicle_id} is already at stop {vehicle.current_state.current_stop_id} - This may be a stale event")
            raise ValueError(f"Vehicle {vehicle_id} is already at stop {vehicle.current_state.current_stop_id}")
        
        # Update vehicle location first
        destination_location = event.data.get('destination')
        self.state_manager.vehicle_worker.update_vehicle_location(
            vehicle_id,
            destination_location
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
        
        # Validate route segments
        logger.debug(f"Route state: {str(route)}")
        
        # Find current segment using segment_id from event or active segment from route
        segment_id = event.data.get('segment_id')
        current_segment = None
        
        if segment_id:
            # Try to find segment by ID first
            for segment in route.segments:
                if segment.id == segment_id:
                    current_segment = segment
                    break
        
        # If not found by ID, use Route's method to get active segment
        if not current_segment:
            current_segment = route.get_active_segment()
            
        # If still not found, let Route recalculate and get current segment
        if not current_segment:
            route.recalc_current_segment_index()
            current_segment = route.get_current_segment()
            
        # Final validation
        if not current_segment:
            raise ValueError(f"Could not determine current segment for vehicle {vehicle_id}")
        
        # Get the destination stop from the segment
        current_stop = current_segment.destination
        if not current_stop:
            raise ValueError(f"Current segment has no destination stop")

        logger.debug(f"Vehicle arrived at {str(current_stop)}")
        
        # Mark the segment as completed using Route's method
        movement_start_time = event.data.get('movement_start_time')
        actual_duration = None
        if movement_start_time:
            actual_duration = (self.context.current_time - movement_start_time).total_seconds()
            
        # Use Route's method to mark segment completed
        route.mark_segment_completed(segment_id=current_segment.id, actual_duration=actual_duration, actual_distance=distance)
        
        # Update vehicle state to AT_STOP
        self.state_manager.vehicle_worker.update_vehicle_status(
            vehicle_id,
            VehicleStatus.AT_STOP,
            self.context.current_time
        )
        
        # Update additional vehicle state
        vehicle.current_state.current_stop_id = current_stop.stop.id
        vehicle.current_state.waiting_for_passengers = True
        self.state_manager.vehicle_worker.update_vehicle(vehicle)
        
        # Mark the stop as arrived
        current_stop.actual_arrival_time = self.context.current_time
        
        # Update route in state
        self.state_manager.route_worker.update_route(route)
        
        # Register vehicle arrival with coordinator AFTER updating vehicle state
        self.stop_coordinator.register_vehicle_arrival(
            stop_id=current_stop.stop.id,
            vehicle_id=vehicle_id,
            arrival_time=self.context.current_time,
            location=current_stop.stop.location,
            event=event,
            segment_id=current_segment.id
        )
        
        # Update metrics
        if self.context.metrics_collector:
            self.context.metrics_collector.log(
                MetricName.VEHICLE_STOPS_SERVED,
                1,
                self.context.current_time,
                { 'vehicle_id': vehicle_id }
            )

    async def handle_vehicle_reroute_request(self, event: Event) -> None:
        """
        Handle rerouting request for vehicle already in service.
        Enhanced with better state tracking and atomicity.
        Ensures proper handling of vehicles waiting at stops.
        """
        vehicle_id = event.vehicle_id
        
        # Acquire lock for this vehicle to prevent concurrent operations
        async with self._get_vehicle_lock(vehicle_id):
            try:
                logger.info(f"Processing vehicle reroute request for vehicle {vehicle_id} with route: {event.data['route_id']}")
                
                # Begin transaction
                self.state_manager.begin_transaction()
                
                vehicle = self.state_manager.vehicle_worker.get_vehicle(vehicle_id)
                route_id = event.data['route_id']
                
                if not vehicle:
                    raise ValueError(f"Vehicle {vehicle_id} not found")
                
                # Get current vehicle state and active route
                current_route_id = self._get_active_route_id(vehicle_id)
                
                # Get the new route
                new_route = self.state_manager.route_worker.get_route(route_id)
                if not new_route:
                    raise ValueError(f"New route {route_id} not found")
                
                # Get current route for later comparison
                current_route = None
                if current_route_id and current_route_id != route_id:
                    current_route = self.state_manager.route_worker.get_route(current_route_id)
                    logger.info(f"Vehicle {vehicle_id} is being rerouted from route {current_route_id} to route {route_id}")
                
                # Check if vehicle is currently at a stop waiting for passengers
                is_at_stop_waiting = (vehicle.current_state.status == VehicleStatus.AT_STOP)

                logger.info(f"DIAGNOSTIC: Reroute request for vehicle {vehicle_id} with status {vehicle.current_state.status}")
                logger.info(f"DIAGNOSTIC: Current route ID: {current_route_id}, New route ID: {route_id}")
                logger.info(f"DIAGNOSTIC: Vehicle at stop: {is_at_stop_waiting}, Current stop ID: {vehicle.current_state.current_stop_id}")
                
                # Cancel any pending movement events for this vehicle
                self._cancel_pending_vehicle_events(vehicle_id)
                
                # If vehicle is at a stop, let it complete operations before following new route
                if is_at_stop_waiting:
                    # Get current stop ID
                    current_stop_id = vehicle.current_state.current_stop_id
                    logger.info(f"Vehicle {vehicle_id} is at stop {current_stop_id}. " 
                            f"Updating route assignment to {route_id}, but operations at current stop "
                            f"will complete before following new route.")
                    
                    # Store current stop related info that needs to be preserved
                    current_waiting_state = vehicle.current_state.waiting_for_passengers
                    
                    # Update vehicle's active route ID to new route
                    self.state_manager.vehicle_worker.update_vehicle_active_route_id(vehicle_id, route_id)
                    
                    # Preserve the vehicle's at-stop state to ensure stop operations complete properly
                    vehicle.current_state.waiting_for_passengers = current_waiting_state
                    vehicle.current_state.status = VehicleStatus.AT_STOP
                    self.state_manager.vehicle_worker.update_vehicle(vehicle)
                    
                    # Mark the route as modified
                    new_route.mark_as_modified()
                    
                    # Update route in state
                    self.state_manager.route_worker.update_route(new_route)
                    
                    # Notify the stop coordinator about the route change
                    if self.stop_coordinator:
                        self.stop_coordinator.handle_route_change_at_stop(vehicle_id, current_stop_id, route_id)
                    
                    # Operations at the stop will continue, and handle_stop_operations_completed
                    # will pick up the new route when stop operations finish
                    self.state_manager.commit_transaction()
                    return
                
                # For vehicles not at a stop, proceed with normal rerouting
                
                # Get current vehicle location
                current_location = vehicle.current_state.current_location
                
                # Use Route's method to update segments for reroute
                new_route.update_segments_for_reroute(current_location)
                
                # Get active segment from the new route
                active_segment = new_route.get_active_segment()
                
                # If no active segment could be determined, recalculate
                if not active_segment:
                    new_route.recalc_current_segment_index()
                    active_segment = new_route.get_current_segment()
                    if active_segment:
                        active_segment.mark_in_progress(current_location)
                        new_route.set_active_segment(active_segment.id, current_location)
                
                # Validate that we have a valid segment
                if not active_segment:
                    raise ValueError(f"No valid active segment found in route {route_id}")
                
                # Update vehicle's active route
                self.state_manager.vehicle_worker.update_vehicle_active_route_id(vehicle_id, route_id)
                
                # Update route in state
                self.state_manager.route_worker.update_route(new_route)
                
                # Create movement event for current segment
                self._create_vehicle_movement_event(
                    vehicle_id=vehicle_id,
                    segment=active_segment,
                    current_time=self.context.current_time
                )
                
                self.state_manager.commit_transaction()
                logger.info(f"Successfully rerouted vehicle {vehicle_id} to route {route_id}")
                
            except Exception as e:
                self.state_manager.rollback_transaction()
                logger.error(f"Error handling vehicle reroute: {traceback.format_exc()}")
                self._handle_vehicle_error(event, str(e))

    async def handle_stop_operations_completed(self, event: Event) -> None:
        """
        Handle completion of all operations at a stop.
        Enhanced with better handling of route changes that occurred during stop operations.
        """
        vehicle_id = event.vehicle_id
        
        # Acquire lock for this vehicle to prevent concurrent operations
        async with self._get_vehicle_lock(vehicle_id):
            try:
                logger.debug(f"=== START STOP OPERATIONS COMPLETION HANDLING ===")
                logger.debug(f"Vehicle {vehicle_id} - Processing stop completion at time: {self.context.current_time}")
                
                self.state_manager.begin_transaction()
                
                event_route_id = event.data.get('route_id')
                segment_id = event.data.get('segment_id')
                stop_id = event.data.get('stop_id')
                
                vehicle = self.state_manager.vehicle_worker.get_vehicle(vehicle_id)
                if not vehicle:
                    raise ValueError(f"Vehicle {vehicle_id} not found")
                
                # Get the current route assigned to the vehicle (which might have changed during stop operations)
                current_route_id = self._get_active_route_id(vehicle_id)

                logger.info(f"DIAGNOSTIC: Stop operations completed for vehicle {vehicle_id}")
                logger.info(f"DIAGNOSTIC: Event route ID: {event_route_id}, Current route ID: {current_route_id}")
                logger.info(f"DIAGNOSTIC: Vehicle status: {vehicle.current_state.status}, At stop: {vehicle.current_state.current_stop_id}")

                if not current_route_id:
                    # Handle edge case where vehicle has no assigned route
                    logger.warning(f"Vehicle {vehicle_id} has no active route after stop operations")
                    # Update vehicle status to IDLE
                    self.state_manager.vehicle_worker.update_vehicle_status(
                        vehicle_id, 
                        VehicleStatus.IDLE,
                        self.context.current_time
                    )
                    # Clear stop-related state
                    vehicle.current_state.current_stop_id = None
                    vehicle.current_state.waiting_for_passengers = False
                    self.state_manager.vehicle_worker.update_vehicle(vehicle)
                    self.state_manager.commit_transaction()
                    return
                
                # Check if route has changed during stop operations
                route_changed = (event_route_id and event_route_id != current_route_id)
                if route_changed:
                    logger.info(f"Route has changed during stop operations: {event_route_id} -> {current_route_id}")
                
                # Use the vehicle's currently assigned route
                route_id = current_route_id
                
                # Get the current route
                route = self.state_manager.route_worker.get_route(route_id)
                if not route:
                    raise ValueError(f"Route {route_id} not found")

                # Mark the completed stop
                completed_stop_id = stop_id or vehicle.current_state.current_stop_id
                if completed_stop_id:
                    # Use Route's mark_stop_completed method if available, otherwise find and mark manually
                    route.mark_stop_completed(completed_stop_id, self.context.current_time, self.context.current_time)
                
                # Check if route is completed using Route's method
                if route.is_completed():
                    logger.debug(f"Vehicle {vehicle_id} - Route {route_id} is completed")
                    route = self._handle_route_completion(vehicle_id, route, self.context.current_time)
                    self.state_manager.vehicle_worker.update_vehicle_active_route_id(vehicle.id, None)
                    
                    # Clear stop-related state
                    self.state_manager.vehicle_worker.update_vehicle_status(
                        vehicle_id,
                        VehicleStatus.IDLE,
                        self.context.current_time
                    )
                    # Update additional vehicle state
                    vehicle = self.state_manager.vehicle_worker.get_vehicle(vehicle_id)
                    if vehicle:
                        vehicle.current_state.current_stop_id = None
                        vehicle.current_state.waiting_for_passengers = False
                        self.state_manager.vehicle_worker.update_vehicle(vehicle)
                else:
                    # Continue with next segment in route
                    logger.debug(f"Vehicle {vehicle_id} - Continuing with route {route_id}")
                    
                    # Clear stop-related state
                    vehicle.current_state.current_stop_id = None
                    vehicle.current_state.waiting_for_passengers = False
                    self.state_manager.vehicle_worker.update_vehicle(vehicle)
                    
                    # Continue with the route
                    route = await self._handle_route_continuation(vehicle_id, route, self.context.current_time)
                
                # Update route in state
                self.state_manager.route_worker.update_route(route)
                
                self.state_manager.commit_transaction()
                
            except Exception as e:
                self.state_manager.rollback_transaction()
                logger.error(f"Error handling stop operations completion: {traceback.format_exc()}")
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
        Handle continuation to next segment in route.
        Enhanced with better segment tracking and state management.
        
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
        if not route.segments:
            logger.error(f"[Route Continuation] Vehicle {vehicle_id} - Route has no segments")
            raise ValueError("Route has no segments")
        
        # Let Route model handle finding the next segment
        route.recalc_current_segment_index()
        
        # Get next segment using Route's method
        next_segment = route.get_current_segment()
        if not next_segment:
            logger.error(f"[Route Continuation] Vehicle {vehicle_id} - No next segment found")
            raise ValueError("No next segment found")
        
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
        
        # Use Route's methods to mark segment as active
        next_segment.mark_in_progress(vehicle.current_state.current_location)
        route.set_active_segment(next_segment.id, vehicle.current_state.current_location)
        
        # Create movement event for next segment
        self._create_vehicle_movement_event(
            vehicle_id=vehicle_id,
            segment=next_segment,
            current_time=current_time
        )
        
        return route

    def _create_vehicle_movement_event(
        self,
        vehicle_id: str,
        segment: RouteSegment,
        current_time: datetime
    ) -> None:
        """
        Create events for vehicle movement along a route segment with waypoint updates.
        Enhanced with better state tracking.
        
        Args:
            vehicle_id: ID of the vehicle
            segment: Segment to create movement events for
            current_time: Current time
        """
        logger.debug(f"=== START MOVEMENT EVENT CREATION ===")
        logger.debug(f"Vehicle {vehicle_id} - Creating movement events at time: {current_time}")
        logger.debug(f"Segment details: {str(segment)}")
        
        if not segment:
            logger.error(f"[Movement Event] Vehicle {vehicle_id} - Segment is None")
            raise ValueError("Cannot create movement event for None segment")
            
        # Get origin and destination locations
        origin_location = (
            segment.origin.stop.location if segment.origin 
            else segment.origin_location
        )
        destination_location = (
            segment.destination.stop.location if segment.destination 
            else segment.destination_location
        )
        
        # Get waypoints from segment
        waypoints = segment.waypoints if segment.waypoints else []
        
        # Calculate total segment duration and distance
        total_duration = segment.estimated_duration
        total_distance = segment.estimated_distance
        
        # Mark segment as in progress
        segment.mark_in_progress(origin_location)
        segment.movement_start_time = current_time
        
        # If we have waypoints, create intermediate position update events (max 4)
        position_update_events = []
        if waypoints:
            # Limit to a maximum of 4 position updates
            max_updates = min(4, len(waypoints))
            
            # Select evenly spaced waypoints
            if len(waypoints) > max_updates:
                step = len(waypoints) / max_updates
                selected_indices = [int(i * step) for i in range(max_updates)]
                selected_waypoints = [waypoints[i] for i in selected_indices]
            else:
                selected_waypoints = waypoints
            
            # Calculate time intervals for updates
            time_between_updates = total_duration / (max_updates + 1)
            distance_between_updates = total_distance / (max_updates + 1)
            
            current_distance = 0
            for i, waypoint in enumerate(selected_waypoints):
                update_time = current_time + timedelta(seconds=(i+1) * time_between_updates)
                current_distance += distance_between_updates
                progress_percentage = ((i+1) / (max_updates + 1)) * 100
                
                # Create position update event
                update_event = Event(
                    event_type=EventType.VEHICLE_POSITION_UPDATE,
                    priority=EventPriority.HIGH,
                    timestamp=update_time,
                    vehicle_id=vehicle_id,
                    data={
                        'segment_id': segment.id,
                        'location': waypoint,
                        'progress_percentage': progress_percentage,
                        'distance_covered': current_distance,
                        'movement_start_time': current_time
                    }
                )
                position_update_events.append(update_event)
        
        # Create final arrival event
        arrival_time = current_time + timedelta(seconds=total_duration)
        arrival_event = Event(
            event_type=EventType.VEHICLE_ARRIVED_STOP,
            priority=EventPriority.HIGH,
            timestamp=arrival_time,
            vehicle_id=vehicle_id,
            data={
                'segment_id': segment.id,
                'movement_start_time': current_time,
                'origin': origin_location,
                'destination': destination_location,
                'actual_duration': total_duration,
                'actual_distance': total_distance,
                'stop_id': segment.destination.id if segment.destination else None
            }
        )
        
        # Publish events in order
        for event in position_update_events:
            self.context.event_manager.publish_event(event)
        
        self.context.event_manager.publish_event(arrival_event)

    def handle_vehicle_wait_timeout(self, event: Event) -> None:
        """Delegate wait timeout handling to StopCoordinator."""
        try:
            # Simply delegate to stop coordinator
            self.stop_coordinator.handle_wait_timeout(event)
        except Exception as e:
            logger.error(f"Error handling wait timeout: {traceback.format_exc()}")
            self._handle_vehicle_error(event, str(e))

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
    
    def _cancel_pending_vehicle_events(self, vehicle_id: str) -> int:
        """
        Cancel all pending movement-related events for a vehicle.
        Enhanced with more comprehensive event cancellation.
        
        Args:
            vehicle_id: The ID of the vehicle
            
        Returns:
            int: Number of events canceled
        """
        logger.info(f"Canceling pending movement events for vehicle {vehicle_id}")
        
        # Get all pending events from the event queue
        events = list(self.context.event_manager.event_queue.queue)
        
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
                logger.info(f"Found event to cancel: {event.event_type.value} at {event.timestamp}")
        
        # Cancel each event
        canceled_count = 0
        for event_id in events_to_cancel:
            success = self.context.event_manager.cancel_event(event_id)
            if success:
                logger.info(f"Successfully canceled event {event_id}")
                canceled_count += 1
            else:
                logger.warning(f"Failed to cancel event {event_id}")
        
        logger.info(f"Canceled {canceled_count} events for vehicle {vehicle_id}")
        return canceled_count
        
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