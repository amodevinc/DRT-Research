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
from drt_sim.core.coordination.stop_coordinator import StopCoordinator
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
        state_manager: StateManager,
        stop_coordinator: StopCoordinator
    ):
        self.config = config
        self.context = context
        self.state_manager = state_manager
        self.vehicle_thresholds = self._setup_vehicle_thresholds()
        self.stop_coordinator = stop_coordinator
    def _setup_vehicle_thresholds(self) -> Dict[str, Any]:
        """Initialize vehicle operation thresholds from config"""
        return {
            'max_pickup_delay': self.config.vehicle.max_pickup_delay,
            'max_dropoff_delay': self.config.vehicle.max_dropoff_delay
        }

    def handle_vehicle_dispatch_request(self, event: Event) -> None:
        """Handle vehicle dispatch request."""
        try:
            logger.debug(f"=== START VEHICLE DISPATCH HANDLING ===")
            logger.debug(f"Vehicle {event.vehicle_id} - Processing dispatch request at time: {self.context.current_time}")
            
            # Use a single transaction for all operations to ensure atomicity
            self.state_manager.begin_transaction()
            
            vehicle_id = event.vehicle_id
            route_id = event.data.get('route_id')
            
            vehicle = self.state_manager.vehicle_worker.get_vehicle(vehicle_id)
            if not vehicle:
                raise ValueError(f"Vehicle {vehicle_id} not found")
            
            route = self.state_manager.route_worker.get_route(route_id)
            if not route:
                raise ValueError(f"No route found with ID {route_id}")
            
            # IMPORTANT: Cancel any pending events for this vehicle before creating new ones
            # This is critical to prevent conflicts, especially during initial dispatch
            self._cancel_pending_vehicle_events(vehicle_id)
            
            # Update vehicle status to IN_SERVICE
            self.state_manager.vehicle_worker.update_vehicle_status(
                vehicle_id,
                VehicleStatus.IN_SERVICE,
                self.context.current_time
            )
            
            # Get current segment
            current_segment = route.get_current_segment()
            if not current_segment:
                raise ValueError(f"No current segment found in route {route_id}")
            
            # Update route status if it's just starting
            if route.status == RouteStatus.CREATED or route.status == RouteStatus.PLANNED:
                route.status = RouteStatus.ACTIVE
                route.actual_start_time = self.context.current_time
                self.state_manager.route_worker.update_route(route)
                logger.info(f"Activated route {route_id}")
            
            # Update vehicle's active route and status
            logger.info(f"Updating vehicle {vehicle_id} active route to {route_id}")
            self.state_manager.vehicle_worker.update_vehicle_active_route_id(
                vehicle_id,
                route_id
            )
            
            # Verify route update
            vehicle = self.state_manager.vehicle_worker.get_vehicle(vehicle_id)
            logger.info(f"Vehicle {vehicle_id} state after route update: {vehicle.current_state.to_dict()}")
            
            # Create movement event for the current segment
            logger.info(f"Creating movement event for vehicle {vehicle_id} on segment {current_segment.id}")
            self._create_vehicle_movement_event(
                vehicle_id=vehicle_id,
                segment=current_segment,
                current_time=self.context.current_time
            )
            
            self.state_manager.commit_transaction()
            logger.info(f"Successfully completed dispatch request for vehicle {vehicle_id} with route {route_id}")
            
        except Exception as e:
            self.state_manager.rollback_transaction()
            logger.error(f"Error handling vehicle dispatch: {str(e)}")
            self._handle_vehicle_error(event, str(e))

    def handle_vehicle_position_update(self, event: Event) -> None:
        """Handle intermediate position updates as vehicle moves between stops."""
        try:
            self.state_manager.begin_transaction()
            
            vehicle_id = event.vehicle_id
            new_location = event.data['location']
            distance_covered = event.data['distance_covered']
            
            # Update vehicle's current location
            self.state_manager.vehicle_worker.update_vehicle_location(
                vehicle_id,
                new_location,
                distance_covered
            )
            
            self.state_manager.commit_transaction()
            
        except Exception as e:
            self.state_manager.rollback_transaction()
            logger.error(f"Error handling vehicle position update: {str(e)}")
            self._handle_vehicle_error(event, str(e))

    def handle_vehicle_reroute_request(self, event: Event) -> None:
        """
        Handle rerouting request for vehicle already in service.
        The route has already been updated in state, so we just need to
        trigger the vehicle to start following the updated route.
        """
        try:
            logger.info(f"Processing vehicle reroute request for vehicle {event.vehicle_id} with route: {event.data['route_id']}")
            # Use a single transaction for all operations to ensure atomicity
            self.state_manager.begin_transaction()
            
            vehicle_id = event.vehicle_id
            vehicle = self.state_manager.vehicle_worker.get_vehicle(vehicle_id)
            route_id = event.data['route_id']
            
            if not vehicle:
                raise ValueError(f"Vehicle {vehicle_id} not found")
                
            # Validate route has segments
            route = self.state_manager.route_worker.get_route(route_id)
            if not route:
                raise ValueError(f"No active route found for vehicle {vehicle_id}")
                
            # Validate route has segments
            if not route.segments or len(route.segments) == 0:
                raise ValueError(f"Route {route_id} has no segments")
                
            logger.info(f"Updated route segments: {[str(seg) for seg in route.segments]}")
            
            try:
                logger.info(f"Vehicle {vehicle_id} current state: {str(vehicle.current_state)}")
                # Check if vehicle is currently at a stop waiting for passengers
                is_at_stop_waiting = (
                    vehicle.current_state.status == VehicleStatus.AT_STOP
                )
                # IMPORTANT: Cancel any pending movement events for this vehicle
                # This is critical to prevent duplicate arrivals and inconsistent state
                self._cancel_pending_vehicle_events(vehicle_id)
                
                route.recalc_current_segment_index()
                
                if is_at_stop_waiting:
                    logger.info(f"Vehicle {vehicle_id} is currently at stop waiting for passengers. "
                              f"Movement will be handled by stop coordinator after passenger operations complete.")
                    self.state_manager.commit_transaction()
                    return
                
               

                
                # Get current segment
                current_segment = route.get_current_segment()
                if not current_segment:
                    raise ValueError(f"No current segment found in route for vehicle {vehicle_id}")
                
                # Create movement event for current segment only if not at stop waiting
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
                
        except Exception as e:
            self.state_manager.rollback_transaction()
            logger.error(f"Error handling vehicle reroute: {str(e)}")
            self._handle_vehicle_error(event, str(e))

    def _cancel_pending_vehicle_events(self, vehicle_id: str) -> int:
        """
        Cancel all pending movement-related events for a vehicle.
        This is critical when rerouting to prevent duplicate arrivals and inconsistent state.
        
        Args:
            vehicle_id: The ID of the vehicle
            
        Returns:
            int: Number of events canceled
        """
        logger.info(f"Canceling pending movement events for vehicle {vehicle_id}")
        
        # Get all pending events from the event queue
        events = list(self.context.event_manager.event_queue.queue)
        
        # Find events related to this vehicle's movement
        events_to_cancel = []
        for event in events:
            if event.vehicle_id == vehicle_id and event.event_type in [
                EventType.VEHICLE_POSITION_UPDATE,
                EventType.VEHICLE_ARRIVED_STOP,
                EventType.VEHICLE_STOP_OPERATIONS_COMPLETED,
                EventType.VEHICLE_DISPATCH_REQUEST,  # Also cancel pending dispatch requests
                EventType.VEHICLE_REROUTE_REQUEST    # Also cancel pending reroute requests
            ]:
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

    def handle_vehicle_rebalancing_required(self, event: Event) -> None:
        """Handle rebalancing request to depot."""
        try:
            self.state_manager.begin_transaction()
            
            vehicle_id = event.vehicle_id

            logger.info(f"Rebalancing Implementation Required for rebalancing vehicles")
            
            self.state_manager.commit_transaction()
            
        except Exception as e:
            self.state_manager.rollback_transaction()
            logger.error(f"Error handling rebalancing request: {str(e)}")
            self._handle_vehicle_error(event, str(e))

    def handle_vehicle_arrived_stop(self, event: Event) -> None:
        """Handle vehicle arrival at a stop."""
        try:
            logger.info(f"Processing vehicle arrival for event: {event.id}")
            
            # Use a single transaction for all operations to ensure atomicity
            self.state_manager.begin_transaction()
            
            vehicle_id = event.vehicle_id
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
            
            # Check if this is a rebalancing arrival or a regular stop arrival
            if event.data.get('is_rebalancing', False):
                self._handle_rebalancing_arrival(vehicle_id, event)
            else:
                try:
                    self._handle_regular_stop_arrival(vehicle_id, event)
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
        
        # Update vehicle status to IDLE
        self.state_manager.vehicle_worker.update_vehicle_status(
            vehicle_id,
            VehicleStatus.IDLE,
            self.context.current_time
        )

    def _handle_regular_stop_arrival(self, vehicle_id: str, event: Event) -> None:
        """Handle regular stop arrival."""
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
        self.state_manager.vehicle_worker.update_vehicle_location(
            vehicle_id,
            event.data.get('destination')
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
        
        active_route_id = self.state_manager.vehicle_worker.get_vehicle_active_route_id(vehicle.id)
        route = self.state_manager.route_worker.get_route(active_route_id)
        if not route:
            logger.error(f"[Stop Arrival] Vehicle {vehicle_id} - No active route found")
            raise ValueError("No active route found")
        
        logger.debug(f"Route state: {str(route)}")
        logger.debug(f"Current segment index: {route.current_segment_index}")
        logger.debug(f"Total segments: {len(route.segments)}")
        logger.debug(f"All segments: {[str(seg) for seg in route.segments]}")
        
        current_segment = route.get_current_segment()
        if not current_segment:
            # Try to recalculate the current segment index
            route.recalc_current_segment_index()
            current_segment = route.get_current_segment()
            
            if not current_segment:
                logger.error(f"[Stop Arrival] Vehicle {vehicle_id} - No current segment found. Route details: {str(route)}")
                raise ValueError("No current segment found in route")
        
        # Check if the segment in the event matches the current segment in the route
        segment_id = event.data.get('segment_id')
        if segment_id and segment_id != current_segment.id:
            logger.warning(f"[Stop Arrival] Vehicle {vehicle_id} - Segment ID in event ({segment_id}) doesn't match current segment ({current_segment.id}) - This may be a stale event")
            # We'll continue processing with the current segment from the route
            logger.info(f"Using current segment {current_segment.id} instead of segment from event {segment_id}")
        
        current_stop = current_segment.destination

        logger.debug(f"Vehicle arrived at {str(current_stop)}")

        
        # Update vehicle state to AT_STOP
        self.state_manager.vehicle_worker.update_vehicle_status(
            vehicle_id,
            VehicleStatus.AT_STOP,
            self.context.current_time
        )
        
        # Update additional vehicle state
        vehicle = self.state_manager.vehicle_worker.get_vehicle(vehicle_id)
        if vehicle:
            vehicle.current_state.current_stop_id = current_stop.stop.id
            vehicle.current_state.waiting_for_passengers = True
            self.state_manager.vehicle_worker.update_vehicle(vehicle)
        
        # Mark the segment as completed when the vehicle arrives at the stop
        # This prevents matching algorithms from modifying segments that have already been traversed
        current_segment.completed = True
        route.recalc_current_segment_index()
        
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
        
        # Update route in state - this is important to persist the segment completion status
        self.state_manager.route_worker.update_route(route)

    def handle_stop_operations_completed(self, event: Event) -> None:
        """Handle completion of all operations at a stop."""
        try:
            logger.debug(f"=== START STOP OPERATIONS COMPLETION HANDLING ===")
            logger.debug(f"Vehicle {event.vehicle_id} - Processing stop operations completion at time: {self.context.current_time}")
            self.state_manager.begin_transaction()
            
            vehicle_id = event.vehicle_id
            route_id = event.data.get('route_id')
            segment_id = event.data.get('segment_id')
            
            vehicle = self.state_manager.vehicle_worker.get_vehicle(vehicle_id)
            if not vehicle:
                raise ValueError(f"Vehicle {vehicle_id} not found")
                
            route = self.state_manager.route_worker.get_route(route_id)
            if not route:
                raise ValueError(f"Route {route_id} not found")

            # Find the completed segment
            current_segment = None
            for segment in route.segments:
                if segment.id == segment_id:
                    current_segment = segment
                    break

            if not current_segment:
                # Instead of raising an error, log a warning and gracefully handle the stale event
                logger.warning(f"Segment {segment_id} not found in route {route_id} - This may be a stale event due to rerouting")
                
                # Check if the vehicle is already at a different segment
                if route.get_current_segment():
                    logger.info(f"Vehicle {vehicle_id} is already at segment {route.get_current_segment().id} - Ignoring stale event")
                    self.state_manager.rollback_transaction()
                    return
                
                # If we can't find the current segment, try to recalculate it
                route.recalc_current_segment_index()
                current_segment = route.get_current_segment()
                
                # If we still can't find a valid segment, we need to handle this case
                if not current_segment:
                    logger.warning(f"Cannot find a valid current segment for route {route_id} - Handling as route completion")
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
                    
                    # Update route in state
                    self.state_manager.route_worker.update_route(route)
                    self.state_manager.commit_transaction()
                    return
                
                # If we found a valid segment, continue with that one
                logger.info(f"Using recalculated current segment {current_segment.id} instead of stale segment {segment_id}")

            current_stop = current_segment.destination
                
            # Update segment completion data
            self._update_segment_completion(route, current_segment, current_stop, event)

            # Check if route is completed
            if route.is_completed():
                logger.debug(f"[Stop Operations Completion] Vehicle {vehicle_id} - Route is completed, handling route completion")
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
                logger.debug(f"[Stop Operations Completion] Vehicle {vehicle_id} - Route not completed, handling route continuation")
                route = self._handle_route_continuation(vehicle_id, route, self.context.current_time)
            
            # Update route in state
            self.state_manager.route_worker.update_route(route)
            
            self.state_manager.commit_transaction()
            
        except Exception as e:
            self.state_manager.rollback_transaction()
            logger.error(f"Error handling stop operations completion for vehicle {event.vehicle_id} on route {route_id} for segment {segment_id}: {traceback.format_exc()}")
            self._handle_vehicle_error(event, str(e))

    def _update_segment_completion(
        self, 
        route: Route,
        segment: RouteSegment, 
        stop: RouteStop, 
        event: Event
    ) -> None:
        """Update segment completion metrics and timing."""
        stop.actual_arrival_time = self.context.current_time
        # segment.completed is now set in _handle_regular_stop_arrival
        # We just update the metrics here
        segment.actual_duration = (
            self.context.current_time - event.data['movement_start_time']
        ).total_seconds()
        segment.actual_distance = event.data.get('actual_distance', 
                                            segment.estimated_distance)
        route.recalc_current_segment_index()

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
        return route

    def handle_vehicle_wait_timeout(self, event: Event) -> None:
        """Delegate wait timeout handling to StopCoordinator."""
        try:
            # Simply delegate to stop coordinator
            self.stop_coordinator.handle_wait_timeout(event)
        except Exception as e:
            logger.error(f"Error handling wait timeout: {traceback.format_exc()}")
            self._handle_vehicle_error(event, str(e))

    def _handle_route_continuation(self, vehicle_id: str, route: 'Route', current_time: datetime) -> None:
        """Handle continuation to next segment in route."""
        logger.debug(f"=== START ROUTE CONTINUATION ===")
        logger.debug(f"Vehicle {vehicle_id} - Processing route continuation at time: {current_time}")
        logger.debug(f"Route details before continuation:")
        logger.debug(f"  Current segment index: {route.current_segment_index}")
        logger.debug(f"  Total segments: {len(route.segments)}")
        logger.debug(f"  Current segment: {str(route.get_current_segment()) if route.get_current_segment() else 'None'}")
        
        # Validate route state before continuation
        if not route.segments:
            logger.error(f"[Route Continuation] Vehicle {vehicle_id} - Route has no segments")
            raise ValueError("Route has no segments")
        
        # Get current segment and validate its state
        current_segment = route.get_current_segment()
        if not current_segment:
            logger.error(f"[Route Continuation] Vehicle {vehicle_id} - No current segment found at index {route.current_segment_index}")
            raise ValueError("No current segment found")
        # Create movement event for next segment
        self._create_vehicle_movement_event(
            vehicle_id=vehicle_id,
            segment=current_segment,
            current_time=current_time
        )
        return route

    def _create_vehicle_movement_event(
        self,
        vehicle_id: str,
        segment: RouteSegment,
        current_time: datetime
    ) -> None:
        """Create events for vehicle movement along a route segment with waypoint updates."""
        logger.debug(f"=== START MOVEMENT EVENT CREATION ===")
        logger.debug(f"Vehicle {vehicle_id} - Creating movement events at time: {current_time}")
        logger.debug(f"Segment details: {str(segment)}")
        logger.debug(f"Movement timing:")
        logger.debug(f"  Start time: {current_time}")
        logger.debug(f"  Estimated duration: {segment.estimated_duration}")
        logger.debug(f"  Expected arrival: {current_time + timedelta(seconds=segment.estimated_duration)}")
        
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
        
        # If we have waypoints, create intermediate position update events (max 4)
        if waypoints:
            # Limit the number of position updates to a maximum of 4
            max_updates = min(4, len(waypoints))
            
            # Select evenly spaced waypoints if we have more than max_updates
            if len(waypoints) > max_updates:
                step = len(waypoints) / max_updates
                selected_indices = [int(i * step) for i in range(max_updates)]
                selected_waypoints = [waypoints[i] for i in selected_indices]
            else:
                selected_waypoints = waypoints
            
            # Calculate time intervals for waypoint updates
            time_between_updates = total_duration / (max_updates + 1)
            distance_between_updates = total_distance / (max_updates + 1)
            
            current_distance = 0
            for i, waypoint in enumerate(selected_waypoints):
                update_time = current_time + timedelta(seconds=(i+1) * time_between_updates)
                current_distance += distance_between_updates
                
                # Create position update event
                update_event = Event(
                    event_type=EventType.VEHICLE_POSITION_UPDATE,
                    priority=EventPriority.HIGH,
                    timestamp=update_time,
                    vehicle_id=vehicle_id,
                    data={
                        'segment_id': segment.id,
                        'location': waypoint,
                        'progress_percentage': ((i+1) / (max_updates + 1)) * 100,
                        'distance_covered': current_distance,
                        'movement_start_time': current_time
                    }
                )
                self.context.event_manager.publish_event(update_event)
        
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
                'actual_distance': total_distance
            }
        )
        self.context.event_manager.publish_event(arrival_event)

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