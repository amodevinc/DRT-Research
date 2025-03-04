from typing import Dict, Any, Optional, List
from datetime import datetime
from collections import defaultdict
import traceback
from drt_sim.core.state.base import StateWorker, StateContainer
from drt_sim.models.route import Route, RouteStatus, RouteSegment
from drt_sim.models.state import RouteSystemState
import logging
logger = logging.getLogger(__name__)

class RouteStateWorker(StateWorker):
    """Manages state for vehicle routes"""
    
    def __init__(self):
        self.routes = StateContainer[Route]()
        self.initialized = False
        
        # Indexes for quick lookups
        self.vehicle_to_route: Dict[str, str] = {}
        self.active_routes: set[str] = set()
    
        # Additional properties needed for RouteSystemState
        self.passenger_route_mapping: Dict[str, str] = {}  # passenger_id -> route_id
    
    def initialize(self, config: Optional[Any] = None) -> None:
        """Initialize route state worker"""
        self.initialized = True
        logger.info("Route state worker initialized")

    def get_active_routes(self) -> List[Route]:
        """Get all active routes"""
        return [route for route in self.routes.items.values() if route.status == RouteStatus.ACTIVE]
    
    def add_route(self, route: Route) -> None:
        """Add a new route to state management."""
        if not self.initialized:
            raise RuntimeError("Worker not initialized")
            
        try:
            if route.id in self.routes.items:
                raise ValueError(f"Route {route.id} already exists")
            
            # Add to main container
            self.routes.add(route.id, route)
            
            # Update indexes
            self.vehicle_to_route[route.vehicle_id] = route.id
            if route.status == RouteStatus.ACTIVE:
                self.active_routes.add(route.id)
            
            logger.debug(f"Added route {route.id} for vehicle {route.vehicle_id}")
            
        except Exception as e:
            logger.error(f"Failed to add route: {str(e)}")
            raise

    def update_segment(self, route_id: str, segment: RouteSegment) -> None:
        """Update a segment in the route"""
        route = self.get_route(route_id)
        if route:
            segment_index = route.segments.index(segment)
            if segment_index != -1:
                route.segments[segment_index] = segment
                self.routes.update(route_id, route)
    
    def get_route(self, route_id: str) -> Optional[Route]:
        """Get route by ID"""
        return self.routes.get(route_id)
    
    def update_route(self, route: Route) -> None:
        """Update existing route state"""
        if not self.initialized:
            raise RuntimeError("Worker not initialized")
            
        try:
            old_route = self.routes.get(route.id)
            if not old_route:
                raise ValueError(f"Route {route.id} not found")
            
            # Log route state changes
            if old_route.current_segment_index != route.current_segment_index:
                logger.info(f"[Route State] Route {route.id} - Segment index changed: {old_route.current_segment_index} -> {route.current_segment_index}")
                
            if old_route.status != route.status:
                logger.info(f"[Route State] Route {route.id} - Status changed: {old_route.status} -> {route.status}")
            
            completed_segments = len([s for s in route.segments if s.completed])
            old_completed = len([s for s in old_route.segments if s.completed])
            if completed_segments != old_completed:
                logger.info(f"[Route State] Route {route.id} - Completed segments changed: {old_completed} -> {completed_segments}")
            
            # Update route
            self.routes.update(route.id, route)
            
        except Exception as e:
            logger.error(f"Failed to update route: {traceback.format_exc()}")
            raise
    
    def update_route_status(
        self,
        route_id: str,
        status: RouteStatus,
        additional_data: Optional[Dict[str, Any]] = None
    ) -> None:
        """Update route status and related data"""
        if not self.initialized:
            raise RuntimeError("Worker not initialized")
            
        try:
            route = self.routes.get(route_id)
            if not route:
                raise ValueError(f"Route {route_id} not found")
            
            old_status = route.status
            route.status = status
            # Update indexes based on status change
            if status == RouteStatus.ACTIVE:
                self.active_routes.add(route_id)
            elif old_status == RouteStatus.ACTIVE:
                self.active_routes.remove(route_id)

            if status == RouteStatus.COMPLETED:
                logger.info(f"Route {route_id} completed")
            # Update additional data if provided
            if additional_data:
                for key, value in additional_data.items():
                    if hasattr(route, key):
                        setattr(route, key, value)
            
            logger.debug(
                f"Updated route {route_id} status from {old_status} to {status}"
            )
            
        except Exception as e:
            logger.error(f"Failed to update route status: {str(e)}")
            raise


    def get_routes_by_status(self, status: RouteStatus) -> List[Route]:
        """Get all routes with specified status"""
        return [
            route for route in self.routes.items.values()
            if route.status == status
        ]

    def get_vehicle_route_history(self, vehicle_id: str) -> List[Route]:
        """Get historical routes for a vehicle"""
        return [
            route for route in self.routes.items.values()
            if route.vehicle_id == vehicle_id
        ]

    def take_snapshot(self, timestamp: datetime) -> None:
        """Take snapshot of route states"""
        if not self.initialized:
            raise RuntimeError("Worker not initialized")
        self.routes.take_snapshot(timestamp)

    @property
    def active_passenger_count(self) -> int:
        """Get count of passengers currently in vehicles"""
        count = 0
        for route in self.routes.items.values():
            if route.status == RouteStatus.ACTIVE:
                count += len(route.current_passengers)
        return count

    @property
    def total_remaining_stops(self) -> int:
        """Get total number of remaining stops across all active routes"""
        count = 0
        for route in self.routes.items.values():
            if route.status == RouteStatus.ACTIVE:
                count += len(route.segments[route.current_segment_index:])
        return count

    @property
    def routes_by_vehicle(self) -> Dict[str, str]:
        """Get mapping of vehicle_id to route_id"""
        return self.vehicle_to_route.copy()

    def get_state(self) -> RouteSystemState:
        """Get current state of the route system"""
        if not self.initialized:
            raise RuntimeError("Worker not initialized")
            
        try:
            # Get routes by status
            routes_by_status = defaultdict(list)
            for rid, route in self.routes.items.items():
                routes_by_status[route.status].append(rid)
            
            return RouteSystemState(
                active_routes={
                    rid: route for rid, route in self.routes.items.items()
                    if route.status == RouteStatus.ACTIVE
                },
                routes_by_status=dict(routes_by_status),
                routes_by_vehicle=self.vehicle_to_route,
                passenger_route_mapping=self.passenger_route_mapping,
            )
            
        except Exception as e:
            logger.error(f"Failed to get route system state: {str(e)}")
            raise

    def update_state(self, state: RouteSystemState) -> None:
        """Update route system state"""
        if not self.initialized:
            raise RuntimeError("Worker not initialized")
            
        try:
            # Update active routes
            for route_id, route in state.active_routes.items():
                if route_id in self.routes.items:
                    self.routes.update(route_id, route)
                else:
                    self.routes.add(route_id, route)
            
            # Update mappings
            self.vehicle_to_route = state.routes_by_vehicle
            self.passenger_route_mapping = state.passenger_route_mapping
            
            
            # Update active routes set
            self.active_routes = set(state.active_routes.keys())
            
            logger.info("Route system state updated successfully")
            
        except Exception as e:
            logger.error(f"Failed to update route system state: {str(e)}")
            raise

    def restore_state(self, saved_state: Dict[str, Any]) -> None:
        """Restore route states from a saved state"""
        if not self.initialized:
            raise RuntimeError("Worker not initialized")
        
        try:
            # Begin transaction
            self.begin_transaction()
            
            try:
                # Reset current state
                self.routes = StateContainer[Route]()
                self.vehicle_to_route.clear()
                self.active_routes.clear()
                
                # Restore state using update_state
                self.update_state(saved_state)
                
                self.commit_transaction()
                logger.debug("Restored route system state")
                
            except Exception as e:
                self.rollback_transaction()
                raise
                
        except Exception as e:
            logger.error(f"Error restoring route system state: {str(e)}")
            raise

    def begin_transaction(self) -> None:
        """Begin state transaction"""
        self.routes.begin_transaction()
    
    def commit_transaction(self) -> None:
        """Commit current transaction"""
        self.routes.commit_transaction()
    
    def rollback_transaction(self) -> None:
        """Rollback current transaction"""
        self.routes.rollback_transaction()

    def cleanup(self) -> None:
        """Clean up resources"""
        self.routes.clear_history()
        self.vehicle_to_route.clear()
        self.active_routes.clear()
        self.initialized = False
        logger.info("Route state worker cleaned up")
