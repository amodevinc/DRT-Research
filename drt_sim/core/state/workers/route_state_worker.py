from typing import Dict, Any, Optional
from datetime import datetime
from dataclasses import dataclass
import logging

from drt_sim.core.state.base import StateWorker, StateContainer
from drt_sim.models.route import Route

logger = logging.getLogger(__name__)

@dataclass
class RouteMetrics:
    """Metrics tracked for routes"""
    total_routes: int = 0
    active_routes: int = 0
    completed_routes: int = 0
    average_route_length: float = 0.0
    average_stops_per_route: float = 0.0

class RouteStateWorker(StateWorker):
    """Manages state for vehicle routes"""
    
    def __init__(self):
        self.routes = StateContainer[Route]()
        self.metrics = RouteMetrics()
        self.initialized = False
    
    def initialize(self, config: Optional[Any] = None) -> None:
        """Initialize route state worker"""
        self.initialized = True
        logger.info("Route state worker initialized")
    
    def take_snapshot(self, timestamp: datetime) -> None:
        """Take snapshot of route states"""
        if not self.initialized:
            raise RuntimeError("Worker not initialized")
        self.routes.take_snapshot(timestamp)
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get current route metrics"""
        return {
            "route_total": self.metrics.total_routes,
            "route_active": self.metrics.active_routes,
            "route_completed": self.metrics.completed_routes,
            "route_avg_length": self.metrics.average_route_length,
            "route_avg_stops": self.metrics.average_stops_per_route
        }
    
    def get_state(self) -> Dict[str, Any]:
        """Get current route states"""
        return {
            route_id: route.to_dict() 
            for route_id, route in self.routes.items.items()
        }
    
    def update_state(self, state: Dict[str, Any]) -> None:
        """Update route states"""
        if not self.initialized:
            raise RuntimeError("Worker not initialized")
            
        for route_id, route_data in state.items():
            if route_id in self.routes.items:
                self.routes.update(route_id, route_data)
            else:
                route = Route.from_dict(route_data)
                self.routes.add(route_id, route)
    
    def restore_state(self, saved_state: Dict[str, Any]) -> None:
        """Restore route states from saved state"""
        self.routes = StateContainer[Route]()
        for route_id, route_data in saved_state.items():
            route = Route.from_dict(route_data)
            self.routes.add(route_id, route)
    
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
        self.initialized = False