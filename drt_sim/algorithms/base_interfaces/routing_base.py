'''Under Construction'''
from abc import ABC, abstractmethod
from typing import List, Tuple, Dict, Any, Optional
from datetime import datetime
from drt_sim.models.location import Location
class RoutingStrategy(ABC):
    """Base interface for routing algorithms"""
    
    @abstractmethod
    def compute_route(
        self,
        start: Location,
        end: Location,
        departure_time: datetime,
        intermediate_stops: Optional[List[Location]] = None,
        constraints: Optional[Dict[str, Any]] = None
    ) -> Tuple[List[Location], float]:
        """
        Compute route between locations.
        
        Returns:
            Tuple of (route_points, estimated_duration)
        """
        pass
    
    @abstractmethod
    def estimate_travel_time(
        self,
        start: Location,
        end: Location,
        departure_time: datetime,
        constraints: Optional[Dict[str, Any]] = None
    ) -> float:
        """Estimate travel time between two points"""
        pass
    
    @abstractmethod
    def update_traffic_conditions(
        self,
        conditions: Dict[str, Any],
        timestamp: datetime
    ) -> None:
        """Update current traffic conditions"""
        pass