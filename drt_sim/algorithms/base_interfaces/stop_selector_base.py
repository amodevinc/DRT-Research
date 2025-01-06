from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
from drt_sim.models.location import Location

class StopSelector(ABC):
    """Base interface for stop selection algorithms"""
    
    @abstractmethod
    def select_stops(
        self,
        region: List[Location],
        demand_patterns: Dict[Location, float],
        existing_stops: Optional[List[Location]] = None,
        constraints: Optional[Dict[str, Any]] = None
    ) -> List[Location]:
        """Select optimal stop locations"""
        pass
    
    @abstractmethod
    def score_stop(
        self,
        stop: Location,
        demand_patterns: Dict[Location, float],
        existing_stops: List[Location]
    ) -> float:
        """Score a potential stop location"""
        pass