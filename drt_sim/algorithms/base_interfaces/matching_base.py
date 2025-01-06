from abc import ABC, abstractmethod
from typing import List, Tuple, Dict, Any, Optional
from datetime import datetime
from drt_sim.models.request import Request
from drt_sim.models.vehicle import Vehicle

class MatchingStrategy(ABC):
    """Base interface for request-vehicle matching algorithms"""
    
    @abstractmethod
    def compute_matches(
        self,
        requests: List[Request],
        vehicles: List[Vehicle],
        current_time: datetime,
        constraints: Optional[Dict[str, Any]] = None
    ) -> Dict[str, List[Tuple[str, float]]]:
        """
        Compute optimal matches between requests and vehicles.
        
        Returns:
            Dict mapping vehicle IDs to lists of (request_id, score) tuples
        """
        pass
    
    @abstractmethod
    def evaluate_match(
        self,
        request: Request,
        vehicle: Vehicle,
        current_time: datetime
    ) -> float:
        """Evaluate quality score for a specific request-vehicle match"""
        pass