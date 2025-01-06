from abc import ABC, abstractmethod
from typing import List, Tuple, Dict, Any, Optional
from datetime import datetime
from drt_sim.models.location import Location

class DemandPredictor(ABC):
    """Base interface for demand prediction"""
    
    @abstractmethod
    def predict_demand(
        self,
        region: List[Location],
        time_period: Tuple[datetime, datetime],
        historical_data: Optional[Dict[str, Any]] = None,
        external_factors: Optional[Dict[str, Any]] = None
    ) -> Dict[Location, List[float]]:
        """
        Predict demand for a region over time period.
        
        Returns:
            Dict mapping locations to lists of demand values over time
        """
        pass
    
    @abstractmethod
    def update_model(
        self,
        actual_demand: Dict[Location, float],
        timestamp: datetime
    ) -> None:
        """Update prediction model with actual demand data"""
        pass