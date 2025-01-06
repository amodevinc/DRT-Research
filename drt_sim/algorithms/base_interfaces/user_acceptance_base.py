from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
from datetime import datetime, timedelta
from drt_sim.models.request import Request

class UserAcceptanceModel(ABC):
    """Base interface for user acceptance models"""
    
    @abstractmethod
    def calculate_acceptance_probability(
        self,
        request: Request,
        proposed_pickup_time: datetime,
        proposed_travel_time: timedelta,
        cost: float,
        service_attributes: Optional[Dict[str, Any]] = None
    ) -> float:
        """Calculate probability of user accepting a proposed service"""
        pass
    
    @abstractmethod
    def update_model(
        self,
        request: Request,
        accepted: bool,
        service_attributes: Dict[str, Any]
    ) -> None:
        """Update model based on user decisions"""
        pass