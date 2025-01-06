from abc import ABC, abstractmethod
from typing import List
from datetime import datetime
from drt_sim.models.location import Location
from drt_sim.models.request import Request
from drt_sim.models.vehicle import Vehicle

class CostCalculator(ABC):
    """Base interface for cost calculation components"""
    
    @abstractmethod
    def calculate_operational_cost(
        self,
        route: List[Location],
        vehicle: Vehicle,
        requests: List[Request]
    ) -> float:
        """Calculate operational cost for a route"""
        ...
    
    def calculate_user_cost(
        self,
        request: Request,
        pickup_time: datetime,
        dropoff_time: datetime
    ) -> float:
        """Calculate cost from user perspective"""
        ...
