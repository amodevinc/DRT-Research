from abc import ABC, abstractmethod
from typing import List, Dict, TypedDict
from datetime import datetime
from dataclasses import dataclass

from drt_sim.models.request import Request
from drt_sim.models.vehicle import Vehicle
from drt_sim.models.location import Location

@dataclass
class DispatchResult:
    """Structured result from dispatch operations"""
    vehicle_assignments: Dict[str, List[str]]  # vehicle_id -> list of request_ids
    unassigned_requests: List[str]  # list of unassigned request_ids
    estimated_pickup_times: Dict[str, datetime]  # request_id -> estimated pickup time
    estimated_dropoff_times: Dict[str, datetime]  # request_id -> estimated dropoff time

class DemandPrediction(TypedDict):
    """Type for demand predictions"""
    location: Location
    predicted_demand: float
    confidence: float
    time_window: datetime

class DispatchStrategy(ABC):
    """Base interface for dispatch algorithms"""
    
    @abstractmethod
    def dispatch(
        self,
        requests: List[Request],
        vehicles: List[Vehicle],
        current_time: datetime
    ) -> DispatchResult:
        """
        Assign vehicles to requests.
        
        Args:
            requests: List of requests to be assigned
            vehicles: List of available vehicles
            current_time: Current simulation time
            
        Returns:
            DispatchResult containing assignments and timing estimates
        
        Raises:
            ValueError: If input parameters are invalid
        """
        pass
    
    @abstractmethod
    def update_assignments(
        self,
        new_requests: List[Request],
        current_assignments: Dict[str, List[str]],
        vehicles: List[Vehicle],
        current_time: datetime
    ) -> DispatchResult:
        """
        Update assignments with new requests
        
        Args:
            new_requests: New requests to be incorporated
            current_assignments: Existing vehicle-request assignments
            vehicles: Available vehicles
            current_time: Current simulation time
            
        Returns:
            Updated DispatchResult
            
        Raises:
            ValueError: If current assignments are invalid
        """
        pass
    
    @abstractmethod
    def rebalance_vehicles(
        self,
        idle_vehicles: List[Vehicle],
        demand_predictions: Dict[Location, DemandPrediction],
        current_time: datetime
    ) -> Dict[str, Location]:
        """
        Rebalance idle vehicles based on predicted demand.
        
        Args:
            idle_vehicles: List of vehicles available for rebalancing
            demand_predictions: Predicted demand at different locations
            current_time: Current simulation time
            
        Returns:
            Dict mapping vehicle IDs to target locations
            
        Raises:
            ValueError: If predictions are invalid or vehicles are not idle
        """
        pass
    
    def validate_assignments(
        self,
        assignments: Dict[str, List[str]],
        vehicles: List[Vehicle],
        requests: List[Request]
    ) -> bool:
        """
        Validate that assignments are feasible
        
        Args:
            assignments: Vehicle-request assignments to validate
            vehicles: Available vehicles
            requests: Requests being assigned
            
        Returns:
            True if assignments are valid, False otherwise
        """
        vehicle_ids = {v.id for v in vehicles}
        request_ids = {r.id for r in requests}
        
        # Check that all vehicle IDs exist
        if not all(v_id in vehicle_ids for v_id in assignments.keys()):
            return False
            
        # Check that all request IDs exist
        assigned_requests = {r_id for r_ids in assignments.values() for r_id in r_ids}
        if not all(r_id in request_ids for r_id in assigned_requests):
            return False
            
        # Check that no request is assigned multiple times
        if len(assigned_requests) != sum(len(r_ids) for r_ids in assignments.values()):
            return False
            
        return True