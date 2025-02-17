'''Under Construction'''
from typing import List, Dict, Any, Optional
from datetime import datetime
from dataclasses import dataclass
from drt_sim.models.request import Request
from drt_sim.models.vehicle import Vehicle, VehicleStatus
from drt_sim.models.state import SimulationState
from drt_sim.algorithms.base_interfaces.matching_base import Assignment
from drt_sim.network.manager import NetworkManager
from drt_sim.core.state.manager import StateManager
from drt_sim.config.config import MatchingAssignmentConfig
@dataclass
class Bid:
    """Represents a vehicle's bid for a request."""
    vehicle_id: str
    request_id: str
    pickup_time: datetime
    dropoff_time: datetime
    cost: float
    waiting_time: float
    detour_time: float
    extra_distance: float

class AuctionAssigner:
    """Assigns requests to vehicles using an auction-based mechanism."""
    
    def __init__(
        self,
        config: MatchingAssignmentConfig,
        network_manager: NetworkManager,
        state_manager: StateManager
    ):
        self.config = config
        self.network_manager = network_manager
        self.state_manager = state_manager
        
        # Configuration parameters
        self.max_waiting_time = config.max_waiting_time
        self.max_detour_ratio = config.max_detour_ratio
        self.cost_weights = config.weights
        self.reserve_price = config.reserve_price if config.reserve_price else float('inf')

    def assign_requests(
        self,
        clustered_requests: Dict[str, List[Request]],
        available_vehicles: List[Vehicle],
        system_state: SimulationState
    ) -> List[Assignment]:
        """Main method to assign requests using auction mechanism."""
        assignments = []
        
        for cluster_id, requests in clustered_requests.items():
            # Sort requests by urgency
            sorted_requests = sorted(
                requests,
                key=lambda r: r.pickup_time or r.request_time
            )
            
            # Conduct sequential auctions for each request
            for request in sorted_requests:
                # Collect bids from eligible vehicles
                bids = self._collect_bids(request, available_vehicles, system_state)
                
                # Select winning bid
                winning_bid = self._select_winning_bid(bids)
                
                if winning_bid:
                    # Create assignment from winning bid
                    assignment = self._create_assignment(request, winning_bid)
                    assignments.append(assignment)
                    
                    # Update vehicle state
                    self._update_vehicle_state(
                        winning_bid.vehicle_id,
                        available_vehicles,
                        request
                    )
        
        return assignments

    def _collect_bids(
        self,
        request: Request,
        vehicles: List[Vehicle],
        system_state: SimulationState
    ) -> List[Bid]:
        """Collect bids from all eligible vehicles."""
        bids = []
        
        for vehicle in vehicles:
            if not self._is_vehicle_eligible(vehicle, request):
                continue
                
            bid = self._calculate_bid(request, vehicle, system_state)
            if bid:
                bids.append(bid)
        
        return bids

    def _calculate_bid(
        self,
        request: Request,
        vehicle: Vehicle,
        system_state: SimulationState
    ) -> Optional[Bid]:
        """Calculate a vehicle's bid for a request."""
        # Calculate basic metrics
        pickup_time, dropoff_time = self._calculate_service_times(
            request, vehicle, system_state.current_time
        )
        
        if not pickup_time or not dropoff_time:
            return None
            
        waiting_time = (pickup_time - request.request_time).total_seconds()
        direct_time = self._calculate_direct_time(request)
        actual_time = (dropoff_time - pickup_time).total_seconds()
        detour_time = actual_time - direct_time
        extra_distance = self._calculate_extra_distance(request, vehicle)
        
        # Check constraints
        if not self._is_bid_feasible(waiting_time, detour_time, direct_time):
            return None
        
        # Calculate cost
        cost = self._calculate_cost(waiting_time, detour_time, extra_distance)
        
        return Bid(
            vehicle_id=vehicle.id,
            request_id=request.id,
            pickup_time=pickup_time,
            dropoff_time=dropoff_time,
            cost=cost,
            waiting_time=waiting_time,
            detour_time=detour_time,
            extra_distance=extra_distance
        )

    def _is_vehicle_eligible(self, vehicle: Vehicle, request: Request) -> bool:
        """Check if vehicle is eligible to bid."""
        return (
            vehicle.current_state.status in [VehicleStatus.IDLE, VehicleStatus.IN_SERVICE] and
            vehicle.current_state.current_occupancy < vehicle.capacity and
            (not request.constraints or
             not request.constraints.required_vehicle_type or
             vehicle.type == request.constraints.required_vehicle_type)
        )

    def _is_bid_feasible(
        self,
        waiting_time: float,
        detour_time: float,
        direct_time: float
    ) -> bool:
        """Check if bid meets constraints."""
        return (
            waiting_time <= self.max_waiting_time and
            (direct_time == 0 or
             (direct_time + detour_time) / direct_time <= self.max_detour_ratio)
        )

    def _calculate_cost(
        self,
        waiting_time: float,
        detour_time: float,
        extra_distance: float
    ) -> float:
        """Calculate bid cost using weighted sum of metrics."""
        normalized_waiting = waiting_time / self.max_waiting_time
        normalized_detour = detour_time / (self.max_detour_ratio * 3600)
        normalized_distance = extra_distance / 5000  # Assuming 5km reference
        
        cost = (
            self.cost_weights['waiting_time'] * normalized_waiting +
            self.cost_weights['detour_time'] * normalized_detour +
            self.cost_weights['distance'] * normalized_distance
        )
        
        return cost

    def _select_winning_bid(self, bids: List[Bid]) -> Optional[Bid]:
        """Select winning bid with lowest cost."""
        if not bids:
            return None
            
        winning_bid = min(bids, key=lambda b: b.cost)
        return winning_bid if winning_bid.cost <= self.reserve_price else None

    # ... (Additional helper methods similar to InsertionAssigner) ...
