from typing import List, Dict
from datetime import datetime, timedelta
from drt_sim.models.vehicle import Vehicle, VehicleStatus
from drt_sim.models.request import Request
from drt_sim.models.location import Location
from ..base_interfaces.dispatch_base import DispatchStrategy, DispatchResult, DemandPrediction

class FCFSDispatch(DispatchStrategy):
    """First-Come-First-Served dispatch strategy"""
    
    def __init__(self, vehicle_speed_kmh: float = 30.0):
        self.vehicle_speed_kmh = vehicle_speed_kmh
        
    def calculate_travel_time(self, loc1: Location, loc2: Location) -> timedelta:
        """Calculate travel time between two locations"""
        distance = loc1.distance_to(loc2)
        hours = distance / self.vehicle_speed_kmh
        return timedelta(hours=hours)
        
    def dispatch(
        self,
        requests: List[Request],
        vehicles: List[Vehicle],
        current_time: datetime
    ) -> DispatchResult:
        assignments: Dict[str, List[str]] = {}
        unassigned: List[str] = []
        pickup_times: Dict[str, datetime] = {}
        dropoff_times: Dict[str, datetime] = {}
        
        # Sort requests by request time
        sorted_requests = sorted(requests, key=lambda r: r.request_time)
        
        for request in sorted_requests:
            # Find nearest available vehicle
            best_vehicle = None
            min_time = timedelta(hours=24)  # Large initial value
            
            for vehicle in vehicles:
                if vehicle.current_state.status == VehicleStatus.IDLE:
                    travel_time = self.calculate_travel_time(
                        vehicle.current_state.current_location,
                        request.pickup_location
                    )
                    if travel_time < min_time:
                        min_time = travel_time
                        best_vehicle = vehicle
            
            if best_vehicle:
                # Assign vehicle
                vehicle_id = best_vehicle.current_state.id
                if vehicle_id not in assignments:
                    assignments[vehicle_id] = []
                assignments[vehicle_id].append(request.id)
                
                # Calculate times
                pickup_time = current_time + min_time
                dropoff_time = pickup_time + self.calculate_travel_time(
                    request.pickup_location,
                    request.dropoff_location
                )
                
                pickup_times[request.id] = pickup_time
                dropoff_times[request.id] = dropoff_time
            else:
                unassigned.append(request.id)
        
        return DispatchResult(
            vehicle_assignments=assignments,
            unassigned_requests=unassigned,
            estimated_pickup_times=pickup_times,
            estimated_dropoff_times=dropoff_times
        )
    
    def update_assignments(
        self,
        new_requests: List[Request],
        current_assignments: Dict[str, List[str]],
        vehicles: List[Vehicle],
        current_time: datetime
    ) -> DispatchResult:
        # For FCFS, just dispatch new requests
        return self.dispatch(new_requests, vehicles, current_time)
    
    def rebalance_vehicles(
        self,
        idle_vehicles: List[Vehicle],
        demand_predictions: Dict[Location, DemandPrediction],
        current_time: datetime
    ) -> Dict[str, Location]:
        # Simple rebalancing - distribute vehicles evenly among predicted demand
        if not demand_predictions or not idle_vehicles:
            return {}
            
        rebalance_targets: Dict[str, Location] = {}
        sorted_predictions = sorted(
            demand_predictions.items(),
            key=lambda x: x[1]['predicted_demand'],
            reverse=True
        )
        
        # Assign vehicles to highest demand areas
        for i, vehicle in enumerate(idle_vehicles):
            pred_idx = i % len(sorted_predictions)
            target_location = sorted_predictions[pred_idx][0]
            rebalance_targets[vehicle.id] = target_location
            
        return rebalance_targets