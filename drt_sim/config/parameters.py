# drt_sim/config/parameters.py
from dataclasses import dataclass
from typing import Optional, List, Dict
from datetime import datetime, timedelta
from enum import Enum
import pydantic
from pathlib import Path

class AlgorithmType(Enum):
    # Routing algorithms
    ROUTING_DIJKSTRA = "dijkstra"
    ROUTING_ASTAR = "astar"
    ROUTING_OSRM = "osrm"
    
    # Vehicle dispatch algorithms
    DISPATCH_IMMEDIATE = "immediate_dispatch"  # Assign vehicles as requests arrive
    DISPATCH_BATCH = "batch_dispatch"  # Group requests into batches
    DISPATCH_ANTICIPATORY = "anticipatory_dispatch"  # Consider future demand
    
    # Request-vehicle matching algorithms
    MATCHING_INSERTION = "insertion_heuristic"  # Insert requests into existing routes
    MATCHING_AUCTION = "auction_based"  # Vehicles bid on requests
    MATCHING_GENETIC = "genetic_algorithm"  # Optimize matching using GA
    
    # Stop selection algorithms
    STOP_KMEANS = "kmeans_clustering"  # Cluster-based stop placement
    STOP_DEMAND = "demand_based"  # Place stops based on demand patterns
    STOP_COVERAGE = "coverage_based"  # Maximize area coverage
    STOP_ACCESSIBILITY = "accessibility_based"  # Optimize for accessibility
    STOP_MULTI_OBJECTIVE_COST_MINIMIZATION = "multi_objective_cost_minimization"  # Minimize cost and maximize coverage

class VehicleMatchingPolicy(Enum):
    NEAREST = "nearest"
    LEAST_DETOUR = "least_detour"
    BALANCED_LOAD = "balanced_load"
    MINIMUM_COST = "minimum_cost"

class BaseParameters(pydantic.BaseModel):
    """Base class for all parameter types with validation"""
    class Config:
        arbitrary_types_allowed = True
        validate_assignment = True

@dataclass
class SimulationParameters(BaseParameters):
    """Core simulation parameters"""
    start_time: datetime
    end_time: datetime
    time_step: timedelta
    random_seed: Optional[int] = None
    warm_up_period: timedelta = timedelta(hours=1)
    cool_down_period: timedelta = timedelta(hours=1)
    snapshot_interval: timedelta = timedelta(minutes=5)
@dataclass
class VehicleParameters(BaseParameters):
    """Vehicle-related parameters"""
    fleet_size: int
    vehicle_capacity: int
    vehicle_speed: float  # km/h
    boarding_time: timedelta
    alighting_time: timedelta
    battery_capacity: Optional[float] = None  # kWh
    charging_rate: Optional[float] = None  # kW
    depot_locations: List[tuple] = None

@dataclass
class DemandParameters(BaseParameters):
    """Demand generation parameters"""
    demand_level: float  # requests per hour
    spatial_distribution: str
    temporal_distribution: str
    max_wait_time: timedelta
    max_detour_ratio: float
    cancellation_probability: float = 0.1
    prebooking_ratio: float = 0.2
    demand_patterns: Dict[str, float] = None

@dataclass
class StopParameters(BaseParameters):
    """Stop selection and management parameters"""
    max_walking_distance: float  # meters
    min_stop_spacing: float  # meters
    stop_capacity: int
    safety_threshold: float
    accessibility_weight: float
    coverage_radius: float  # meters

@dataclass
class AlgorithmParameters(BaseParameters):
    """Algorithm-specific parameters"""
    routing_algorithm: AlgorithmType
    dispatch_algorithm: AlgorithmType
    matching_algorithm: AlgorithmType
    matching_policy: VehicleMatchingPolicy
    reoptimization_interval: timedelta
    max_computation_time: timedelta
    cost_weights: Dict[str, float]

@dataclass
class ScenarioParameters(BaseParameters):
    """Complete scenario parameters"""
    name: str
    description: str
    simulation: SimulationParameters
    vehicle: VehicleParameters
    demand: DemandParameters
    stop: StopParameters
    algorithm: AlgorithmParameters
    network_file: Path
    output_directory: Path

@dataclass
class StopSelectionCosts(BaseParameters):
    """Costs related to stop selection and placement"""
    walking_distance: float = 0.3
    coverage: float = 0.2
    accessibility: float = 0.15
    safety: float = 0.15
    demand_density: float = 0.1
    transfer_potential: float = 0.05
    infrastructure: float = 0.05

@dataclass
class MatchingCosts(BaseParameters):
    """Costs related to matching passengers to vehicles"""
    wait_time: float = 0.3
    detour_time: float = 0.25
    vehicle_capacity: float = 0.15
    ride_time: float = 0.15
    passenger_preference: float = 0.1
    vehicle_suitability: float = 0.05

@dataclass
class DispatchCosts(BaseParameters):
    """Costs related to vehicle dispatch decisions"""
    idle_time: float = 0.25
    deadhead_distance: float = 0.2
    fleet_distribution: float = 0.15
    energy_consumption: float = 0.15
    vehicle_load: float = 0.15
    time_to_pickup: float = 0.1

@dataclass
class RoutingCosts(BaseParameters):
    """Costs related to route calculation"""
    distance: float = 0.3
    travel_time: float = 0.25
    reliability: float = 0.15
    road_type: float = 0.1
    traffic_conditions: float = 0.1
    turns_complexity: float = 0.05
    elevation: float = 0.05

@dataclass
class RebalancingCosts(BaseParameters):
    """Costs related to vehicle rebalancing"""
    demand_prediction: float = 0.3
    current_coverage: float = 0.25
    distance_to_hotspot: float = 0.2
    time_of_day: float = 0.15
    historical_patterns: float = 0.1

@dataclass
class AlgorithmParameters(BaseParameters):
    """Refined algorithm-specific parameters"""
    # Algorithm selections
    routing_algorithm: AlgorithmType
    dispatch_algorithm: AlgorithmType
    matching_algorithm: AlgorithmType
    matching_policy: VehicleMatchingPolicy
    
    # Timing parameters
    reoptimization_interval: timedelta
    max_computation_time: timedelta
    
    # Component-specific costs
    stop_selection_costs: StopSelectionCosts
    matching_costs: MatchingCosts
    dispatch_costs: DispatchCosts
    routing_costs: RoutingCosts
    rebalancing_costs: RebalancingCosts
    
    # Additional algorithm-specific parameters
    batch_size: int = 10
    batch_interval: timedelta = timedelta(seconds=30)
    max_matches_per_vehicle: int = 3
    max_waiting_requests: int = 100
    min_pickup_delay: timedelta = timedelta(minutes=5)
    max_pickup_delay: timedelta = timedelta(minutes=30)
    
    def validate_costs(self):
        """Validate that each cost component sums to 1.0"""
        cost_groups = [
            ('stop_selection_costs', self.stop_selection_costs),
            ('matching_costs', self.matching_costs),
            ('dispatch_costs', self.dispatch_costs),
            ('routing_costs', self.routing_costs),
            ('rebalancing_costs', self.rebalancing_costs)
        ]
        
        for name, cost_group in cost_groups:
            total = sum(float(getattr(cost_group, field.name)) 
                       for field in fields(cost_group))
            if not math.isclose(total, 1.0, rel_tol=1e-5):
                raise ValueError(
                    f"{name} weights sum to {total}, expected 1.0"
                )