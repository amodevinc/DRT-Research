# drt_sim/config/parameters.py

from __future__ import annotations
from typing import Optional, List, Dict, Union, Any
from datetime import datetime, timedelta
from enum import Enum
from pydantic import BaseModel, Field, validator
from pathlib import Path
import math

class AlgorithmType(Enum):
    """Algorithm types with research-relevant descriptions"""
    # Routing algorithms
    ROUTING_DIJKSTRA = "dijkstra"
    ROUTING_ASTAR = "astar"
    ROUTING_COST_MINIMIZATION = "routing_cost_minimization"
    
    # Vehicle dispatch algorithms
    DISPATCH_IMMEDIATE = "immediate_dispatch"
    DISPATCH_BATCH = "batch_dispatch"
    DISPATCH_ANTICIPATORY = "anticipatory_dispatch"
    DISPATCH_FCFS = "fcfs_dispatch"
    
    # Request-vehicle matching algorithms
    MATCHING_INSERTION = "insertion_heuristic"
    MATCHING_AUCTION = "auction_based"
    MATCHING_GENETIC = "genetic_algorithm"
    
    # Stop selection algorithms
    STOP_KMEANS = "kmeans_clustering"
    STOP_DEMAND = "demand_based"
    STOP_COVERAGE = "coverage_based"
    STOP_ACCESSIBILITY = "accessibility_based"
    STOP_MULTI_OBJECTIVE = "multi_objective_cost_minimization"

class ExperimentType(Enum):
    """Types of experiments that can be conducted"""
    ALGORITHM_COMPARISON = "algorithm_comparison"
    PARAMETER_SENSITIVITY = "parameter_sensitivity"
    DEMAND_PATTERN = "demand_pattern"
    FLEET_OPTIMIZATION = "fleet_optimization"
    STOP_STRATEGY = "stop_strategy"
    MULTI_FACTOR = "multi_factor"

class BaseParameters(BaseModel):
    """Base class for all parameter models with experiment support"""
    class Config:
        arbitrary_types_allowed = True
        json_encoders = {
            datetime: lambda v: v.isoformat(),
            timedelta: lambda v: str(v)
        }

    def modify_for_experiment(self, modifications: Dict[str, Any]) -> BaseParameters:
        """Create a new instance with modifications for an experiment"""
        data = self.dict()
        # Special handling for DemandParameters
        if isinstance(self, DemandParameters):
            # Check if generator_type is being modified
            new_generator_type = None
            if 'generator_type' in modifications:
                new_generator_type = modifications['generator_type']
            
            # Handle nested generator_config modifications
            if 'generator_config' in modifications:
                mod_config = modifications['generator_config']
                
                # If we have a new generator type, use the new config directly
                if new_generator_type:
                    if hasattr(mod_config, 'dict'):
                        data['generator_type'] = new_generator_type
                        data['generator_config'] = mod_config.dict()
                    else:
                        data['generator_type'] = new_generator_type
                        data['generator_config'] = mod_config
                else:
                    # Otherwise, merge with existing config
                    current_config = data['generator_config']
                    
                    # Convert configs to dict if they're Pydantic models
                    if hasattr(mod_config, 'dict'):
                        mod_config = mod_config.dict()
                    if hasattr(current_config, 'dict'):
                        current_config = current_config.dict()
                    
                    # Merge configs
                    if isinstance(mod_config, dict):
                        data['generator_config'] = {**current_config, **mod_config}
                    else:
                        data['generator_config'] = mod_config
                
                # Remove from modifications to prevent double-processing
                modifications = {k: v for k, v in modifications.items() if k != 'generator_config' and k != 'generator_type'}

        # Apply remaining modifications
        for key, value in modifications.items():
            if "." in key:
                # Handle nested modifications
                parts = key.split(".")
                current = data
                for part in parts[:-1]:
                    if part not in current:
                        current[part] = {}
                    current = current[part]
                current[parts[-1]] = value
            else:
                data[key] = value

        return self.__class__(**data)

class SimulationParameters(BaseParameters):
    """Enhanced simulation parameters with research support"""
    # Core timing parameters
    start_time: datetime
    end_time: datetime
    time_step: timedelta
    
    # Simulation control
    random_seed: Optional[int] = Field(None, description="Set for reproducible experiments")
    warm_up_period: timedelta = Field(default=timedelta(hours=1), description="Initial warmup period")
    cool_down_period: timedelta = Field(default=timedelta(hours=1), description="Final cooldown period")
    
    # Technical parameters
    logging_interval: timedelta = Field(default=timedelta(minutes=5))
    reoptimization_interval: timedelta = Field(default=timedelta(minutes=5))
    snapshot_interval: timedelta = Field(default=timedelta(minutes=5))
    snapshot_retention_period: timedelta = Field(default=timedelta(hours=24))
    time_scale_factor: float = Field(default=1.0, description="Speed up or slow down simulation")
    
    # Experiment-specific
    replications: int = Field(default=1, description="Number of replications for statistical validity")
    experiment_tags: List[str] = Field(default_factory=list, description="Tags for experiment categorization")

class DemandGeneratorType(Enum):
    """Types of demand generators available"""
    RANDOM = "random"
    CSV = "csv"
    PREDICTED = "predicted"
    PATTERN_BASED = "pattern_based"
    HISTORICAL = "historical"

class SpatialDistributionType(Enum):
    """Types of spatial distributions for demand"""
    UNIFORM = "uniform"
    CLUSTERED = "clustered"
    HOTSPOT = "hotspot"
    CUSTOM = "custom"

class TemporalDistributionType(Enum):
    """Types of temporal distributions for demand"""
    CONSTANT = "constant"
    PEAK_HOURS = "peak_hours"
    TIME_VARYING = "time_varying"
    CUSTOM = "custom"

class VehicleParameters(BaseParameters):
    """Enhanced vehicle parameters with research support"""
    # Fleet configuration
    fleet_size: int = Field(..., gt=0)
    vehicle_capacity: int = Field(..., gt=0)
    vehicle_speed: float = Field(..., gt=0)  # km/h
    
    # Service times
    boarding_time: timedelta
    alighting_time: timedelta
    
    # Spatial configuration
    depot_locations: List[tuple] = None
    
    # Research-specific
    fleet_compositions: Dict[str, Dict[str, Union[int, List[int]]]] = Field(
        default_factory=dict,
        description="Different fleet compositions for experiments. Can specify single values or lists for parameters."
    )
    cost_factors: Dict[str, float] = Field(
        default_factory=dict,
        description="Vehicle-specific cost factors"
    )

class CostParameters(BaseParameters):
    """Base class for all cost-related parameters"""
    def __init__(self, **data):
        super().__init__(**data)
        self.validate_costs()

    def validate_costs(self):
        """Ensure costs sum to 1.0"""
        total = sum(float(getattr(self, field)) for field in self.__fields__)
        if not math.isclose(total, 1.0, rel_tol=1e-5):
            raise ValueError(f"Costs sum to {total}, expected 1.0")

class StopParameters(BaseParameters):
    """Enhanced stop selection parameters"""
    # Physical constraints
    max_walking_distance: float  # meters
    min_stop_spacing: float  # meters
    stop_capacity: int
    
    # Selection criteria
    safety_threshold: float = Field(..., ge=0.0, le=1.0)
    accessibility_weight: float = Field(..., ge=0.0, le=1.0)
    coverage_radius: float  # meters
    
    # Research-specific
    stop_strategies: Dict[str, Dict[str, Any]] = Field(
        default_factory=dict,
        description="Predefined stop selection strategies"
    )

class AlgorithmParameters(BaseParameters):
    """Enhanced algorithm parameters with research capabilities"""
    # Algorithm selections
    routing_algorithm: AlgorithmType
    dispatch_algorithm: AlgorithmType
    matching_algorithm: AlgorithmType
    
    # Timing constraints
    reoptimization_interval: timedelta
    max_computation_time: timedelta
    
    # Component-specific costs
    stop_selection_costs: Dict[str, float]
    matching_costs: Dict[str, float]
    dispatch_costs: Dict[str, float]
    routing_costs: Dict[str, float]
    
    # Operational parameters
    batch_size: int = Field(default=10, gt=0)
    batch_interval: timedelta = Field(default=timedelta(seconds=30))
    max_matches_per_vehicle: int = Field(default=3, gt=0)
    
    # Research-specific
    algorithm_variants: Dict[str, Dict[str, Any]] = Field(
        default_factory=dict,
        description="Predefined algorithm variants for testing"
    )
    
    @validator('stop_selection_costs', 'matching_costs', 'dispatch_costs', 'routing_costs')
    def validate_cost_dict(cls, v):
        """Validate that cost dictionaries sum to 1.0"""
        if abs(sum(v.values()) - 1.0) > 1e-5:
            raise ValueError("Costs must sum to 1.0")
        return v

class ExperimentParameters(BaseParameters):
    """Parameters specific to research experiments"""
    # Experiment identification
    name: str
    type: ExperimentType
    description: str
    tags: List[str] = Field(default_factory=list)
    
    # Execution parameters
    replications: int = Field(default=30, gt=0)
    random_seeds: Optional[List[int]] = None
    
    # Parameter variations
    parameter_ranges: Dict[str, List[Any]] = Field(
        default_factory=dict,
        description="Parameters to vary in the experiment"
    )
    
    # Analysis configuration
    metrics: List[str] = Field(default_factory=list)
    statistical_tests: List[str] = Field(default_factory=list)
    
    @validator('random_seeds')
    def validate_seeds(cls, v, values):
        """Ensure enough random seeds for replications"""
        if v is not None and len(v) < values['replications']:
            raise ValueError("Must provide at least as many random seeds as replications")
        return v

class ScenarioParameters(BaseParameters):
    """Enhanced scenario parameters with research support"""
    # Identification
    name: str
    description: str
    
    # Core components
    simulation: SimulationParameters
    vehicle: VehicleParameters
    demand: DemandParameters
    stop: StopParameters
    algorithm: AlgorithmParameters
    
    # Infrastructure
    network_file: Path
    output_directory: Path
    
    # Research-specific
    experiment: Optional[ExperimentParameters] = None
    variant_tag: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)
    
    def create_variant(
        self,
        variant_name: str,
        modifications: Dict[str, Any]
    ) -> ScenarioParameters:
        """Create a variant of this scenario with specified modifications"""
        data = self.dict()
        data['name'] = f"{self.name}_{variant_name}"
        data['variant_tag'] = variant_name
        
        # Apply modifications
        for key, value in modifications.items():
            components = key.split('.')
            current = data
            for component in components[:-1]:
                current = current[component]
            current[components[-1]] = value
            
        return ScenarioParameters(**data)