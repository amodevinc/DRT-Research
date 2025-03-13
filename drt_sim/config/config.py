from dataclasses import dataclass, field, is_dataclass
from enum import Enum
from typing import List, Optional, Dict, Any, Union, Tuple
from datetime import datetime, timedelta
from pathlib import Path
from drt_sim.models.matching.enums import AssignmentMethod, OptimizationMethod
from copy import deepcopy
import yaml
import logging
import math
logger = logging.getLogger(__name__)

class DataclassYAMLMixin:
    """Mixin class to make dataclasses YAML serializable"""
    def to_dict(self) -> Dict[str, Any]:
        """Convert dataclass instance to a dictionary for YAML serialization"""
        def _serialize(obj: Any) -> Any:
            if is_dataclass(obj):
                return {k: _serialize(v) for k, v in obj.__dict__.items()}
            elif isinstance(obj, Enum):
                return obj.value
            elif isinstance(obj, (datetime, timedelta)):
                return str(obj)
            elif isinstance(obj, dict):
                return {k: _serialize(v) for k, v in obj.items()}
            elif isinstance(obj, (list, tuple)):
                return [_serialize(i) for i in obj]
            elif isinstance(obj, Path):
                return str(obj)
            return obj

        return _serialize(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> Any:
        """Create a dataclass instance from a dictionary"""
        def _deserialize(value: Any, field_type: type) -> Any:
            if is_dataclass(field_type):
                return field_type.from_dict(value)
            elif isinstance(field_type, type) and issubclass(field_type, Enum):
                return field_type(value)
            elif field_type == datetime:
                return datetime.fromisoformat(value)
            elif field_type == timedelta:
                return timedelta(seconds=float(value))
            elif field_type == Path:
                return Path(value)
            elif hasattr(field_type, "__origin__"):  # Handle generic types
                if field_type.__origin__ == list:
                    item_type = field_type.__args__[0]
                    return [_deserialize(item, item_type) for item in value]
                elif field_type.__origin__ == dict:
                    key_type, val_type = field_type.__args__
                    return {_deserialize(k, key_type): _deserialize(v, val_type) for k, v in value.items()}
                elif field_type.__origin__ == Union:
                    if type(None) in field_type.__args__ and value is None:
                        return None
                    for arg in field_type.__args__:
                        if arg != type(None):
                            try:
                                return _deserialize(value, arg)
                            except:
                                continue
                    raise ValueError(f"Could not deserialize {value} as {field_type}")
            return value

        field_types = {field.name: field.type for field in cls.__dataclass_fields__.values()}
        kwargs = {key: _deserialize(value, field_types[key]) for key, value in data.items() if key in field_types}
        return cls(**kwargs)

class StudyType(Enum):
    PARAMETER_SWEEP = "parameter_sweep"
    SCENARIO_COMPARISON = "scenario_comparison"
    SENSITIVITY_ANALYSIS = "sensitivity_analysis"
    VALIDATION = "validation"

@dataclass
class StudyMetadata(DataclassYAMLMixin):
    """Metadata about the study"""
    name: str = ""
    description: str = ""
    tags: List[str] = field(default_factory=list)
    authors: List[str] = field(default_factory=list)
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    version: str = "1.0.0"

@dataclass
class StudyPaths(DataclassYAMLMixin):
    """Paths configuration for study outputs"""
    base_dir: str = "studies"
    results_dir: str = "results"
    configs_dir: str = "configs"
    logs_dir: str = "logs"
    artifacts_dir: str = "artifacts"

@dataclass
class MLflowConfig(DataclassYAMLMixin):
    """MLflow configuration"""
    tracking_uri: str = "sqlite:///mlflow.db"
    experiment_name: str = ""
    artifact_location: Optional[str] = None
    tags: Dict[str, str] = field(default_factory=dict)

@dataclass
class RayConfig(DataclassYAMLMixin):
    address: str = "auto"
    num_cpus: Optional[int] = None
    num_gpus: Optional[int] = None
    include_dashboard: bool = True
    ignore_reinit_error: bool = True
    log_to_driver: bool = True

@dataclass
class ExecutionConfig(DataclassYAMLMixin):
    """Execution configuration"""
    distributed: bool = False
    max_parallel: int = 4
    continue_on_error: bool = False
    save_intermediate: bool = True

@dataclass
class NetworkInfo:
    """Stores basic information about a loaded network"""
    name: str
    node_count: int
    edge_count: int
    bbox: Tuple[float, float, float, float]  # min_lon, min_lat, max_lon, max_lat
    crs: str

@dataclass
class NetworkConfig(DataclassYAMLMixin):
    network_file: str = ""
    walk_network_file: Optional[str] = None
    transfer_points_file: Optional[str] = None
    coordinate_system: str = "EPSG:4326"
    walking_speed: float = 1.4
    driving_speed: float = 8.33
    service_area_polygon: Optional[List[Tuple[float, float]]] = None

@dataclass
class CSVFileConfig(DataclassYAMLMixin):
    file_path: str
    weight: float = 1.0
    start_time: Optional[str] = None
    end_time: Optional[str] = None
    region: Optional[str] = None

@dataclass
class CSVDemandGeneratorConfig(DataclassYAMLMixin):
    files: List[CSVFileConfig] = field(default_factory=list)
    datetime_format: str = "%Y-%m-%d %H:%M:%S"
    columns: Dict[str, str] = field(default_factory=lambda: {
        "request_time": "request_time",
        "pickup_lat": "pickup_lat",
        "pickup_lon": "pickup_lon",
        "dropoff_lat": "dropoff_lat",
        "dropoff_lon": "dropoff_lon",
    })
    demand_multiplier: float = 1.0
    combine_method: str = "weighted_sum"
    service_area_polygon: Optional[List[Tuple[float, float]]] = None

    def __post_init__(self):
        """Initialize nested configurations"""
        self.files = [
            f if isinstance(f, CSVFileConfig) else CSVFileConfig(**f)
            for f in self.files
        ]

@dataclass
class RandomDemandGeneratorConfig(DataclassYAMLMixin):
    demand_level: float = 0.0
    spatial_distribution: str = "uniform"
    temporal_distribution: str = "uniform"
    service_area: Tuple[Tuple[float, float], Tuple[float, float]] = field(
        default_factory=lambda: ((0.0, 0.0), (1.0, 1.0))
    )
    hotspots: Optional[List[Dict[str, Union[Tuple[float, float], float]]]] = None

@dataclass
class DemandConfig(DataclassYAMLMixin):
    generator_type: str = "csv"
    csv_config: CSVDemandGeneratorConfig = field(default_factory=CSVDemandGeneratorConfig)
    random_config: Optional[RandomDemandGeneratorConfig] = None
    num_requests: int = 9999
    
    def __post_init__(self):
        """Initialize nested configurations"""
        # Convert csv_config if it's a dictionary
        if isinstance(self.csv_config, dict):
            self.csv_config = CSVDemandGeneratorConfig(**self.csv_config)
            
        # Convert random_config if it's a dictionary
        if isinstance(self.random_config, dict):
            self.random_config = RandomDemandGeneratorConfig(**self.random_config)

@dataclass
class VehicleConfig(DataclassYAMLMixin):
    fleet_size: int = 10
    capacity: int = 4
    speed: float = 10.0
    boarding_time: int = 10
    alighting_time: int = 10
    min_dwell_time: int = 10
    max_dwell_time: int = 300
    max_pickup_delay: int = 10
    max_dropoff_delay: int = 10
    depot_locations: List[Tuple[float, float]] = field(default_factory=lambda: [(37.5666, 127.0000)])
    battery_capacity: Optional[float] = None
    charging_rate: Optional[float] = None
class CostFactorType(Enum):
    """Types of cost factors that can be considered in matching"""
    DISTANCE = "distance"
    TIME = "time"
    DELAY = "delay"
    WAITING_TIME = "waiting_time"
    DETOUR_TIME = "detour_time"
    CAPACITY_UTILIZATION = "capacity_utilization"

class MatchingAssignmentConfig:
    """Configuration for matching assignment."""

    def __init__(
        self,
        constraints: Dict[str, float],
        weights: Dict[str, float],
        reserve_price: Optional[float] = None
    ):
        """Initialize matching assignment configuration.
        
        Args:
            constraints: Dictionary of constraint values including:
                - max_waiting_time_secs: Maximum passenger waiting time in seconds
                - max_in_vehicle_time_secs: Maximum in-vehicle time in seconds
                - max_vehicle_access_time_secs: Maximum vehicle access time in seconds
                - max_detour_time_secs: Maximum detour time in seconds
                - max_existing_passenger_delay_secs: Maximum delay for existing passengers in seconds
                - max_distance_meters: Maximum distance in meters
            weights: Dictionary of weights for optimization including:
                - passenger_waiting_time: Weight for passenger waiting time
                - passenger_in_vehicle_time: Weight for passenger in-vehicle time
                - existing_passenger_delay: Weight for delay to existing passengers
                - vehicle_detour: Weight for vehicle detour
                - distance: Weight for distance
                - passenger_walk_time: Weight for passenger walking time
                - operational_cost: Weight for operational costs
            reserve_price: Optional reserve price for auction-based assignment
        """
        self.constraints = constraints
        self.weights = weights
        self.reserve_price = reserve_price
        
        self._validate_constraints()
        self._validate_weights()
    
    def _validate_constraints(self):
        """Validate that all required constraints are present and have valid values."""
        required_constraints = {
            "max_waiting_time_secs",
            "max_in_vehicle_time_secs", 
            "max_vehicle_access_time_secs",
            "max_existing_passenger_delay_secs",
            "max_distance_meters",
        }
        
        # Check all required constraints are present
        missing = required_constraints - set(self.constraints.keys())
        if missing:
            raise ValueError(f"Missing required constraints: {missing}")
            
        # Validate constraint values
        for key, value in self.constraints.items():
            if not isinstance(value, (int, float)):
                raise ValueError(f"Constraint {key} must be numeric")
            if value <= 0:
                raise ValueError(f"Constraint {key} must be positive")
    
    def _validate_weights(self):
        """Validate that all required weights are present and have valid values."""
        required_weights = {
            "passenger_waiting_time",
            "passenger_in_vehicle_time",
            "existing_passenger_delay",
            "distance",
        }
        
        # Check all required weights are present
        missing = required_weights - set(self.weights.keys())
        if missing:
            raise ValueError(f"Missing required weights: {missing}")
            
        # Validate weight values
        for key, value in self.weights.items():
            if not isinstance(value, (int, float)):
                raise ValueError(f"Weight {key} must be numeric")
            if value < 0:
                raise ValueError(f"Weight {key} must be non-negative")
        
        # Validate weights sum to 1
        total = sum(self.weights.values())
        if not math.isclose(total, 1.0, rel_tol=1e-5):
            raise ValueError(f"Weights must sum to 1.0, got {total}")

@dataclass 
class MatchingClusteringConfig(DataclassYAMLMixin):
    """Configuration for matching clustering"""
    max_cluster_size: int = 10
    max_cluster_radius: float = 1000.0
    min_cluster_size: int = 2
    time_window: int = 300
    spatial_weight: float = 0.6
    temporal_weight: float = 0.4

@dataclass
class MatchingOptimizationConfig(DataclassYAMLMixin):
    """Configuration for matching optimization"""
    optimization_interval: int = 300
    max_optimization_time: int = 60
    improvement_threshold: float = 0.05
    max_iterations: int = 100
    convergence_threshold: float = 0.01

@dataclass
class MatchingConfig(DataclassYAMLMixin):
    """Configuration for matching strategy"""
    assignment_method: AssignmentMethod = AssignmentMethod.INSERTION
    optimization_method: OptimizationMethod = OptimizationMethod.NONE
    assignment_config: MatchingAssignmentConfig = field(default_factory=MatchingAssignmentConfig)
    optimization_config: MatchingOptimizationConfig = field(default_factory=MatchingOptimizationConfig)

    def __post_init__(self):
        """Initialize nested configurations and convert string values to enums"""
        if isinstance(self.assignment_method, str):
            self.assignment_method = AssignmentMethod(self.assignment_method)
        if isinstance(self.optimization_method, str):
            self.optimization_method = OptimizationMethod(self.optimization_method)

        if isinstance(self.assignment_config, dict):
            self.assignment_config = MatchingAssignmentConfig(**self.assignment_config)
        if isinstance(self.optimization_config, dict):
            self.optimization_config = MatchingOptimizationConfig(**self.optimization_config)

@dataclass
class StopAssignerConfig(DataclassYAMLMixin):
    """Configuration for stop assignment"""
    strategy: str = "nearest"
    max_alternatives: int = 3
    thresholds: Dict[str, float] = field(default_factory=lambda: {
        "max_walking_distance": 400.0,
        "max_driving_time": 900.0,
    })
    weights: Dict[str, float] = field(default_factory=lambda: {
        "vehicle_access_time": 0.3,
        "passenger_access_time": 0.7,
    })
    custom_params: Dict[str, Any] = field(default_factory=dict)

@dataclass
class StopSelectorConfig(DataclassYAMLMixin):
    """Configuration for stop selection"""
    strategy: str = "coverage_based"
    min_demand_threshold: float = 0.1
    max_walking_distance: float = 400.0
    min_stop_spacing: float = 100.0
    max_stops: int = 10
    coverage_radius: float = 1000.0
    accessibility_weights: Dict[str, float] = field(default_factory=lambda: {
        "walk": 0.4,
        "demand": 0.3,
        "vehicle": 0.2,
        "coverage": 0.1
    })
    custom_params: Dict[str, Any] = field(default_factory=dict)

@dataclass
class AlgorithmConfig(DataclassYAMLMixin):
    """Configuration for system algorithms"""
    routing_algorithm: str = "dijkstra"
    cost_function: str = "simple"
    user_acceptance_model: str = "default"
    rebalancing_interval: int = 300
    stop_selector: str = "coverage_based"
    stop_assigner: str = "nearest"
    rebalancing_algorithm: str = "naive"

    routing_params: Optional[Dict[str, Any]] = None
    cost_function_params: Optional[Dict[str, Any]] = None
    user_acceptance_params: Optional[Dict[str, Any]] = None
    stop_selector_params: Optional[Dict[str, Any]] = None
    stop_assigner_params: Optional[Dict[str, Any]] = None
    rebalancing_params: Optional[Dict[str, Any]] = None
    def __post_init__(self):
        """Initialize nested configurations"""
        # Convert dictionary parameters if they're provided
        if isinstance(self.stop_selector_params, dict):
            self.stop_selector_params = deepcopy(self.stop_selector_params)
            
        if isinstance(self.stop_assigner_params, dict):
            self.stop_assigner_params = deepcopy(self.stop_assigner_params)
            
        logger.info(f"Final stop_selector_params: {self.stop_selector_params}")
        logger.info(f"Final stop_assigner_params: {self.stop_assigner_params}")

@dataclass
class UserAcceptanceConfig(DataclassYAMLMixin):
    """Configuration for user acceptance models"""
    
    # Model configuration
    model: Dict[str, Any] = field(default_factory=lambda: {
        "type": "default",
        "parameters": {
            "max_walking_time_to_origin": 15.0,
            "max_walking_time_from_destination": 45.0,
            "max_waiting_time": 30.0,
            "max_in_vehicle_time": 25.0,
            "max_cost": 30.0,
            "feature_weights": {
                "walking_time_to_origin": -0.4,
                "walking_time_from_destination": -0.3,
                "waiting_time": -0.3,
                "in_vehicle_time": -0.2,
                "cost": -0.1
            }
        }
    })
    
    # General settings
    max_history_size: int = 1000
    enable_user_specific_models: bool = False
    default_acceptance_threshold: float = 0.5
    learning_rate: float = 0.01
    user_profiles_dir_path: str = "data/users/user_profiles"
    
    # Feature extractor configuration
    feature_extractor_config: Dict[str, Any] = field(default_factory=lambda: {
        "feature_registry": None,
        "enabled_features": None,
        "normalization_overrides": {
            "walking_time_to_origin": 15.0,
            "walking_time_from_destination": 15.0,
            "waiting_time": 30.0,
            "in_vehicle_time": 60.0,
            "cost": 50.0
        }
    })
    
    # Feature provider configuration
    feature_provider_config: Dict[str, Any] = field(default_factory=lambda: {
        "time_based": {
            "enabled": True,
            "holidays": []
        },
        "spatial": {
            "enabled": True,
            "urban_areas": {},
            "region_info": {}
        },
        "user_history": {
            "enabled": True
        },
        "weather": {
            "enabled": False,
            "service": None
        },
        "service_quality": {
            "enabled": True
        },
        "custom_providers": []
    })
    
    # Feature weights (used as defaults)
    feature_weights: Dict[str, float] = field(default_factory=lambda: {
        "walking_time_to_origin": -0.4,
        "walking_time_from_destination": -0.3,
        "waiting_time": -0.4,
        "in_vehicle_time": -0.2,
        "cost": -0.1
    })
    
    # Custom parameters for extensibility
    custom_params: Dict[str, Any] = field(default_factory=dict)

    def get(self, key: str, default: Any = None) -> Any:
        """Get a configuration value by key"""
        return getattr(self, key, default)
    
    def __post_init__(self):
        """Initialize nested configurations"""
        # Ensure model is a dictionary
        if not isinstance(self.model, dict):
            self.model = {
                "type": "default",
                "parameters": {
                    "max_walking_time_to_origin": 15.0,
                    "max_walking_time_from_destination": 15.0,
                    "max_waiting_time": 30.0,
                    "max_in_vehicle_time": 60.0,
                    "max_cost": 50.0,
                    "feature_weights": {
                        "walking_time_to_origin": -0.4,
                        "walking_time_from_destination": -0.3,
                        "waiting_time": -0.3,
                        "in_vehicle_time": -0.2,
                        "cost": -0.1
                    }
                }
            }
        
        # Ensure model has type and parameters
        if "type" not in self.model:
            self.model["type"] = "default"
        if "parameters" not in self.model:
            self.model["parameters"] = {}
            
        # Set default parameters based on model type if not provided
        model_type = self.model["type"]
        if model_type == "default" and not self.model["parameters"]:
            self.model["parameters"] = {
                "max_walking_time_to_origin": 15.0,
                "max_walking_time_from_destination": 15.0,
                "max_waiting_time": 30.0,
                "max_in_vehicle_time": 60.0,
                "max_cost": 50.0,
                "feature_weights": {
                    "walking_time_to_origin": -0.4,
                    "walking_time_from_destination": -0.3,
                    "waiting_time": -0.3,
                    "in_vehicle_time": -0.2,
                    "cost": -0.1
                }
            }
        elif model_type == "logit" and not self.model["parameters"]:
            self.model["parameters"] = {
                "default_coefficients": {
                    "walking_time_to_origin": -2.0,
                    "walking_time_from_destination": -1.5,
                    "waiting_time": -2.5,
                    "in_vehicle_time": -1.5,
                    "cost": -2.0
                },
                "max_training_samples": 1000,
                "logistic_params": {
                    "solver": "lbfgs",
                    "max_iter": 1000,
                    "class_weight": "balanced"
                }
            }
        elif model_type == "rl" and not self.model["parameters"]:
            self.model["parameters"] = {
                "alpha": 0.1,
                "gamma": 0.9,
                "epsilon": 0.2,
                "min_epsilon": 0.05,
                "epsilon_decay": 0.9999,
                "num_bins": 10,
                "memory_size": 1000,
                "batch_size": 64
            }
        
        # Ensure feature extractor config exists
        if not isinstance(self.feature_extractor_config, dict):
            self.feature_extractor_config = {
                "feature_registry": None,
                "enabled_features": None,
                "normalization_overrides": {
                    "walking_time_to_origin": 15.0,
                    "walking_time_from_destination": 15.0,
                    "waiting_time": 30.0,
                    "in_vehicle_time": 60.0,
                    "cost": 50.0
                }
            }
        
        # Ensure feature provider config exists
        if not isinstance(self.feature_provider_config, dict):
            self.feature_provider_config = {
                "time_based": {
                    "enabled": True,
                    "holidays": []
                },
                "spatial": {
                    "enabled": True,
                    "urban_areas": {},
                    "region_info": {}
                },
                "user_history": {
                    "enabled": True
                },
                "weather": {
                    "enabled": False,
                    "service": None
                },
                "service_quality": {
                    "enabled": True
                },
                "custom_providers": []
            }
    
    def get_model_config(self) -> Dict[str, Any]:
        """Get the model configuration in a format suitable for the ModelFactory"""
        return {
            "model_type": self.model["type"],
            "parameters": self.model["parameters"],
            "feature_extractor_config": self.feature_extractor_config
        }
    
    def set_model_type(self, model_type: str) -> None:
        """
        Set the model type and update parameters with appropriate defaults.
        
        Args:
            model_type: The model type (e.g., "default", "logit", "rl")
        """
        old_type = self.model.get("type", "default")
        self.model["type"] = model_type
        
        # If changing model type and parameters are still for the old type,
        # update with appropriate defaults
        if old_type != model_type and self.model.get("parameters", {}):
            self.__post_init__()
    
    def update_model_parameters(self, parameters: Dict[str, Any]) -> None:
        """
        Update model parameters.
        
        Args:
            parameters: Dictionary of parameters to update
        """
        if "parameters" not in self.model:
            self.model["parameters"] = {}
            
        self.model["parameters"].update(parameters)
    
    def update_feature_extractor_config(self, config: Dict[str, Any]) -> None:
        """
        Update feature extractor configuration.
        
        Args:
            config: Dictionary of configuration parameters to update
        """
        self.feature_extractor_config.update(config)
    
    def enable_feature_provider(self, provider_name: str, config: Optional[Dict[str, Any]] = None) -> None:
        """
        Enable a feature provider and optionally update its configuration.
        
        Args:
            provider_name: Name of the provider
            config: Configuration parameters for the provider
        """
        if provider_name not in self.feature_provider_config:
            self.feature_provider_config[provider_name] = {"enabled": True}
        else:
            self.feature_provider_config[provider_name]["enabled"] = True
            
        if config:
            self.feature_provider_config[provider_name].update(config)
    
    def disable_feature_provider(self, provider_name: str) -> None:
        """
        Disable a feature provider.
        
        Args:
            provider_name: Name of the provider
        """
        if provider_name in self.feature_provider_config:
            self.feature_provider_config[provider_name]["enabled"] = False
    
    def add_custom_feature_provider(self, name: str, class_path: str, args: Optional[Dict[str, Any]] = None) -> None:
        """
        Add a custom feature provider configuration.
        
        Args:
            name: Provider name
            class_path: Full import path to provider class (e.g., "my_package.providers.MyProvider")
            args: Arguments to pass to the provider constructor
        """
        if "custom_providers" not in self.feature_provider_config:
            self.feature_provider_config["custom_providers"] = []
            
        provider_config = {
            "name": name,
            "class": class_path
        }
        
        if args:
            provider_config["args"] = args
            
        # Remove any existing provider with the same name
        self.feature_provider_config["custom_providers"] = [
            p for p in self.feature_provider_config["custom_providers"] 
            if p.get("name") != name
        ]
        
        # Add the new provider
        self.feature_provider_config["custom_providers"].append(provider_config)
        
@dataclass
class SUMOConfig(DataclassYAMLMixin):
    """Configuration for SUMO integration"""
    enabled: bool = False
    sumo_binary: str = "sumo-gui"  # Use "sumo" for headless mode
    network_file: Optional[str] = None
    route_file: Optional[str] = None
    additional_files: List[str] = field(default_factory=list)
    gui_settings_file: Optional[str] = None
    step_length: float = 1.0
    begin_time: float = 0.0
    end_time: float = 86400.0
    use_geo_coordinates: bool = True
    port: int = 8813
    seed: int = 42
    auto_convert_network: bool = True
    visualization: bool = True
    custom_params: Dict[str, Any] = field(default_factory=dict)
@dataclass
class SimulationConfig(DataclassYAMLMixin):
    start_time: str = "2025-01-01 07:00:00"
    end_time: str = "2025-01-01 19:00:00"
    duration: int = 86400
    warm_up_duration: int = 1800
    random_seed: int = 42
    time_step: int = 60
    time_scale_factor: float = 1.0
    save_state: bool = True
    save_interval: int = 3600
    sumo: SUMOConfig = field(default_factory=SUMOConfig)

    def __post_init__(self):
        """Initialize nested configurations"""
        if isinstance(self.sumo, dict):
            self.sumo = SUMOConfig(**self.sumo)

@dataclass
class ParameterSweepConfig(DataclassYAMLMixin):
    enabled: bool = False
    method: str = "grid"
    parameters: Dict[str, List[Any]] = field(default_factory=dict)
    n_samples: Optional[int] = None
    seed: int = 42

@dataclass
class ServiceConfig(DataclassYAMLMixin):
    """Configuration for service handling"""
    max_wait_time: int = 600
    max_ride_time: int = 600
    max_walking_distance: float = 400.0
    max_journey_time: int = 1200

@dataclass
class RouteConfig(DataclassYAMLMixin):
    """Configuration for route handling"""
    max_detour_factor: float = 1.5
    max_route_duration: int = 1800
    max_stops_per_route: int = 10
    max_segment_delay: int = 600
    reoptimization_trigger_delay: int = 600

@dataclass
class StopConfig(DataclassYAMLMixin):
    """Configuration for stop handling"""
    max_occupancy: int = 5
    congestion_threshold: int = 2
    max_vehicle_queue: int = 10
    max_passenger_queue: int = 10
    min_service_interval: int = 60
    max_dwell_time: int = 120
@dataclass
class ParameterSet(DataclassYAMLMixin):
    """A set of parameters to run a simulation with"""
    name: str
    description: str = ""
    service: ServiceConfig = field(default_factory=ServiceConfig)
    route: RouteConfig = field(default_factory=RouteConfig)
    stop: StopConfig = field(default_factory=StopConfig)
    demand: DemandConfig = field(default_factory=DemandConfig)
    vehicle: VehicleConfig = field(default_factory=VehicleConfig)
    algorithm: AlgorithmConfig = field(default_factory=AlgorithmConfig)
    matching: MatchingConfig = field(default_factory=MatchingConfig)
    network: NetworkConfig = field(default_factory=NetworkConfig)
    user_acceptance: UserAcceptanceConfig = field(default_factory=UserAcceptanceConfig)
    replications: int = 1
    tags: List[str] = field(default_factory=list)

    def __post_init__(self):
        """Initialize nested configurations"""
        if isinstance(self.service, dict):
            self.service = ServiceConfig(**self.service)
        if isinstance(self.route, dict):
            self.route = RouteConfig(**self.route)
        if isinstance(self.stop, dict):
            self.stop = StopConfig(**self.stop)
        if isinstance(self.demand, dict):
            self.demand = DemandConfig(**self.demand)
        if isinstance(self.vehicle, dict):
            self.vehicle = VehicleConfig(**self.vehicle)
        if isinstance(self.algorithm, dict):
            self.algorithm = AlgorithmConfig(**self.algorithm)
        if isinstance(self.matching, dict):
            self.matching = MatchingConfig(**self.matching)
        if isinstance(self.network, dict):
            self.network = NetworkConfig(**self.network)
        if isinstance(self.user_acceptance, dict):
            self.user_acceptance = UserAcceptanceConfig(**self.user_acceptance)

@dataclass
class StudyConfig(DataclassYAMLMixin):
    name: str = ""
    description: str = ""
    version: str = ""
    authors: List[str] = field(default_factory=list)
    tags: List[str] = field(default_factory=list)
    mlflow: MLflowConfig = field(default_factory=MLflowConfig)
    execution: ExecutionConfig = field(default_factory=ExecutionConfig)
    simulation: SimulationConfig = field(default_factory=SimulationConfig)
    # The base_config contains keys such as network, demand, service, algorithm.
    base_config: Dict[str, Any] = field(default_factory=dict)
    # parameter_sets is a dict mapping names (e.g., "small_fleet") to dicts with overrides.
    parameter_sets: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def load(cls, path: str) -> "StudyConfig":
        """
        Load a StudyConfig from a YAML file.
        """
        with open(path, 'r') as f:
            data = yaml.safe_load(f)

        # Load nested configurations for mlflow, execution, and simulation.
        mlflow_config = MLflowConfig(**data.get("mlflow", {}))
        execution_config = ExecutionConfig(**data.get("execution", {}))
        simulation_config = SimulationConfig(**data.get("simulation", {}))

        instance = cls(
            name=data.get("name", ""),
            description=data.get("description", ""),
            version=data.get("version", ""),
            authors=data.get("authors", []),
            tags=data.get("tags", []),
            mlflow=mlflow_config,
            execution=execution_config,
            simulation=simulation_config,
            base_config=data.get("base_config", {}),
            parameter_sets=data.get("parameter_sets", {})
        )
        return instance

    def dump(self, path: str):
        """
        Dump the current StudyConfig to a YAML file.
        """
        data = {
            "name": self.name,
            "description": self.description,
            "version": self.version,
            "authors": self.authors,
            "tags": self.tags,
            "mlflow": self.mlflow.__dict__,
            "execution": self.execution.__dict__,
            "simulation": self.simulation.__dict__,
            "base_config": self.base_config,
            "parameter_sets": self.parameter_sets,
        }
        with open(path, 'w') as f:
            yaml.dump(data, f)

    def get_parameter_set(self, name: str) -> ParameterSet:
        """
        Retrieve a ParameterSet by name, merging the base_config with any
        overrides specified in the given parameter set. The merging is done
        recursively so that nested dictionaries are properly updated.
        """
        if name not in self.parameter_sets:
            raise ValueError(f"Parameter set '{name}' not found.")

        # Create a deep copy of the base configuration.
        base = deepcopy(self.base_config)
        overrides = self.parameter_sets[name]

        # Merge the overrides into the base configuration recursively.
        merged_config = self.deep_update(base, overrides)

        # Prepare a dictionary for the ParameterSet.
        # Notice that base_config in the YAML provided keys for service, demand,
        # algorithm, and network, while the parameter set (e.g., "small_fleet")
        # provides additional keys such as vehicle and matching.
        params = {
            "name": merged_config.get("name", name),
            "description": merged_config.get("description", ""),
            "service": merged_config.get("service", {}),
            "route": merged_config.get("route", {}),  # default empty if not provided
            "stop": merged_config.get("stop", {}),      # default empty if not provided
            "demand": merged_config.get("demand", {}),
            "vehicle": merged_config.get("vehicle", {}),
            "algorithm": merged_config.get("algorithm", {}),
            "matching": merged_config.get("matching", {}),
            "network": merged_config.get("network", {}),
            "user_acceptance": merged_config.get("user_acceptance", {}),
            "replications": merged_config.get("replications", 1),
            "tags": merged_config.get("tags", []),
        }
        return ParameterSet(**params)
    
    @staticmethod
    def deep_update(source: Dict[str, Any], overrides: Dict[str, Any]) -> Dict[str, Any]:
        """
        Recursively update a dictionary with another dictionary. If a key exists in both
        and the corresponding values are dictionaries, then update them recursively.
        Otherwise, the override value will replace the source value.
        """
        for key, value in overrides.items():
            if key in source and isinstance(source[key], dict) and isinstance(value, dict):
                source[key] = StudyConfig.deep_update(source[key], value)
            else:
                source[key] = value
        return source

