from dataclasses import dataclass, field, is_dataclass
from enum import Enum
from typing import List, Optional, Dict, Any, Union, Tuple
from datetime import datetime, timedelta
from pathlib import Path
from drt_sim.models.matching.enums import AssignmentMethod, OptimizationMethod
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

class MetricType(Enum):
    TEMPORAL = "temporal"
    AGGREGATE = "aggregate"
    EVENT = "event"
    CUSTOM = "custom"

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
class MetricDefinition(DataclassYAMLMixin):
    """Definition of a single metric"""
    name: str
    type: MetricType
    unit: str
    description: str
    aggregation: str = "mean"
    dependencies: List[str] = field(default_factory=list)
    custom_function: Optional[str] = None
    threshold: Optional[float] = None
    tags: List[str] = field(default_factory=list)

@dataclass
class MetricsConfig(DataclassYAMLMixin):
    """Configuration for metrics collection and analysis"""
    collect_interval: int = 300
    save_interval: int = 1800
    batch_size: int = 1000
    storage_format: str = "csv"
    compression: Optional[str] = None
    default_metrics: List[str] = field(default_factory=lambda: [
        "waiting_time",
        "in_vehicle_time",
        "total_distance",
        "occupancy_rate",
        "service_rate",
        "rejection_rate"
    ])
    additional_metrics: List[str] = field(default_factory=list)
    definitions: Dict[str, MetricDefinition] = field(default_factory=lambda: {
        "waiting_time": MetricDefinition(
            name="waiting_time",
            type=MetricType.TEMPORAL,
            unit="seconds",
            description="Time between request and pickup"
        ),
        "service_rate": MetricDefinition(
            name="service_rate",
            type=MetricType.AGGREGATE,
            unit="percentage",
            description="Percentage of requests served"
        ),
        "vehicle_utilization": MetricDefinition(
            name="vehicle_utilization",
            type=MetricType.TEMPORAL,
            unit="percentage",
            description="Percentage of time vehicles are occupied"
        ),
        "total_distance": MetricDefinition(
            name="total_distance",
            type=MetricType.AGGREGATE,
            unit="meters",
            description="Total distance traveled by all vehicles"
        ),
        "occupancy_rate": MetricDefinition(
            name="occupancy_rate",
            type=MetricType.TEMPORAL,
            unit="percentage",
            description="Average vehicle occupancy rate"
        ),
        "rejection_rate": MetricDefinition(
            name="rejection_rate",
            type=MetricType.AGGREGATE,
            unit="percentage",
            description="Percentage of requests rejected"
        ),
        "in_vehicle_time": MetricDefinition(
            name="in_vehicle_time",
            type=MetricType.TEMPORAL,
            unit="seconds",
            description="Time spent by passengers in vehicles"
        )
    })
    analysis: Dict[str, Any] = field(default_factory=lambda: {
        "confidence_level": 0.95,
        "statistical_tests": ["t-test", "mann_whitney"],
        "visualization_types": ["boxplot", "timeseries", "histogram"]
    })

    def get_active_metrics(self) -> List[str]:
        return list(set(self.default_metrics + self.additional_metrics))

    def validate_metrics(self) -> bool:
        active_metrics = self.get_active_metrics()
        return all(metric in self.definitions for metric in active_metrics)
    
    def __post_init__(self):
        """Initialize nested configurations"""
        # Convert definition dictionaries to MetricDefinition instances
        self.definitions = {
            k: (v if isinstance(v, MetricDefinition) else MetricDefinition(**v))
            for k, v in self.definitions.items()
        }

@dataclass
class MLflowConfig(DataclassYAMLMixin):
    tracking_uri: str = "sqlite:///experiments/mlflow.db"
    experiment_name: str = ""
    tags: Dict[str, str] = field(default_factory=dict)
    log_artifacts: bool = True
    log_params: bool = True

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
    distributed: bool = False
    max_parallel: int = 4
    chunk_size: int = 1
    timeout: int = 3600
    retry_attempts: int = 3


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
    max_dwell_time: int = 60
    max_pickup_delay: int = 10
    max_dropoff_delay: int = 10
    rebalancing_enabled: bool = False
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

@dataclass
class MatchingAssignmentConfig(DataclassYAMLMixin):
    """Configuration for matching assignment"""
    # Time constraints (in seconds)
    max_waiting_time_mins: int = 10  # mins
    max_detour_time_mins: int = 10  # mins
    max_in_vehicle_time_mins: int = 10  # mins
    max_distance: float = 10000.0  # meters
    max_delay_mins: int = 10  # mins
    capacity_threshold: float = 0.8

    # Base weights for different cost factors
    weights: Dict[str, float] = field(default_factory=lambda: {
        "waiting_time": 0.35,
        "detour_time": 0.25,
        "delay": 0.20,
        "distance": 0.15,
        "capacity_utilization": 0.05
    })

    default_weight: float = 0.1

    def validate_weights(self) -> None:
        """Validate that weights are properly configured"""
        # Ensure all weights are valid cost factors
        valid_factors = {factor.value for factor in CostFactorType}
        invalid_factors = set(self.weights.keys()) - valid_factors
        if invalid_factors:
            raise ValueError(f"Invalid cost factors in weights: {invalid_factors}")
            
        # Normalize weights to sum to 1.0
        total_weight = sum(self.weights.values())
        if total_weight != 0:
            self.weights = {k: v/total_weight for k, v in self.weights.items()}

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
class MatchingConfig:
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
class AlgorithmConfig(DataclassYAMLMixin):
    """Configuration for system algorithms"""
    routing_algorithm: str = "dijkstra"
    cost_function: str = "simple"
    user_acceptance_model: str = "logit"
    rebalancing_interval: int = 300
    stop_selector: str = "coverage_based"
    stop_assigner: str = "nearest"

    routing_params: Optional[Dict[str, Any]] = None
    cost_function_params: Optional[Dict[str, Any]] = None
    user_acceptance_params: Optional[Dict[str, Any]] = None
    stop_selector_params: Optional[Dict[str, Any]] = None
    stop_assigner_params: Optional[Dict[str, Any]] = None

    def __post_init__(self):
        """Initialize parameter objects based on selected strategies"""
        if not self.routing_params:
            self.routing_params = self._get_default_params(self.routing_algorithm)
        if not self.cost_function_params:
            self.cost_function_params = self._get_default_params(self.cost_function)
        if not self.user_acceptance_params:
            self.user_acceptance_params = self._get_default_params(self.user_acceptance_model)
        if not self.stop_selector_params:
            self.stop_selector_params = self._get_default_params(self.stop_selector)
        if not self.stop_assigner_params:
            self.stop_assigner_params = self._get_default_params(self.stop_assigner)



    def _get_default_params(self, strategy: str) -> Dict[str, Any]:
        """Get default parameters for a given strategy"""
        defaults = {
            # Routing defaults
            "dijkstra": {},
            "time_dependent": {
                "update_interval": 300,
                "max_alternatives": 3
            },
            # User acceptance defaults
            "logit": {
                "coefficients": {
                    "waiting_time": -0.01,
                    "travel_time": -0.005,
                    "fare": -0.1
                }
            },
            # Stop selector defaults
            "coverage_based": {
                "strategy": "coverage_based",
                "max_walking_distance": 400.0,
                "min_stop_spacing": 100.0,
                "max_stops": 10,
                "coverage_radius": 1000.0,
                "min_demand_threshold": 0.1,
                "accessibility_weights": {"distance": 0.5, "time": 0.5}
            },
            "demand_based": {
                "strategy": "demand_based",
                "min_demand_threshold": 0.1,
                "max_walking_distance": 400.0,
                "min_stop_spacing": 100.0,
                "max_stops": 10,
                "coverage_radius": 1000.0,
                "accessibility_weights": {"distance": 0.5, "time": 0.5}
            },
            # Stop assigner defaults
            "nearest": {
                "strategy": "nearest",
                "max_walking_distance": 400.0,
                "walking_speed": 1.4,
                "max_wait_time": 600,
                "max_alternatives": 3,
                "weights": {"distance": 0.5, "congestion": 0.3, "accessibility": 0.2}
            },
            "multi_objective": {
                "strategy": "multi_objective",
                "max_walking_distance": 400.0,
                "walking_speed": 1.4,
                "max_wait_time": 600,
                "max_alternatives": 3,
                "weights": {"distance": 0.5, "congestion": 0.3, "accessibility": 0.2}
            },
            "accessibility": {
                "strategy": "accessibility",
                "max_walking_distance": 400.0,
                "walking_speed": 1.4,
                "max_wait_time": 600,
                "max_alternatives": 3,
                "weights": {"distance": 0.5, "congestion": 0.3, "accessibility": 0.2}
            }
        }
        return defaults.get(strategy, {})

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
class ScenarioConfig(DataclassYAMLMixin):
    """Configuration for a specific simulation scenario"""
    name: str
    description: str = ""
    replications: int = 1
    service: Union[ServiceConfig, Dict[str, Any]] = field(default_factory=ServiceConfig)
    route: Union[RouteConfig, Dict[str, Any]] = field(default_factory=RouteConfig)
    stop: Union[StopConfig, Dict[str, Any]] = field(default_factory=StopConfig)
    demand: Union[DemandConfig, Dict[str, Any]] = field(default_factory=DemandConfig)
    vehicle: Union[VehicleConfig, Dict[str, Any]] = field(default_factory=VehicleConfig)
    algorithm: Union[AlgorithmConfig, Dict[str, Any]] = field(default_factory=AlgorithmConfig)
    matching: Union[MatchingConfig, Dict[str, Any]] = field(default_factory=MatchingConfig)
    network: Union[NetworkConfig, Dict[str, Any]] = field(default_factory=NetworkConfig)
    metrics: Optional[Union[MetricsConfig, Dict[str, Any]]] = None

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
        if isinstance(self.metrics, dict):
            self.metrics = MetricsConfig(**self.metrics)

@dataclass
class ExperimentConfig(DataclassYAMLMixin):
    """Configuration for a specific experiment that may contain multiple scenarios"""
    name: str
    description: str
    scenarios: Dict[str, ScenarioConfig]
    metrics: Optional[MetricsConfig] = None
    variant: str = "default"
    tags: List[str] = field(default_factory=list)
    def __post_init__(self):
        """Initialize nested configurations"""
        # Convert scenario dictionaries to ScenarioConfig instances
        self.scenarios = {
            k: (v if isinstance(v, ScenarioConfig) else ScenarioConfig(**v))
            for k, v in self.scenarios.items()
        }
        # Convert metrics dictionary to MetricsConfig instance if needed
        if isinstance(self.metrics, dict):
            self.metrics = MetricsConfig(**self.metrics)

@dataclass
class StudyConfig(DataclassYAMLMixin):
    """Top-level configuration for a simulation study"""
    metadata: StudyMetadata
    type: StudyType
    paths: StudyPaths = field(default_factory=StudyPaths)
    base_config: Dict[str, Any] = field(default_factory=dict)
    experiments: Dict[str, ExperimentConfig] = field(default_factory=dict)
    metrics: MetricsConfig = field(default_factory=MetricsConfig)
    parameter_sweep: Optional[ParameterSweepConfig] = None
    mlflow: MLflowConfig = field(default_factory=MLflowConfig)
    ray: RayConfig = field(default_factory=RayConfig)
    execution: ExecutionConfig = field(default_factory=ExecutionConfig)
    simulation: SimulationConfig = field(default_factory=SimulationConfig)
    settings: Dict[str, Any] = field(default_factory=lambda: {
        "parallel_experiments": False,
        "max_parallel": 1,
        "continue_on_error": False,
        "save_intermediate": True,
        "backup_existing": True
    })

    def __post_init__(self):
        """Initialize all nested configurations"""
        # Convert metadata dictionary to StudyMetadata instance if needed
        if isinstance(self.metadata, dict):
            self.metadata = StudyMetadata(**self.metadata)

        # Convert type string to StudyType enum if needed
        if isinstance(self.type, str):
            self.type = StudyType(self.type)

        # Convert paths dictionary to StudyPaths instance if needed
        if isinstance(self.paths, dict):
            self.paths = StudyPaths(**self.paths)

        # Convert experiments dictionaries to ExperimentConfig instances
        self.experiments = {
            k: (v if isinstance(v, ExperimentConfig) else ExperimentConfig(**v))
            for k, v in self.experiments.items()
        }

        # Convert metrics dictionary to MetricsConfig instance if needed
        if isinstance(self.metrics, dict):
            self.metrics = MetricsConfig(**self.metrics)

        # Convert parameter_sweep dictionary to ParameterSweepConfig instance if needed
        if isinstance(self.parameter_sweep, dict):
            self.parameter_sweep = ParameterSweepConfig(**self.parameter_sweep)

        # Convert other configurations
        if isinstance(self.mlflow, dict):
            self.mlflow = MLflowConfig(**self.mlflow)
        if isinstance(self.ray, dict):
            self.ray = RayConfig(**self.ray)
        if isinstance(self.execution, dict):
            self.execution = ExecutionConfig(**self.execution)
        if isinstance(self.simulation, dict):
            self.simulation = SimulationConfig(**self.simulation)

        # Set default name if not provided
        if not self.metadata.name:
            self.metadata.name = f"study_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    def get_scenario_metrics(self, experiment_name: str, scenario_name: str) -> MetricsConfig:
        """Get metrics configuration for a specific scenario with inheritance"""
        if experiment_name not in self.experiments:
            raise KeyError(f"Experiment '{experiment_name}' not found")
        
        experiment = self.experiments[experiment_name]
        if scenario_name not in experiment.scenarios:
            raise KeyError(f"Scenario '{scenario_name}' not found in experiment '{experiment_name}'")
        
        # Start with study-wide metrics
        metrics = MetricsConfig(**self.metrics.__dict__)
        
        # Apply experiment-level metrics
        if experiment.metrics:
            metrics.additional_metrics.extend(experiment.metrics.additional_metrics)
        
        # Apply scenario-level metrics
        scenario = experiment.scenarios[scenario_name]
        if scenario.metrics:
            metrics.additional_metrics.extend(scenario.metrics.additional_metrics)
        
        return metrics

    def get_scenario_config(self, experiment_name: str, scenario_name: str) -> ScenarioConfig:
        """Get complete configuration for a specific scenario"""
        if experiment_name not in self.experiments:
            raise KeyError(f"Experiment '{experiment_name}' not found")
            
        experiment = self.experiments[experiment_name]
        if scenario_name not in experiment.scenarios:
            raise KeyError(f"Scenario '{scenario_name}' not found in experiment '{experiment_name}'")
            
        scenario = experiment.scenarios[scenario_name]
        return scenario.merge_with_base(self.base_config)

    def save(self, path: Optional[Path] = None) -> Path:
        """Save study configuration to file"""
        import yaml
        
        def _convert_to_dict(obj: Any) -> Any:
            """Recursively convert dataclass instances to dictionaries"""
            if isinstance(obj, Enum):
                return obj.value
            
            if hasattr(obj, '__dataclass_fields__'):
                return {k: _convert_to_dict(v) for k, v in obj.__dict__.items()}
            
            if isinstance(obj, (list, tuple)):
                return [_convert_to_dict(item) for item in obj]
            
            if isinstance(obj, dict):
                return {k: _convert_to_dict(v) for k, v in obj.items()}
            
            return obj
        
        if path is None:
            path = self.paths.get_study_dir(self.metadata.name) / "study_config.yaml"
        
        # Convert the StudyConfig instance to a dictionary
        config_dict = _convert_to_dict(self)
        
        with open(path, 'w') as f:
            yaml.dump(config_dict, f, default_flow_style=False)
        
        return path

    @classmethod
    def load(cls, path: Path) -> 'StudyConfig':
        """Load study configuration from file with proper dataclass instantiation"""
        import yaml
        from typing import get_type_hints, get_args, get_origin
        from copy import deepcopy
        
        def _convert_to_dataclass(data: Dict, target_class: Any) -> Any:
            """Recursively convert dictionaries to dataclass instances"""
            if data is None:
                return None
                
            if not isinstance(data, dict):
                return data
                
            hints = get_type_hints(target_class)
            kwargs = {}
            
            for key, value in data.items():
                if key not in hints:
                    # For fields not in type hints (like base_config contents),
                    # we want to preserve the complete dictionary structure
                    kwargs[key] = deepcopy(value)
                    continue
                    
                target_type = hints[key]
                
                # Handle Optional types
                if get_origin(target_type) is Union and type(None) in get_args(target_type):
                    target_type = get_args(target_type)[0]
                
                # Handle Enums
                if isinstance(value, str) and isinstance(target_type, type) and issubclass(target_type, Enum):
                    kwargs[key] = target_type(value)
                    continue
                
                # Special handling for base_config field
                if key == 'base_config':
                    kwargs[key] = deepcopy(value)
                    continue
                
                # Handle Lists
                if get_origin(target_type) is list:
                    element_type = get_args(target_type)[0]
                    if hasattr(element_type, '__dataclass_fields__'):
                        kwargs[key] = [_convert_to_dataclass(item, element_type) for item in value]
                    else:
                        kwargs[key] = value
                    continue
                
                # Handle Dicts
                if get_origin(target_type) is dict:
                    value_type = get_args(target_type)[1]
                    if hasattr(value_type, '__dataclass_fields__'):
                        kwargs[key] = {k: _convert_to_dataclass(v, value_type) for k, v in value.items()}
                    else:
                        kwargs[key] = value
                    continue
                
                # Handle nested dataclasses
                if hasattr(target_type, '__dataclass_fields__'):
                    kwargs[key] = _convert_to_dataclass(value, target_type)
                else:
                    kwargs[key] = value
            
            return target_class(**kwargs)

        try:
            with open(path) as f:
                data = yaml.safe_load(f)
                
            # Ensure base_config exists and is properly copied
            if 'base_config' in data:
                data['base_config'] = deepcopy(data['base_config'])
                
            # Convert the loaded data into a StudyConfig instance
            return _convert_to_dataclass(data, cls)
        except Exception as e:
            raise ValueError(f"Failed to load study configuration: {str(e)}") from e

@dataclass
class GlobalOptimizationConfig(DataclassYAMLMixin):
    """Configuration for global system optimization"""
    optimization_interval: int = 300  # How often to run optimization (seconds)
    max_optimization_time: int = 60   # Maximum time allowed for optimization (seconds)
    improvement_threshold: float = 0.05  # Minimum improvement required to apply changes (5%)
    max_iterations: int = 100  # Maximum number of iterations for optimization
    convergence_threshold: float = 0.01  # Threshold for convergence (1%)
    
    # Weights for different optimization objectives
    weights: Dict[str, float] = field(default_factory=lambda: {
        "waiting_time": 0.3,
        "vehicle_utilization": 0.3,
        "total_distance": 0.2,
        "load_balancing": 0.2
    })
    
    # Constraints for optimization
    constraints: Dict[str, Any] = field(default_factory=lambda: {
        "max_reassignments": 5,  # Maximum number of requests to reassign per vehicle
        "max_route_changes": 3,  # Maximum number of route changes per vehicle
        "min_improvement_per_change": 0.02  # Minimum improvement required per change (2%)
    })
    
    # Time windows for looking ahead/behind
    time_windows: Dict[str, int] = field(default_factory=lambda: {
        "look_ahead": 1800,  # Look ahead window (30 minutes)
        "look_behind": 900   # Look behind window (15 minutes)
    })