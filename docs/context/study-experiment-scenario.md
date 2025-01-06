# DRT Research Platform Development Context

## Current System Structure

The platform follows a hierarchical structure for running simulations:

```
Study (StudyRunner)
└── Experiments (ExperimentRunner)
    └── Scenarios (ScenarioRunner)
        └── Simulations
```

## Key Configuration Classes

All configuration classes are defined in `config/config.py`:

```python
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional, Dict, Any, Union, Tuple
from datetime import datetime
from pathlib import Path

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
class StudyMetadata:
    """Metadata about the study"""
    name: str = ""
    description: str = ""
    tags: List[str] = field(default_factory=list)
    authors: List[str] = field(default_factory=list)
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    version: str = "1.0.0"

@dataclass
class StudyPaths:
    """Paths configuration for study outputs"""
    base_dir: str = "studies"
    results_dir: str = "results"
    configs_dir: str = "configs"
    logs_dir: str = "logs"
    artifacts_dir: str = "artifacts"

    def get_study_dir(self, study_name: str) -> Path:
        return Path(self.base_dir) / study_name

    def create_study_dirs(self, study_name: str) -> Dict[str, Path]:
        study_dir = self.get_study_dir(study_name)
        dirs = {
            "base": study_dir,
            "results": study_dir / self.results_dir,
            "configs": study_dir / self.configs_dir,
            "logs": study_dir / self.logs_dir,
            "artifacts": study_dir / self.artifacts_dir
        }
        for dir_path in dirs.values():
            dir_path.mkdir(parents=True, exist_ok=True)
        return dirs

@dataclass
class MetricDefinition:
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
class MetricsConfig:
    """Configuration for metrics collection and analysis"""
    collect_interval: int = 300
    save_interval: int = 1800
    batch_size: int = 1000
    storage_format: str = "parquet"
    compression: Optional[str] = "snappy"
    
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

@dataclass
class MLflowConfig:
    tracking_uri: str = "sqlite:///experiments/mlflow.db"
    experiment_name: str = ""
    tags: Dict[str, str] = field(default_factory=dict)
    log_artifacts: bool = True
    log_params: bool = True

@dataclass
class RayConfig:
    address: str = "auto"
    num_cpus: Optional[int] = None
    num_gpus: Optional[int] = None
    include_dashboard: bool = True
    ignore_reinit_error: bool = True
    log_to_driver: bool = True

@dataclass
class ExecutionConfig:
    distributed: bool = False
    max_parallel: int = 4
    chunk_size: int = 1
    timeout: int = 3600
    retry_attempts: int = 3

@dataclass
class NetworkConfig:
    network_file: str = ""
    walk_network_file: Optional[str] = None
    transfer_points_file: Optional[str] = None
    coordinate_system: str = "EPSG:4326"
    walking_speed: float = 1.4

@dataclass
class CSVFileConfig:
    file_path: str
    weight: float = 1.0
    start_time: Optional[str] = None
    end_time: Optional[str] = None
    region: Optional[str] = None

@dataclass
class CSVDemandGeneratorConfig:
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

@dataclass
class RandomDemandGeneratorConfig:
    demand_level: float = 0.0
    spatial_distribution: str = "uniform"
    temporal_distribution: str = "uniform"
    service_area: Tuple[Tuple[float, float], Tuple[float, float]] = field(
        default_factory=lambda: ((0.0, 0.0), (1.0, 1.0))
    )
    hotspots: Optional[List[Dict[str, Union[Tuple[float, float], float]]]] = None

@dataclass
class DemandConfig:
    generator_type: str = "csv"
    csv_config: CSVDemandGeneratorConfig = field(default_factory=CSVDemandGeneratorConfig)
    random_config: Optional[RandomDemandGeneratorConfig] = None

@dataclass
class VehicleConfig:
    fleet_size: int = 10
    capacity: int = 4
    speed: float = 10.0
    boarding_time: int = 5
    alighting_time: int = 5
    depot_locations: List[Tuple[float, float]] = field(default_factory=lambda: [(37.5666, 127.0000)])
    battery_capacity: Optional[float] = None
    charging_rate: Optional[float] = None

@dataclass
class AlgorithmConfig:
    dispatch_strategy: str = "fcfs"
    matching_algorithm: str = "batch"
    routing_algorithm: str = "dijkstra"
    cost_function: str = "simple"
    user_acceptance_model: str = "logit"
    batch_interval: int = 30
    optimization_horizon: int = 1800
    rebalancing_interval: int = 300

    dispatch_params: Optional[Dict[str, Any]] = None
    matching_params: Optional[Dict[str, Any]] = None
    routing_params: Optional[Dict[str, Any]] = None
    cost_function_params: Optional[Dict[str, Any]] = None
    user_acceptance_params: Optional[Dict[str, Any]] = None
    stop_selection_params: Optional[Dict[str, Any]] = None

    def __post_init__(self):
        """Initialize parameter objects based on selected strategies"""
        if not self.dispatch_params:
            self.dispatch_params = self._get_default_params(self.dispatch_strategy)
        if not self.matching_params:
            self.matching_params = self._get_default_params(self.matching_algorithm)
        if not self.routing_params:
            self.routing_params = self._get_default_params(self.routing_algorithm)
        if not self.cost_function_params:
            self.cost_function_params = self._get_default_params(self.cost_function)
        if not self.user_acceptance_params:
            self.user_acceptance_params = self._get_default_params(self.user_acceptance_model)

    def _get_default_params(self, strategy: str) -> Dict[str, Any]:
        """Get default parameters for a given strategy"""
        defaults = {
            # Dispatch defaults
            "fcfs": {},
            "genetic": {
                "population_size": 100,
                "num_generations": 50,
                "mutation_rate": 0.1,
                "crossover_rate": 0.8
            },
            # Matching defaults
            "batch": {
                "batch_interval": self.batch_interval,
                "max_delay": 300,
                "max_detour": 1.5
            },
            # Routing defaults
            "dijkstra": {},
            "time_dependent": {
                "update_interval": 300,
                "max_alternatives": 3
            },
            # Cost function defaults
            "simple": {
                "weights": {
                    "waiting_time": 0.4,
                    "travel_time": 0.3,
                    "vehicle_distance": 0.2,
                    "occupancy": 0.1
                },
                "constraints": {
                    "max_waiting_time": 1800,
                    "max_travel_time": 1800,
                    "max_vehicle_distance": 1800,
                    "max_occupancy": 1800
                }
            },
            # User acceptance defaults
            "logit": {
                "coefficients": {
                    "waiting_time": -0.01,
                    "travel_time": -0.005,
                    "fare": -0.1
                }
            }
        }
        return defaults.get(strategy, {})

@dataclass
class SimulationConfig:
    start_time: str = "2025-01-01 07:00:00"
    end_time: str = "2025-01-01 19:00:00"
    duration: int = 86400
    warm_up_duration: int = 1800
    time_step: int = 60
    time_scale_factor: float = 1.0
    replications: int = 1
    save_state: bool = True
    save_interval: int = 3600

@dataclass
class ParameterSweepConfig:
    enabled: bool = False
    method: str = "grid"
    parameters: Dict[str, List[Any]] = field(default_factory=dict)
    n_samples: Optional[int] = None
    seed: int = 42

@dataclass
class ScenarioConfig:
    """Configuration for a specific simulation scenario"""
    name: str
    description: str = ""
    demand: DemandConfig = field(default_factory=DemandConfig)
    vehicle: VehicleConfig = field(default_factory=VehicleConfig)
    algorithm: AlgorithmConfig = field(default_factory=AlgorithmConfig)
    network: NetworkConfig = field(default_factory=NetworkConfig)
    metrics: Optional[MetricsConfig] = None

    def merge_with_base(self, base_config: Dict[str, Any]) -> 'ScenarioConfig':
        """Merge this scenario config with base configuration"""
        def deep_merge(d1: Dict[str, Any], d2: Dict[str, Any]) -> Dict[str, Any]:
            result = d1.copy()
            for k, v in d2.items():
                if k in result and isinstance(result[k], dict) and isinstance(v, dict):
                    result[k] = deep_merge(result[k], v)
                else:
                    result[k] = v
            return result
            
        merged_dict = deep_merge(base_config, self.__dict__)
        return ScenarioConfig(**merged_dict)

@dataclass
class ExperimentConfig:
    """Configuration for a specific experiment that may contain multiple scenarios"""
    name: str
    description: str
    scenarios: Dict[str, ScenarioConfig]
    metrics: Optional[MetricsConfig] = None
    variant: str = "default"
    tags: List[str] = field(default_factory=list)
    random_seed: int = 42

@dataclass
class StudyConfig:
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
        self.paths.create_study_dirs(self.metadata.name)
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
                    kwargs[key] = value
                    continue
                    
                target_type = hints[key]
                
                # Handle Optional types
                if get_origin(target_type) is Union and type(None) in get_args(target_type):
                    target_type = get_args(target_type)[0]
                
                # Handle Enums
                if isinstance(value, str) and isinstance(target_type, type) and issubclass(target_type, Enum):
                    kwargs[key] = target_type(value)
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

        with open(path) as f:
            data = yaml.safe_load(f)
            
        # Convert the loaded data into a StudyConfig instance
        return _convert_to_dataclass(data, cls)
```

## Recent Changes & Issues Addressed

1. MLflow Integration:
   - Added proper nested runs
   - Fixed run cleanup
   - Ensured runs are properly ended

2. Configuration Handling:
   - Removed assumptions about optional attributes
   - Added proper type checking
   - Better handling of distributed execution settings

3. Error Handling:
   - Added comprehensive try/except blocks
   - Better cleanup handling
   - Improved logging

## Key Files

```
drt_research_platform/
├── run_study.py                    # Main entry point
├── drt_sim/
│   ├── runners/
│   │   ├── study_runner.py        # Manages overall study execution
│   │   ├── experiment_runner.py   # Handles experiment execution
│   │   └── scenario_runner.py     # Runs individual scenarios
│   ├── config/
│   │   └── config.py             # Configuration classes
│   └── studies/
│       └── configs/              # Study configuration YAML files
└── requirements.txt
```

## Current Work

Currently focused on ensuring proper:
1. Configuration handling across all levels
2. MLflow integration for experiment tracking
3. Error handling and cleanup
4. Execution mode handling (distributed vs sequential)

## Next Steps

1. Test with different study types:
   - Parameter sweeps
   - Scenario comparisons
   - Sensitivity analyses
   - Validation studies

2. Verify proper handling of:
   - Metrics collection
   - Result saving
   - State management
   - Resource cleanup

## Example Usage

```python
# Main entry point (run_study.py)
def main():
    # Load and validate study configuration
    study_config = load_study_config(args.study_name)
    
    # Create and run study
    runner = StudyRunner(study_config)
    try:
        runner.setup()
        results = runner.run()
        return results
    finally:
        runner.cleanup()
```
