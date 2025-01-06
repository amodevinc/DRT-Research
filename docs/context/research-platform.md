# DRT Research Platform Documentation

## Table of Contents
1. [Overview](#overview)
2. [System Architecture](#system-architecture)
3. [Configuration System](#configuration-system)
4. [Execution Hierarchy](#execution-hierarchy)
5. [Running Studies](#running-studies)
6. [Metrics and Analysis](#metrics-and-analysis)
7. [MLflow Integration](#mlflow-integration)
8. [Distributed Execution](#distributed-execution)
9. [Error Handling and Recovery](#error-handling-and-recovery)
10. [Best Practices](#best-practices)

## Overview

The DRT (Demand Responsive Transportation) Research Platform is a comprehensive simulation framework designed for conducting research experiments in transportation systems. The platform supports various dispatch strategies, demand patterns, and vehicle configurations while providing robust tools for experiment management and analysis.

### Key Features
- Hierarchical study management
- Parameter sweep capabilities
- Distributed execution support
- Comprehensive metrics collection
- MLflow experiment tracking
- Scenario comparison tools
- State management and persistence

## System Architecture

### Core Components

```
Study (StudyRunner)
└── Experiments (ExperimentRunner)
    └── Scenarios (ScenarioRunner)
        └── Simulations
```

Each level of the hierarchy serves a specific purpose:

1. **StudyRunner**: Top-level coordinator
   - Manages overall study execution
   - Handles experiment organization
   - Coordinates MLflow tracking
   - Manages distributed execution

2. **ExperimentRunner**: Mid-level executor
   - Runs multiple scenarios
   - Handles replications
   - Collects experiment-level metrics
   - Manages experiment state

3. **ScenarioRunner**: Low-level executor
   - Executes individual simulations
   - Collects detailed metrics
   - Manages simulation state
   - Handles core simulation logic

## Configuration System

The platform uses a hierarchical configuration system based on Python dataclasses.

### Configuration Hierarchy

```python
StudyConfig
├── StudyMetadata
├── StudyPaths
├── ExperimentConfig
│   ├── ScenarioConfig
│   │   ├── DemandConfig
│   │   ├── VehicleConfig
│   │   ├── AlgorithmConfig
│   │   └── NetworkConfig
│   └── MetricsConfig
└── ExecutionConfig
```

### Key Configuration Classes

#### StudyConfig
```python
@dataclass
class StudyConfig:
    metadata: StudyMetadata
    type: StudyType
    paths: StudyPaths
    experiments: Dict[str, ExperimentConfig]
    metrics: MetricsConfig
    execution: ExecutionConfig
```

#### ExperimentConfig
```python
@dataclass
class ExperimentConfig:
    name: str
    description: str
    scenarios: Dict[str, ScenarioConfig]
    metrics: Optional[MetricsConfig]
    variant: str
    random_seed: int
```

## Running Studies

### Command Line Interface

The platform provides a command-line interface through `run_study.py`:

```bash
# List available studies
python run_study.py --list

# Run a specific study
python run_study.py <study_name>
```

### Study Configuration File

Studies are defined in YAML configuration files:

```yaml
metadata:
  name: "fleet_size_study"
  description: "Investigating impact of fleet size"
  version: "1.0.0"
  
type: "parameter_sweep"

parameter_sweep:
  enabled: true
  method: "grid"
  parameters:
    vehicle.fleet_size: [10, 20, 30, 40]
    
experiments:
  base_experiment:
    name: "baseline"
    scenarios:
      default:
        demand:
          generator_type: "csv"
          csv_config:
            file_path: "data/demands/base_real_time.csv"
```

## Metrics and Analysis

### Metric Types
- Temporal: Time-series metrics
- Aggregate: Summary statistics
- Event: Event-based measurements
- Custom: User-defined metrics

### Collection Configuration
```python
@dataclass
class MetricsConfig:
    collect_interval: int = 300  # 5 minutes
    save_interval: int = 1800    # 30 minutes
    batch_size: int = 1000
    storage_format: str = "parquet"
```

## MLflow Integration

The platform integrates with MLflow for experiment tracking:

1. **Run Hierarchy**
   - Study-level parent runs
   - Experiment-level child runs
   - Scenario-level nested runs

2. **Tracked Information**
   - Parameters
   - Metrics
   - Artifacts
   - Tags

## Distributed Execution

### Ray Integration

The platform supports distributed execution using Ray:

```python
@ray.remote
class RayScenarioRunner:
    def __init__(self, scenario_cfg: ScenarioConfig, output_dir: str):
        self.runner = ScenarioRunner(
            cfg=scenario_cfg,
            output_dir=Path(output_dir)
        )
```

### Configuration
```python
@dataclass
class ExecutionConfig:
    distributed: bool = False
    max_parallel: int = 4
    chunk_size: int = 1
    timeout: int = 3600
    retry_attempts: int = 3
```

## Error Handling and Recovery

### State Management
- Regular state snapshots
- Error state preservation
- Cleanup procedures

### Recovery Mechanisms
1. Automatic retry for failed scenarios
2. State restoration capabilities
3. Error logging and diagnostics

## Best Practices

### Study Organization
1. **Clear Naming Conventions**
   - Use descriptive study names
   - Version your configurations
   - Tag experiments appropriately

2. **Configuration Management**
   - Use parameter sweeps for systematic exploration
   - Leverage inheritance for shared configurations
   - Document parameter choices

3. **Resource Management**
   - Monitor memory usage in distributed mode
   - Configure appropriate timeouts
   - Use cleanup handlers

### Recommended Workflow
1. Start with small-scale tests
2. Validate configurations
3. Run parameter sweeps
4. Analyze results using MLflow
5. Export and archive results

### Example Study Structure

```
studies/
├── configs/
│   ├── fleet_size_study.yaml
│   └── demand_patterns_study.yaml
├── results/
│   └── fleet_size_study/
│       ├── metrics/
│       ├── states/
│       └── analysis/
└── logs/
```

## Contributing

When extending the platform:
1. Follow the existing configuration patterns
2. Implement proper cleanup methods
3. Add appropriate logging
4. Update documentation
5. Add test cases

## Future Directions

Potential areas for enhancement:
1. Advanced visualization tools
2. Real-time monitoring capabilities
3. Enhanced statistical analysis tools
4. Additional dispatch strategies
5. Integration with external traffic simulators