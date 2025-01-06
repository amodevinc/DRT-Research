# DRT Research Platform - Experiment System Documentation

## Table of Contents
1. [Overview](#overview)
2. [Directory Structure](#directory-structure)
3. [Configuration System](#configuration-system)
4. [Running Experiments](#running-experiments)
5. [Creating New Experiments](#creating-new-experiments)
6. [Parameter Sweeps](#parameter-sweeps)
7. [Results and Outputs](#results-and-outputs)
8. [Best Practices](#best-practices)
9. [Troubleshooting](#troubleshooting)

## Overview

The DRT Research Platform uses Hydra for configuration management, MLflow for experiment tracking, and Ray for distributed execution. The system is designed to:
- Manage different experiment configurations
- Handle parameter sweeps
- Track results and metrics
- Organize outputs by experiment
- Support distributed execution

## Directory Structure

```
drt_sim/
├── experiments/
│   ├── configs/
│   │   ├── config.yaml             # Main configuration
│   │   └── experiment/             # Experiment configurations
│   │       ├── base.yaml           # Base experiment config
│   │       ├── fleet_demand_study.yaml
│   │       └── other_studies.yaml
│   └── results/
│       └── {experiment_name}/      # Results organized by experiment
│           ├── runs/               # Individual run results
│           │   └── YYYY-MM-DD_HH-MM-SS/
│           ├── sweeps/             # Parameter sweep results
│           ├── mlflow/             # MLflow tracking data
│           └── metrics/            # Collected metrics
```

## Configuration System

### Main Components

1. **Base Configuration (config.yaml)**
   - Defines global settings
   - Sets up output directories
   - Configures MLflow and Ray
   ```yaml
   defaults:
     - _self_
     - experiment: base
   
   hydra:
     run:
       dir: experiments/results/${experiment.name}/runs/${now:%Y-%m-%d_%H-%M-%S}
   ```

2. **Base Experiment (experiment/base.yaml)**
   - Provides default settings for all experiments
   - Defines common parameters
   ```yaml
   experiment:
     name: "base"
     description: "Base configuration"
     variant: "default"
   ```

3. **Specific Experiments**
   - Inherit from base
   - Override specific settings
   - Define parameter sweeps

### Configuration Sections

1. **Experiment Settings**
   ```yaml
   experiment:
     name: "study_name"
     description: "Study description"
     variant: "variant_name"
     replications: 5
     random_seed: 42
     duration: 86400  # 24 hours
   ```

2. **Network Configuration**
   ```yaml
   network:
     network_file: "data/networks/network.graphml"
     walk_network_file: "data/networks/walk.graphml"
     coordinate_system: "EPSG:4326"
   ```

3. **Vehicle Settings**
   ```yaml
   vehicle:
     fleet_size: 10
     capacity: 4
     speed: 10.0
   ```

## Running Experiments

### Basic Execution
```bash
# Run with default configuration
python run_study.py

# Run specific experiment
python run_study.py experiment=fleet_demand_study

# Override parameters
python run_study.py experiment=fleet_demand_study vehicle.fleet_size=20
```

### Parameter Sweeps
```bash
# Run with multirun
python run_study.py -m experiment=fleet_demand_study vehicle.fleet_size=5,10,15

# Run with configured sweep
python run_study.py experiment=fleet_demand_study parameter_sweep.enabled=true
```

## Creating New Experiments

1. Create new configuration file:
```yaml
# drt_sim/experiments/configs/experiment/new_study.yaml
defaults:
  - base
  - _self_

experiment:
  name: "new_study"
  description: "Description of new study"
  variant: "baseline"
```

2. Define parameters and sweeps:
```yaml
parameter_sweep:
  enabled: true
  method: "grid"
  parameters:
    vehicle.fleet_size: [5, 10, 15]
    algorithm.dispatch_strategy: ["fcfs", "batch"]
```

## Results and Outputs

### Directory Organization
- Each experiment gets its own directory
- Results organized by run date/time
- MLflow tracking data stored separately
- Metrics collected in structured format

### Accessing Results
1. **MLflow UI**
   ```bash
   mlflow ui --backend-store-uri experiments/results/study_name/mlflow
   ```

2. **Direct Access**
   - Results in `experiments/results/{study_name}/runs/`
   - Metrics in `experiments/results/{study_name}/metrics/`
   - Configuration files preserved with results

## Best Practices

1. **Configuration Management**
   - Use clear, descriptive names
   - Document parameters in config files
   - Keep base configuration minimal
   - Use parameter sweeps for systematic studies

2. **Experiment Organization**
   - One experiment per scientific question
   - Clear naming conventions
   - Document study objectives in description
   - Use variants for related investigations

3. **Results Management**
   - Review MLflow logs regularly
   - Back up important results
   - Document significant findings
   - Keep configuration with results

## Troubleshooting

### Common Issues

1. **Configuration Not Found**
   ```
   Could not find 'experiment/base'
   ```
   - Check config directory structure
   - Verify file names match config
   - Ensure defaults are properly set

2. **Interpolation Errors**
   ```
   InterpolationKeyError: Interpolation key 'X' not found
   ```
   - Check variable references in configs
   - Ensure required values are defined
   - Verify configuration load order

3. **MLflow Issues**
   - Check tracking URI is correct
   - Verify directory permissions
   - Ensure MLflow is properly initialized

### Solutions

1. **Configuration Problems**
   - Use `--info` flag for more details
   - Check Hydra debug logs
   - Verify YAML syntax

2. **Execution Issues**
   - Enable debug logging
   - Check resource availability
   - Verify input data exists

3. **Result Issues**
   - Check directory permissions
   - Verify disk space
   - Ensure clean experiment state

## Example Workflows

1. **Simple Study**
   ```bash
   python run_study.py experiment=my_study
   ```

2. **Parameter Sweep**
   ```bash
   python run_study.py -m \
     experiment=my_study \
     vehicle.fleet_size=5,10,15 \
     algorithm.dispatch_strategy=fcfs,batch
   ```

3. **Distributed Execution**
   ```bash
   python run_study.py \
     experiment=my_study \
     execution.distributed=true
   ```

Remember to:
- Test configurations before large studies
- Monitor system resources
- Back up important results
- Document study parameters
- Track experiment outcomes

For additional help or feature requests, please refer to the project documentation or open an issue in the repository.
