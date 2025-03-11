# DRT Research Platform

## Overview

The Demand Responsive Transportation (DRT) Research Platform is a comprehensive simulation environment for studying various dispatch strategies, routing algorithms, and service configurations in demand-responsive transportation systems. This platform enables researchers and practitioners to evaluate different operational strategies, optimize fleet management, and analyze service performance under various scenarios.

## Table of Contents

- [System Architecture](#system-architecture)
- [Getting Started](#getting-started)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
- [Running Simulations](#running-simulations)
  - [Basic Usage](#basic-usage)
  - [Command-line Options](#command-line-options)
- [Configuration](#configuration)
  - [Study Configuration](#study-configuration)
  - [Parameter Sets](#parameter-sets)
  - [Creating Your Own Experiment](#creating-your-own-experiment)
- [Post-Analysis](#post-analysis)
- [Directory Structure](#directory-structure)
- [Core Components](#core-components)
- [Extending the Platform](#extending-the-platform)

## System Architecture

The DRT Research Platform implements a hierarchical execution structure:

1. **Study Runner** (`run_simulation.py`): Top-level coordinator for simulation studies
   - Manages MLflow experiment tracking and configuration
   - Handles parameter set management and execution
   - Coordinates parallel or sequential execution

2. **Simulation Runner** (`SimulationRunner`): Manages individual simulation runs and replications
   - Handles MLflow run tracking and metrics collection
   - Controls replication execution
   - Aggregates results and performs analysis

3. **Simulation Orchestrator** (`SimulationOrchestrator`): Core simulation coordinator
   - Manages simulation components and flow
   - Handles event processing and state management
   - Coordinates between handlers and managers

The simulation follows an event-driven architecture with the following key components:

- **Event System**: Manages events with different priority levels
- **State Management**: Tracks and updates system state through specialized workers
- **Service Components**: Handles demand, routing, and network management
- **Monitoring & Metrics**: Collects and analyzes performance metrics

## Getting Started

### Prerequisites

- Python 3.8 or higher
- Required Python packages (see `requirements.txt`)
- MLflow for experiment tracking

### Installation

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd drt-research-platform
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Running Simulations

### Basic Usage

The main entry point for running simulations is the `run_simulation.py` script in the `scripts` directory:

```bash
python -m drt_research_platform.scripts.run_simulation <study_name>
```

Where `<study_name>` corresponds to the `name` field in the configuration file in the `studies/configs` directory.

### Command-line Options

The `run_simulation.py` script supports several command-line options:

- `--output-dir`: Directory for simulation outputs (default: `studies/results`)
- `--parameter-sets`, `-p`: Specific parameter sets to run (default: all)
- `--max-parallel`: Maximum number of parallel simulations
- `--parallel`: Run parameter sets in parallel

Running in parallel might be buggy so it's recommended to run without the parallel flag until the system is properly debugged for parallel runs.

Example:
```bash
python -m drt_research_platform.scripts.run_simulation fleet_optimization
```

## Configuration

### Study Configuration

Study configurations are defined in YAML files located in the `studies/configs` directory. Each configuration file defines a complete study with multiple parameter sets.

The configuration structure includes:

- **Study metadata**: name, description, version, authors, tags
- **MLflow configuration**: Tracking URI, experiment name, artifact location
- **Execution settings**: Parallelization, error handling
- **Simulation settings**: Time parameters, random seed
- **Base configuration**: Network, demand, service, and algorithm settings
- **Parameter sets**: Specific configurations to evaluate

### Parameter Sets

Parameter sets define specific configurations to be evaluated within a study. Each parameter set can override any settings from the base configuration and includes:

- **Metadata**: Name, description, tags
- **Replications**: Number of simulation runs with different random seeds
- **Vehicle configuration**: Fleet size, capacity, depot locations
- **Matching configuration**: Assignment method, constraints, weights

### Creating Your Own Experiment

To create your own experiment:

1. Create a new YAML configuration file in `studies/configs/`
2. Define the study metadata, MLflow configuration, and execution settings
3. Configure the base settings for network, demand, service, and algorithms
4. Define parameter sets with specific configurations to evaluate
5. Run the simulation using the `run_simulation.py` script

Example minimal configuration:

```yaml
name: my_experiment
description: "My custom experiment"
version: "1.0.0"
authors:
  - "Your Name"

mlflow:
  tracking_uri: "sqlite:///studies/mlflow.db"
  experiment_name: "My Experiment"
  artifact_location: "studies/artifacts"

execution:
  distributed: false
  max_parallel: 2

simulation:
  start_time: "2024-01-01 07:00:00"
  end_time: "2024-01-01 19:00:00"
  warm_up_duration: 1800
  time_step: 1

base_config:
  # Define base configuration here
  
parameter_sets:
  my_parameter_set:
    name: "my_parameter_set"
    description: "My parameter set"
    replications: 3
    # Override specific parameters here
```

## Post-Analysis

The platform includes a comprehensive post-analysis script (`post_analysis.py`) for analyzing simulation results:

```bash
python -m drt_research_platform.scripts.post_analysis --input-path <path_to_results> --output-dir <output_directory>
```

The post-analysis script generates various visualizations and metrics, including:

- Vehicle performance analysis
- Passenger experience metrics
- Service efficiency indicators
- Request rejection analysis
- Spatial pattern analysis
- System performance metrics

### Result Files

After running a simulation, the result files are stored in parquet format at:
```
studies/results/<study_name>/<date_of_run>/<parameter_set>/archive/
```

These parquet files contain granular metrics logged during the running of the simulation that can be analyzed using the post-analysis script or other data analysis tools. The structured format allows for efficient querying and processing of large simulation datasets.

## Directory Structure

```
drt_research_platform/
├── data/                      # Input data for simulations
│   ├── candidate_stops/       # Candidate stop locations
│   ├── demands/               # Demand data files
│   └── networks/              # Network data files
├── drt_sim/                   # Core simulation package
│   ├── algorithms/            # Implementation of algorithms
│   │   ├── base_interfaces/   # Base interfaces for algorithms
│   │   ├── matching/          # Matching algorithms
│   │   ├── optimization/      # Optimization algorithms
│   │   ├── routing/           # Routing algorithms
│   │   ├── stop/              # Stop selection and assignment
│   │   └── user_acceptance_models/ # User acceptance models
│   ├── config/                # Configuration handling
│   ├── core/                  # Core simulation components
│   │   ├── coordination/      # Coordination mechanisms
│   │   ├── demand/            # Demand generation and management
│   │   ├── events/            # Event system
│   │   ├── monitoring/        # Metrics and monitoring
│   │   ├── simulation/        # Simulation engine
│   │   ├── state/             # State management
│   │   └── user/              # User behavior models
│   ├── handlers/              # Event handlers
│   ├── integration/           # External integrations
│   ├── models/                # Data models
│   │   └── state/             # State models
│   ├── network/               # Network management
│   ├── runners/               # Simulation runners
│   └── utils/                 # Utility functions
├── scripts/                   # Execution and analysis scripts
│   ├── run_simulation.py      # Main simulation runner
│   ├── post_analysis.py       # Post-simulation analysis
│   └── other utility scripts
├── studies/                   # Study configurations and results
│   ├── artifacts/             # MLflow artifacts
│   └── configs/               # Study configuration files
└── requirements.txt           # Package dependencies
```

## Core Components

### Simulation Engine

The simulation engine manages time progression and event processing, executing simulation steps with transaction support and tracking performance metrics.

### Event System

The event-driven architecture uses an `EventManager` to handle events with different priority levels (CRITICAL, HIGH, NORMAL) and provides transaction support for atomic operations.

### State Management

The state management system uses a `StateManager` with specialized workers for different entities (requests, vehicles, passengers, routes, stops) and supports transactions with commit/rollback capabilities.

### Handlers

The platform includes specialized handlers for different aspects of the simulation:

- **Request Handler**: Processes transportation requests
- **Vehicle Handler**: Manages vehicle dispatching and movements
- **Passenger Handler**: Manages passenger journeys
- **Stop Handler**: Determines and manages virtual stops
- **Matching Handler**: Matches requests to vehicles

### Algorithms

The platform supports various algorithms for different aspects of DRT operations:

- **Matching Algorithms**: Insertion, auction-based, batch matching
- **Routing Algorithms**: Dijkstra, time-dependent routing, genetic algorithms
- **Stop Selection**: Coverage-based, demand-based
- **Stop Assignment**: Nearest, multi-objective, accessibility-based

## Extending the Platform

The DRT Research Platform is designed to be extensible. You can:

1. **Add new algorithms**: Implement the appropriate base interface in the `algorithms` package
2. **Create custom metrics**: Extend the metrics system in the `monitoring` package
3. **Implement new demand generators**: Add to the `demand` package
4. **Develop custom event handlers**: Create new handlers in the `handlers` package

