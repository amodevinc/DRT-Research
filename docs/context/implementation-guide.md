# DRT Research Platform Implementation Guide

## System Context

The DRT (Demand Responsive Transportation) Research Platform is an event-driven simulation system designed to study various transportation strategies. This guide provides context for implementing the top-level execution components: `run_simulation.py` and `SimulationRunner`.

## Core Components Overview

### Simulation Orchestrator
The system's core is the `SimulationOrchestrator` which:
- Manages the simulation lifecycle
- Coordinates between various handlers and managers
- Processes events through the event system
- Maintains simulation state
- Collects metrics and results

### Event Handlers
The system includes specialized handlers for different aspects:
- `RequestHandler`: Manages transportation requests
- `VehicleHandler`: Controls vehicle operations
- `PassengerHandler`: Handles passenger journeys
- `RouteHandler`: Manages route planning
- `StopHandler`: Controls stop operations
- `MatchingHandler`: Matches requests to vehicles

### State Management
The state system consists of:
- `StateManager`: Central state coordination
- Specialized workers for different entities (vehicles, requests, passengers, etc.)
- Transaction support for state changes
- State persistence capabilities

## Implementation Requirements

### run_simulation.py Requirements

1. **Command Line Interface**
   - Accept configuration file path
   - Support output directory specification
   - Allow parameter set selection
   - Control parallel execution

2. **MLflow Integration**
   - Experiment tracking setup
   - Run management
   - Metric logging
   - Artifact storage

3. **Study Management**
   - Load and validate study configuration
   - Set up directory structure
   - Manage parameter sets
   - Handle parallel/sequential execution
   - Collect and aggregate results

4. **Error Handling**
   - Graceful error management
   - Comprehensive logging
   - State preservation on failure

### SimulationRunner Requirements

1. **Core Functionality**
   - Manage individual simulation runs
   - Handle replications
   - Initialize simulation components
   - Coordinate with MLflow
   - Collect and process results

2. **Configuration Management**
   - Parameter set handling
   - Simulation configuration
   - Directory structure setup
   - Logging configuration

3. **Metrics and Results**
   - Metrics collection
   - Result aggregation
   - Artifact management
   - State snapshots

4. **Resource Management**
   - Component initialization
   - Cleanup procedures
   - Resource tracking
   - Error handling

## Implementation Guidelines

### Configuration Structure
```yaml
study:
  name: "example_study"
  description: "Study description"
  mlflow:
    experiment_name: "drt_experiment"
    tracking_uri: "sqlite:///mlflow.db"
    artifact_location: "./artifacts"
    tags:
      version: "1.0.0"
  parameter_sets:
    set1:
      description: "Parameter set 1"
      replications: 3
      demand:
        type: "random"
        rate: 10
      fleet:
        size: 5
      # Additional parameters...
```

### Expected Directory Structure
```
studies/
├── configs/
│   └── study_config.yaml
├── results/
│   └── study_name/
│       ├── parameter_set_1/
│       │   ├── replication_1/
│       │   ├── replication_2/
│       │   └── metrics/
│       └── aggregated_results/
└── mlruns/
```

### Key Implementation Considerations

1. **MLflow Integration**
   - Use nested runs for parameter sets and replications
   - Maintain proper run hierarchy
   - Handle artifact storage efficiently
   - Track relevant metrics at each level

2. **Parallel Execution**
   - Support both parallel and sequential execution
   - Manage resource allocation
   - Handle synchronization
   - Preserve logging clarity

3. **Error Handling**
   - Implement proper cleanup on failures
   - Preserve partial results
   - Provide detailed error reporting
   - Support recovery mechanisms

4. **Results Management**
   - Aggregate metrics across replications
   - Generate summary statistics
   - Store raw and processed results
   - Maintain result reproducibility

## Example Usage

```python
# Command line usage
python run_simulation.py studies/configs/study_config.yaml \
    --output-dir studies/results \
    --parameter-sets set1 set2 \
    --max-parallel 4

# API usage
runner = SimulationRunner(
    parameter_set=parameter_set,
    sim_cfg=simulation_config,
    output_dir=output_path,
    run_name="example_run"
)
results = await runner.run_replication(1)
```

## Integration Points

1. **State Management**
   ```python
   state_manager = StateManager(config=cfg, sim_cfg=sim_cfg)
   ```

2. **Event System**
   ```python
   event_manager = EventManager()
   ```

3. **Simulation Context**
   ```python
   context = SimulationContext(
       start_time=start_time,
       end_time=end_time,
       time_step=time_step,
       event_manager=event_manager,
       metrics_collector=metrics_collector
   )
   ```

4. **Orchestrator Integration**
   ```python
   orchestrator = SimulationOrchestrator(
       cfg=cfg,
       sim_cfg=sim_cfg,
       output_dir=output_dir,
       metrics_collector=metrics_collector
   )
   ```

## Metrics and Monitoring

1. **Required Metrics**
   - Simulation performance metrics
   - Resource utilization
   - Service quality indicators
   - System state snapshots

2. **Logging Requirements**
   - Component-level logging
   - Performance tracking
   - Error reporting
   - Debug information

3. **Result Collection**
   - Raw event data
   - Aggregated metrics
   - State snapshots
   - Configuration records

## Testing Considerations

1. **Test Scenarios**
   - Single replication execution
   - Multiple parameter sets
   - Parallel execution
   - Error handling
   - Resource cleanup

2. **Validation Points**
   - Configuration loading
   - Directory structure
   - Result aggregation
   - Metric collection
   - MLflow integration

## Performance Considerations

1. **Resource Management**
   - Memory usage monitoring
   - CPU utilization
   - Disk I/O optimization
   - Network usage (if applicable)

2. **Scalability**
   - Parallel execution efficiency
   - Resource allocation
   - Result storage optimization
   - Memory management

## Documentation Requirements

1. **Code Documentation**
   - Clear function documentation
   - Type hints
   - Usage examples
   - Error handling documentation

2. **User Documentation**
   - Configuration guide
   - CLI usage
   - API reference
   - Best practices 