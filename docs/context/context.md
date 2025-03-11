# DRT Research Platform Implementation Context

## System Overview
Implementing a Demand Responsive Transportation (DRT) research simulation platform for studying various dispatch strategies, routing algorithms, and service configurations.

## Execution Hierarchy

### Study Management
The platform implements a hierarchical execution structure:

1. **Study Runner (`run_simulation.py`)**
   - Top-level coordinator for simulation studies
   - Manages MLflow experiment tracking and configuration
   - Features:
     - MLflow integration for experiment tracking
     - Parameter set management
     - Study-level configuration management
     - Parallel or sequential execution support
     - Comprehensive logging and result collection
     - Study-wide artifact management

2. **Simulation Runner (`SimulationRunner`)**
   - Manages individual simulation runs and replications
   - Handles MLflow run tracking and metrics collection
   - Features:
     - Nested MLflow run management
     - Replication execution control
     - Result aggregation and analysis
     - Metric collection and persistence
     - Artifact management and logging
     - Component initialization and cleanup

3. **Simulation Orchestrator (`SimulationOrchestrator`)**
   - Core simulation coordinator 
   - Manages simulation components and flow
   - Handles event processing and state management
   - Features:
     - Component initialization and cleanup
     - Transaction management
     - Event coordination
     - State persistence

### Execution Flow
1. **Study Initialization**
   - Load study configuration
   - Set up MLflow experiment
   - Configure logging and output directories
   - Initialize parameter sets
   - Set up artifact locations

2. **Parameter Set Execution**
   - Create simulation runners for each parameter set
   - Set up nested MLflow runs
   - Configure replication parameters
   - Manage parallel or sequential execution
   - Collect and aggregate results

3. **Replication Processing**
   - Initialize simulation components
   - Set up metrics collection
   - Execute simulation steps
   - Track and log metrics
   - Manage state and cleanup

4. **Simulation Execution**
   - Component initialization
   - Event processing
   - State management
   - Result collection
   - Resource cleanup

## Core Architecture

### Simulation Flow
The simulation follows an event-driven architecture with the following key components working together:

1. **Simulation Orchestrator**
   - High-level coordinator managing interactions between subsystems
   - Initializes and manages core components (state, events, demand, network)
   - Handles simulation lifecycle (initialization, execution, cleanup)
   - Coordinates between handlers and managers

2. **Simulation Engine**
   - Manages time progression and event processing
   - Executes simulation steps with transaction support
   - Processes events based on priority and timestamp
   - Tracks performance metrics and simulation progress
   - Handles warm-up period and completion conditions

3. **Simulation Context**
   - Maintains simulation time and status
   - Manages time step progression
   - Handles warm-up period tracking
   - Provides simulation progress monitoring

### Event System
- Event-driven architecture using `EventManager`
- Event priority levels (CRITICAL, HIGH, NORMAL)
- Event validation and error handling
- Transaction support for atomic operations
- Event types for different simulation aspects:
  - Request events (received, validated, assigned, rejected)
  - Vehicle events (dispatch, arrival, route updates)
  - Passenger events (journey start, boarding, alighting)
  - Stop events (activation, deactivation, capacity)
  - System events (errors, optimization, metrics)

### State Management
- Robust state management through `StateManager`
- Multiple specialized workers for different entities:
  - RequestStateWorker: Request lifecycle and status
  - VehicleStateWorker: Vehicle states and assignments
  - PassengerStateWorker: Passenger journey tracking
  - RouteStateWorker: Route planning and execution
  - StopStateWorker: Stop management and metrics
  - StopAssignmentWorker: Stop allocation
- Transaction support with commit/rollback
- State snapshots and history tracking

### Event Handlers
1. **Request Handler**
   - Processes new transportation requests
   - Validates request parameters
   - Initiates virtual stop determination
   - Manages request lifecycle

2. **Vehicle Handler**
   - Manages vehicle dispatching
   - Handles vehicle movements
   - Processes route updates
   - Manages vehicle status changes

3. **Passenger Handler**
   - Manages passenger journey events
   - Handles boarding and alighting
   - Tracks service level violations
   - Manages passenger states

4. **Stop Handler**
   - Determines virtual stops
   - Manages stop activation/deactivation
   - Handles passenger arrivals/departures
   - Monitors stop capacity

5. **Matching Handler**
   - Matches requests to vehicles
   - Handles dispatch optimization
   - Processes assignments
   - Manages matching failures

### Service Components

#### Demand Management
- Multiple demand generation strategies:
  - RandomDemandGenerator: Random demand patterns
  - CSVDemandGenerator: Historical demand data
- Demand validation and processing
- Request tracking and metrics

#### Route Management
- Route planning and optimization
- Stop sequencing
- Route metrics calculation
- Constraint validation
- Route status tracking
- Deviation handling

#### Network Management
- Network topology management
- Distance calculations
- Walking time estimation
- Route optimization support

### Monitoring & Metrics
- Comprehensive metrics collection
- Service level monitoring
- Resource utilization tracking
- Performance metrics
- Custom metric extractors
- Metrics persistence

## Technical Details

### Error Handling
- Comprehensive error tracking
- Transaction rollback support
- Error event propagation
- Validation systems
- Recovery mechanisms

### State Tracking
- Historical state tracking
- State snapshots
- Transaction support
- State restoration

### Configuration
- Scenario configuration
- Simulation parameters
- Service thresholds
- Network settings
- Vehicle parameters

### Configuration Structure
- Hierarchical configuration system:
  - StudyConfig: Top-level study configuration
    - Metadata (name, version, authors, tags)
    - Study type (standard/parameter sweep)
    - Experiment configurations
    - Global parameters
  - ExperimentConfig: Experiment-level settings
    - Scenario definitions
    - Metrics configuration
    - Execution settings
  - ScenarioConfig: Scenario-specific parameters
    - Vehicle configuration
    - Demand settings
    - Network parameters
    - Service thresholds
  - SimulationConfig: Simulation runtime settings
    - Time parameters
    - State persistence settings
    - Logging configuration
    - Resource limits

### Data Models
1. **Core Models**
   - Event: Base class for all simulation events
   - SimulationState: Current system state representation
   - Route: Vehicle route with stops and segments
   - Request: Passenger transportation request
   - Vehicle: Vehicle state and properties
   - Stop: Physical or virtual stop point
   - Assignment: Request-to-vehicle assignment

2. **State Models**
   - PassengerState: Passenger journey tracking
   - VehicleState: Vehicle operational state
   - RouteState: Route execution state
   - RequestState: Request processing state
   - StopState: Stop utilization state

3. **Metric Models**
   - ServiceMetrics: Service quality indicators
   - OperationalMetrics: System performance metrics
   - ResourceMetrics: Resource utilization metrics
   - TemporalMetrics: Time-based performance metrics

### System Validation
1. **Configuration Validation**
   - Schema validation for all configuration levels
   - Parameter range and constraint checking
   - Dependency validation between components
   - Configuration version compatibility

2. **Runtime Validation**
   - State consistency checks
   - Transaction validation
   - Event sequence validation
   - Resource limit monitoring

3. **Result Validation**
   - Metric boundary validation
   - Statistical significance checks
   - Data consistency verification
   - Output format validation

### File Organization
- Configuration templates in YAML format
- Standardized directory structure for:
  - Study artifacts
  - Experiment results
  - Metrics data
  - Log files
  - State snapshots
  - Event history

### Integration Points
1. **External Systems**
   - MLflow for experiment tracking
   - Ray for distributed execution
   - Network data providers
   - Visualization tools

2. **Data Exchange**
   - JSON serialization for state/events
   - CSV export for metrics
   - YAML for configuration
   - Pickle for large state objects

3. **Extension Points**
   - Custom metric extractors
   - Demand generation plugins
   - Matching algorithm interfaces
   - Network provider adapters

## Directory Structure
```
- data/
    - candidate_stops/
        - hwaseong.csv
        - hwaseong_candidate_virtual_stops.csv
        - knut.csv
    - demands/
        - S1_real_time.csv
        - S2_real_time.csv
        - S3_real_time.csv
        - base_real_time.csv
        - cleaned_knut_passenger_requests.csv
        - knut_passenger_requests.csv
        - knut_weekend_data_1.csv
        - knut_weekend_data_2.csv
        - knut_weekend_data_3.csv
        - preprocessed_base_real_time.csv
        - valid_requests_S1_real_time.csv
        - valid_requests_S2_real_time.csv
        - valid_requests_S3_real_time.csv
        - valid_requests_base_real_time.csv
    - networks/
        - hwaseong_drive.edg.xml
        - hwaseong_drive.graphml
        - hwaseong_drive.net.xml
        - hwaseong_drive.nod.xml
        - hwaseong_walk.edg.xml
        - hwaseong_walk.graphml
        - hwaseong_walk.net.xml
        - hwaseong_walk.nod.xml
        - knut_drive.graphml
        - knut_walk.graphml
        - nyc_network.geojson
- drt_sim/
    - User_preference_weight_calculation/
        - RL.py
        - RL_documented.py
        - features.csv
        - user_history.csv
        - weights.csv
    - __init__.py
    - algorithms/
        - __init__.py
        - base_interfaces/
            - __init__.py
            - demand_predictor_base.py
            - matching_base.py
            - routing_base.py
            - stop_assigner_base.py
            - stop_selector_base.py
            - user_acceptance_base.py
        - matching/
            - __init__.py
            - assignment/
                - auction.py
                - insertion.py
            - batch_matching.py
            - exact_optimizer.py
            - heuristic_matching.py
        - optimization/
            - global_optimizer.py
        - routing/
            - __init__.py
            - dijkstra_routing.py
            - genetic_routing.py
            - time_dependent_routing.py
        - stop/
            - assigner/
                - accessibility.py
                - multi_objective.py
                - nearest.py
            - selector/
                - coverage_based.py
                - demand_based.py
        - user_acceptance_models/ #to be implemented
            - __init__.py
    - config/
        - __init__.py
        - config.py
    - core/
        - __init__.py
        - coordination/
        - demand/
            - __init__.py
            - generators.py
            - manager.py
            - user_acceptance.py
            - user_profiles.py
        - events/
            - manager.py
        - logging_config.py
        - monitoring/
            - metrics/
                - aggregator.py
                - collector.py
                - manager.py
                - metrics.yaml
                - mlflow_adapter.py
                - registry.py
                - storage.py
            - types/
                - metrics.py
            - visualization/
                - manager.py
        - paths.py
        - processors/
        - services/
            - route_service.py
        - simulation/
            - context.py
            - engine.py
            - orchestrator.py
        - simulation_context.py
        - state/
            - base.py
            - manager.py
            - workers/
                - __init__.py
                - assignment_state_worker.py
                - passenger_state_worker.py
                - request_state_worker.py
                - route_state_worker.py
                - stop_assignment_state_worker.py
                - stop_state_worker.py
                - vehicle_state_worker.py
        - user/
            - manager.py
    - handlers/
        - __init__.py
        - matching_handler.py
        - passenger_handler.py
        - request_handler.py
        - route_handler.py
        - stop_handler.py
        - vehicle_handler.py
    - integration/
        - __init__.py
        - external_solvers.py
        - traffic_sim_integration.py
    - models/
        - base.py
        - event.py
        - location.py
        - matching/
            - __init__.py
            - enums.py
            - types.py
        - metrics.py
        - passenger.py
        - rejection.py
        - request.py
        - route.py
        - simulation.py
        - state/
            - __init__.py
            - assignment_system_state.py
            - passenger_system_state.py
            - request_system_state.py
            - route_system_state.py
            - simulation_state.py
            - status.py
            - stop_system_state.py
            - vehicle_system_state.py
        - stop.py
        - user.py
        - vehicle.py
    - network/
        - __init__.py
        - manager.py
    - runners/
        - __init__.py
        - simulation_runner.py
    - utils/
        - __init__.py
        - caching.py
        - geometry.py
- requirements.txt
- setup.py
- studies/
    - artifacts/
        - DRT Fleet Optimization/
    - configs/
        - dispatch_comparison.yaml
        - fleet_optimization.yaml
        - stop_assigner_test.yaml
        - sumo_integration_example.yaml
        - test.yaml
        - user_behavior.yaml
    - mlflow.db

```
