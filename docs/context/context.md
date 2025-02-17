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

6. **Route Handler**
   - Processes route updates
   - Manages route completion
   - Handles route optimization
   - Coordinates vehicle dispatching

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
core/
├── demand/
│   ├── generators.py
│   └── manager.py
├── events/
│   └── manager.py
├── monitoring/
│   ├── metric_extractors.py
│   ├── metrics_collector.py
│   └── resource_monitor.py
├── services/
│   └── route_service.py
├── simulation/
│   ├── context.py
│   ├── engine.py
│   └── orchestrator.py
└── state/
    ├── base.py
    ├── manager.py
    └── workers/
        ├── assignment_state_worker.py
        ├── passenger_state_worker.py
        ├── request_state_worker.py
        ├── route_state_worker.py
        ├── stop_state_worker.py
        └── stop_assignment_state_worker.py
```
