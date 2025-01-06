# DRT Research Platform Implementation Context

## System Overview
Implementing a Demand Responsive Transportation (DRT) research simulation platform for studying various dispatch strategies, routing algorithms, and service configurations.

## Core Components Implemented

### 1. Orchestration Layer
- **SimulationOrchestrator**: Central coordinator managing simulation lifecycle
  - Initialization of all components
  - Step-by-step simulation execution
  - Resource management and cleanup
  - Event coordination

- **SimulationEngine**: Core simulation engine
  - Time progression management
  - Event scheduling and processing
  - State transitions
  - Performance tracking

### 2. State Management
- **StateManager**: Coordinates multiple state workers
  - Transaction management
  - State snapshots
  - State persistence
  - Metrics collection

#### State Workers:
- **VehicleStateWorker**: Manages vehicle states
- **RequestStateWorker**: Handles request states
- **RouteStateWorker**: Manages route states
- **PassengerStateWorker**: Tracks passenger journey states
  - Complete journey tracking
  - Timing metrics
  - Service quality monitoring
  - State transitions

### 3. Event System
- **Event Types**: Comprehensive enumeration covering:
  - System events
  - Request lifecycle
  - Vehicle operations
  - Passenger journey
  - Route management
  - Service quality
  - Metrics collection

### 4. Event Handlers
- **BaseHandler**: Foundation for all handlers with common functionality
- **RequestHandler**: Manages request lifecycle
- **VehicleHandler**: Handles vehicle operations
- **PassengerHandler**: Tracks passenger journeys
- **RouteHandler**: Manages route operations
- **StopHandler**: Handles pickup/dropoff locations

### 5. Models
- **Event**: Core event model with priority and status
- **PassengerState**: Complete passenger journey state
- **Route**: Route planning and execution
- **Location**: Geographical positioning
- **Request**: Service request representation
- **Vehicle**: Vehicle state and capabilities
- **Stop**: Pickup/dropoff location management

## Status Tracking
Each component maintains comprehensive status enumerations:
- Request Status (Created → Assigned → Completed)
- Vehicle Status (Idle → Assigned → Moving)
- Passenger Status (Walking → Waiting → In-Vehicle)
- Route Status (Created → Active → Completed)

## Key Features Implemented
1. **Event-Driven Architecture**
   - Priority-based event processing
   - Event validation and history
   - Error handling and recovery

2. **State Management**
   - Transactional state updates
   - Historical state tracking
   - State persistence

3. **Metrics Collection**
   - Service quality metrics
   - Vehicle utilization
   - Passenger experience
   - System performance

4. **Location Handling**
   - Stop management
   - Route planning
   - Distance calculations

## Next Steps
1. **Implementation Needed**:
   - Dispatch algorithms
   - Routing optimization
   - Demand prediction
   - Schedule optimization
   - Network management

2. **Integration Points**:
   - External routing services
   - Traffic simulation
   - Demand modeling
   - Weather effects

3. **Research Features**:
   - Parameter sweeps
   - Scenario comparison
   - Performance analysis
   - Service quality assessment

## Notes
- The system uses Python dataclasses for model definitions
- Logging is implemented throughout all components
- Error handling follows a consistent pattern
- State persistence uses JSON serialization
- Event handling is modular and extensible

## Configuration
The system uses hierarchical configuration with:
- ScenarioConfig: Overall simulation parameters
- SimulationConfig: Technical simulation settings
- VehicleConfig: Vehicle-specific parameters
- RequestConfig: Request handling parameters
- RouteConfig: Routing parameters

## Directory Structure
Current implementation follows the structure outlined in the blueprint, with core components in corresponding directories:
- `drt_sim/core/`: Core system components
- `drt_sim/handlers/`: Event handlers
- `drt_sim/models/`: Data models
- `drt_sim/algorithms/`: Algorithm implementations
- `drt_sim/config/`: Configuration management