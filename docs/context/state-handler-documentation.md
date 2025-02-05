# DRT System Architecture: Handlers and State Workers

## Handler Overview
The system uses a handler-based architecture to process events and manage different aspects of the DRT simulation. Each handler is responsible for a specific domain and works in conjunction with corresponding state workers.

### Core Handler Components

1. **MatchingHandler**
   - Primary Responsibilities:
     - Matches requests to available vehicles
     - Manages dispatch optimization
     - Handles assignment failures and retries
   - Key State Workers:
     - RequestWorker: Manages request lifecycle
     - VehicleWorker: Tracks vehicle availability
     - StopAssignmentWorker: Handles stop assignments
     - RouteWorker: Manages active routes

2. **PassengerHandler**
   - Primary Responsibilities:
     - Manages passenger journey lifecycle
     - Tracks passenger states and transitions
     - Handles boarding/alighting processes
   - Key State Workers:
     - PassengerWorker: Maintains passenger states
     - StopWorker: Updates stop occupancy
     - MetricsWorker: Records service metrics

3. **RequestHandler**
   - Primary Responsibilities:
     - Processes incoming requests
     - Validates request constraints
     - Manages request state transitions
   - Key State Workers:
     - RequestWorker: Stores request data
     - ValidationWorker: Checks request validity
     - UserProfileWorker: Manages user preferences

4. **RouteHandler**
   - Primary Responsibilities:
     - Manages vehicle routes
     - Handles route updates and optimization
     - Tracks route metrics
   - Key State Workers:
     - RouteWorker: Maintains route information
     - NetworkWorker: Provides routing services
     - MetricsWorker: Records performance data

5. **StopHandler**
   - Primary Responsibilities:
     - Manages stop operations
     - Handles virtual stop creation
     - Controls stop activation/deactivation
   - Key State Workers:
     - StopWorker: Maintains stop states
     - StopAssignmentWorker: Manages assignments
     - NetworkWorker: Handles stop connectivity

6. **VehicleHandler**
   - Primary Responsibilities:
     - Manages vehicle states and operations
     - Handles vehicle movements
     - Tracks vehicle capacity and maintenance
   - Key State Workers:
     - VehicleWorker: Maintains vehicle states
     - RouteWorker: Updates vehicle routes
     - MaintenanceWorker: Tracks service needs

## State Worker Responsibilities

### Core Workers
1. **RequestWorker**
   - Manages request lifecycle
   - Stores request metadata
   - Tracks request status changes
   - Implements request validation

2. **VehicleWorker**
   - Maintains vehicle states
   - Tracks availability and capacity
   - Manages vehicle assignments
   - Records operational metrics

3. **PassengerWorker**
   - Tracks passenger journeys
   - Manages passenger states
   - Records service metrics
   - Handles passenger preferences

4. **RouteWorker**
   - Stores active routes
   - Manages route updates
   - Tracks route metrics
   - Handles route optimization

5. **StopWorker**
   - Maintains stop states
   - Manages stop capacity
   - Handles stop activation
   - Tracks stop utilization

6. **StopAssignmentWorker**
   - Manages stop assignments
   - Tracks assignment validity
   - Handles assignment updates
   - Records assignment metrics

## State Management Principles

1. **Transaction Management**
   - All state changes are wrapped in transactions
   - Rollback support for failed operations
   - Consistent state updates across workers
   - Atomic operations for critical updates

2. **Event-Driven Updates**
   - State changes trigger relevant events
   - Events are processed in priority order
   - Handlers respond to state changes
   - Workers maintain consistency

3. **Concurrency Control**
   - Lock management for shared resources
   - Version control for state updates
   - Conflict resolution mechanisms
   - Deadlock prevention

4. **Error Handling**
   - Graceful error recovery
   - State consistency checks
   - Error event publishing
   - Transaction rollback support

## Interaction Patterns

1. **Handler-Worker Communication**
   ```
   Handler --> StateManager --> Worker
     │            │             │
     └──────────>│<────────────┘
                 │
          Event Publishing
   ```

2. **State Update Flow**
   ```
   Event --> Handler --> StateManager
     │         │            │
     │         │         Worker
     │         │            │
     └─────────└────────────┘
           Event Chain
   ```