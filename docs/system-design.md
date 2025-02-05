# DRT Research Platform Comprehensive Documentation

## Table of Contents
1. [System Overview](#system-overview)
2. [Platform Architecture](#platform-architecture)
3. [Core Components](#core-components)
   - [Study Runner](#study-runner)
   - [Experiment Runner](#experiment-runner)
   - [Scenario Runner](#scenario-runner)
   - [Simulation Orchestrator](#simulation-orchestrator)
   - [Simulation Engine](#simulation-engine)
   - [State Manager](#state-manager)
   - [Event Manager](#event-manager)
4. [Configuration System](#configuration-system)
5. [Study Types](#study-types)
6. [Event System](#event-system)
7. [State Management](#state-management)
8. [Integration Points](#integration-points)
9. [Best Practices and Usage](#best-practices-and-usage)

## System Overview

The DRT (Demand Responsive Transportation) Research Platform is a comprehensive simulation framework designed for studying various aspects of demand-responsive transportation systems. The platform enables researchers and practitioners to:

- Conduct systematic comparisons of different dispatch strategies
- Perform parameter sweep studies to optimize system configurations
- Analyze the impact of different service configurations
- Evaluate system performance under varying demand patterns
- Study the effects of different routing algorithms
- Answer key questions relating to Demand Responsive Transit via conducting simulations and visualizing results.

### Key Features

- Hierarchical study management system
- Event-driven architecture
- Comprehensive state management
- Extensive metrics collection
- Distributed execution support
- Transaction-based state updates
- Flexible configuration system
- Ability to provide a visual playback of the simulation.

## Platform Architecture

### Hierarchical Structure

The platform follows a strict hierarchical structure with clear separation of concerns:

```plaintext
CLI Entry Point (run_study.py)
└── StudyRunner
    ├── Configuration Management
    ├── MLflow Integration
    ├── Execution Control
    └── ExperimentRunner[]
        ├── Replication Management
        ├── Metrics Collection
        └── ScenarioRunner[]
            ├── Simulation Control
            └── SimulationOrchestrator
                ├── SimulationEngine
                ├── StateManager
                ├── EventManager
                └── Core Components
                    ├── DemandManager
                    ├── NetworkManager
                    └── Various Handlers
```

### Data Flow

The platform implements a robust data flow pattern:

1. Configuration Flow:
   - YAML files → Configuration Objects → Component Initialization
2. Event Flow:
   - Event Generation → Event Queue → Handlers → State Updates
3. State Flow:
   - State Changes → Transaction Management → State Workers → Persistence
4. Metrics Flow:
   - Collection → Aggregation → MLflow → Analysis

## Core Components

### Study Runner

The StudyRunner is the top-level coordinator responsible for managing the entire study execution process.

#### Key Responsibilities:
- Study configuration management
- MLflow experiment tracking
- Distributed execution coordination
- Resource management
- Results aggregation

#### Implementation Details:

```python
class StudyRunner:
    def __init__(self, cfg: StudyConfig):
        self.cfg = cfg
        self.experiment_runners: Dict[str, ExperimentRunner] = {}
        self.sim_paths = create_simulation_environment()
        self.study_paths = self.sim_paths.get_study_paths(self.cfg.metadata.name)
        
    async def run(self) -> Dict[str, Any]:
        results = {}
        active_run = None
        
        try:
            active_run = mlflow.start_run(run_name=self.cfg.metadata.name)
            self._log_study_params()

            if self.cfg.execution.distributed:
                results = await self._run_parallel()
            else:
                results = await self._run_sequential()

            processed_results = self._process_results(results)
            self._save_results(processed_results)
            self._log_study_metrics(processed_results)
            
            return processed_results
            
        finally:
            if active_run:
                mlflow.end_run()
```

#### Study Types Handler:

```python
def _initialize_experiments(self) -> None:
    if self.cfg.type == StudyType.PARAMETER_SWEEP:
        self._initialize_parameter_sweep()
    else:
        self._initialize_standard_experiments()

def _initialize_parameter_sweep(self) -> None:
    param_configs = self._generate_parameter_combinations()
    
    for i, param_config in enumerate(param_configs):
        exp_name = f"sweep_{i}"
        exp_paths = self.study_paths.get_experiment_paths(exp_name)
        exp_config = self._create_sweep_experiment_config(param_config, exp_name)
        self.experiment_runners[exp_name] = ExperimentRunner(
            cfg=exp_config,
            sim_cfg=self.cfg.simulation,
            paths=exp_paths
        )
```

### Experiment Runner

The ExperimentRunner manages individual experiments and their replications.

#### Key Responsibilities:
- Scenario execution coordination
- Replication management
- Experiment-level metrics collection
- Results aggregation
- State persistence

#### Implementation Details:

```python
class ExperimentRunner:
    def __init__(
        self,
        cfg: Union[ExperimentConfig, Dict[str, Any]],
        sim_cfg: SimulationConfig,
        paths: ExperimentPaths
    ):
        self.cfg = cfg if isinstance(cfg, ExperimentConfig) else ExperimentConfig(**cfg)
        self.sim_cfg = sim_cfg
        self.paths = paths
        self.scenario_runners: Dict[str, Union[ScenarioRunner, ray.actor.ActorHandle]] = {}

    async def run(self) -> Dict[str, Any]:
        logger.info(f"Running experiment: {self.cfg.name}")
        start_time = datetime.now()
        
        try:
            with mlflow.start_run(run_name=self.cfg.name, nested=True):
                self._log_experiment_params()
                
                if self.cfg.execution.distributed:
                    results = await self._run_distributed()
                else:
                    results = await self._run_sequential()
                    
                processed_results = self._process_results(results, start_time)
                self._save_results(processed_results)
                self._log_experiment_metrics(processed_results)
                
                return processed_results
                
        except Exception as e:
            logger.error(f"Experiment execution failed: {str(e)}")
            raise
```

#### Replication Management:

```python
async def _run_sequential(self) -> Dict[str, Any]:
    results = {}
    
    for scenario_name, runner in self.scenario_runners.items():
        scenario_results = []
        
        num_replications = self.cfg.simulation.replications
        
        for rep in range(num_replications):
            try:
                logger.info(f"Running scenario {scenario_name} replication {rep}")
                rep_results = await runner.run(rep)
                scenario_results.append({
                    "replication": rep,
                    "status": "completed",
                    "results": rep_results
                })
            except Exception as e:
                logger.error(f"Scenario {scenario_name} replication {rep} failed: {str(e)}")
                scenario_results.append({
                    "replication": rep,
                    "status": "failed",
                    "error": str(e)
                })
                raise
        
        results[scenario_name] = scenario_results
```
### Scenario Runner

The ScenarioRunner is responsible for executing individual simulation scenarios and managing their lifecycle.

#### Key Responsibilities:
- Simulation initialization and execution
- State management
- Metrics collection
- Event history tracking
- Error handling and recovery

#### Implementation Details:

```python
class ScenarioRunner:
    def __init__(
        self,
        cfg: ScenarioConfig,
        sim_cfg: SimulationConfig,
        paths: ScenarioPaths
    ):
        self.cfg = cfg
        self.sim_cfg = sim_cfg
        self.paths = paths
        self.orchestrator: Optional[SimulationOrchestrator] = None
        self.metrics_collector: Optional[MetricsCollector] = None
        self.current_replication: int = 0
        self.state_history: Dict[int, Any] = {}

    async def run(self, replication: int) -> Dict[str, Any]:
        """Execute the scenario for a specific replication."""
        self.current_replication = replication
        rep_paths = self.paths.get_replication_paths(replication)
        rep_paths.ensure_replication_structure()
        
        try:
            self.setup(rep_paths)
            
            with mlflow.start_run(
                run_name=f"{self.cfg.name}_rep_{replication}", 
                nested=True
            ):
                self._log_scenario_params(replication)
                results = await self._execute_simulation(rep_paths)
                self._log_metrics(results)
                return results
                
        except Exception as e:
            logger.error(f"Scenario execution failed: {str(e)}")
            await self._save_error_state(rep_paths)
            raise
            
        finally:
            self.cleanup()
```

#### Simulation Execution:

```python
async def _execute_simulation(self, rep_paths: ReplicationPaths) -> Dict[str, Any]:
    step_results = []
    last_metrics_collection = datetime.now()
    last_state_save = datetime.now()
    
    try:
        # Initialize simulation
        self.orchestrator.initialize()
        initial_state = self.orchestrator.get_state()
        
        if self.sim_cfg.save_state:
            self.state_history[0] = initial_state
            self._save_state(initial_state, 0, rep_paths)
        
        # Main simulation loop
        while self.orchestrator.context.status not in [
            SimulationStatus.COMPLETED,
            SimulationStatus.FAILED,
            SimulationStatus.STOPPED
        ]:
            current_time = self.orchestrator.context.current_time
            
            # Execute simulation step
            step_result = await self.orchestrator.step()
            step_results.append(step_result)
            
            # Collect metrics if configured
            if (self.metrics_collector and 
                (current_time - last_metrics_collection).total_seconds() >= 
                self.cfg.metrics.collect_interval):
                metrics = self.metrics_collector.collect(
                    self.orchestrator.get_state()
                )
                self._save_metrics(metrics, current_time, rep_paths)
                last_metrics_collection = current_time
            
            # Save state if enabled
            if (self.sim_cfg.save_state and 
                (current_time - last_state_save).total_seconds() >= 
                self.sim_cfg.save_interval):
                state = self.orchestrator.get_state()
                self.state_history[int(current_time.timestamp())] = state
                self._save_state(state, int(current_time.timestamp()), rep_paths)
                last_state_save = current_time
            
            # Process results
            results = self._process_results(step_results)
            self._save_results(results, rep_paths)
            
            return results
            
    except Exception as e:
        logger.error("Error during simulation execution", exc_info=True)
        raise
```

### Simulation Orchestrator

The SimulationOrchestrator is the central coordinator for the simulation system, managing interactions between various components.

#### Key Responsibilities:
- Component lifecycle management
- Event coordination
- State management
- Resource coordination
- System initialization
- Error handling

#### Implementation Details:

```python
class SimulationOrchestrator:
    def __init__(
        self,
        cfg: ScenarioConfig,
        sim_cfg: SimulationConfig,
        output_dir: Optional[Path] = None
    ):
        self.cfg = cfg
        self.sim_cfg = sim_cfg
        self.output_dir = output_dir
        
        # Core components
        self.context: Optional[SimulationContext] = None
        self.engine: Optional[SimulationEngine] = None
        self.state_manager: Optional[StateManager] = None
        self.event_manager: Optional[EventManager] = None
        self.demand_manager: Optional[DemandManager] = None
        self.network_manager: Optional[NetworkManager] = None
        
        # Handlers
        self.request_handler: Optional[RequestHandler] = None
        self.vehicle_handler: Optional[VehicleHandler] = None
        self.passenger_handler: Optional[PassengerHandler] = None
        self.route_handler: Optional[RouteHandler] = None
        self.stop_handler: Optional[StopHandler] = None
```

#### Initialization and Setup:

```python
def initialize(self) -> None:
    """Initialize all simulation components."""
    if self.initialized:
        logger.warning("Simulation already initialized")
        return
        
    try:
        logger.info("Initializing simulation components")
    
        # Initialize state management
        self.state_manager = StateManager(
            config=self.cfg, 
            sim_cfg=self.sim_cfg
        )
        
        # Initialize event system
        self.event_manager = EventManager()

        # Create context
        self.context = SimulationContext(
            start_time=datetime.fromisoformat(self.sim_cfg.start_time),
            end_time=datetime.fromisoformat(self.sim_cfg.end_time),
            time_step=timedelta(seconds=self.sim_cfg.time_step),
            warm_up_duration=timedelta(seconds=self.sim_cfg.warm_up_duration),
            event_manager=self.event_manager
        )
        
        # Initialize managers
        self.demand_manager = DemandManager(config=self.cfg.demand)
        self.network_manager = NetworkManager(config=self.cfg.network)
        
        # Initialize handlers
        self._initialize_handlers()
        
        # Initialize engine last
        self.engine = SimulationEngine(
            context=self.context,
            state_manager=self.state_manager,
            event_manager=self.event_manager
        )
        
        # Register handlers and schedule initial events
        self._register_handlers()
        self._schedule_all_demand()

        self.initialized = True
        logger.info("Simulation initialization completed successfully")
        
    except Exception as e:
        logger.error(f"Failed to initialize simulation: {str(e)}")
        self.cleanup()
        raise
```

#### Handler Registration:

```python
def _register_handlers(self) -> None:
    """Register event handlers with the event manager."""
    handlers_map = {
        # Request Events
        EventType.REQUEST_RECEIVED: self.request_handler.handle_request_received,
        EventType.REQUEST_VALIDATION_FAILED: 
            self.request_handler.handle_request_validation_failed,
        EventType.REQUEST_ASSIGNED: self.request_handler.handle_request_assigned,
        
        # Vehicle Events
        EventType.VEHICLE_CREATED: self.vehicle_handler.handle_vehicle_activation,
        EventType.VEHICLE_ACTIVATED: self.vehicle_handler.handle_vehicle_activation,
        EventType.VEHICLE_ASSIGNMENT: self.vehicle_handler.handle_vehicle_assignment,
        
        # Passenger Events
        EventType.PASSENGER_BOARDING: self.passenger_handler.handle_passenger_boarding,
        EventType.PASSENGER_IN_VEHICLE: self.passenger_handler.handle_passenger_in_vehicle,
        
        # Route Events
        EventType.ROUTE_CREATED: self.route_handler.handle_route_creation,
        EventType.ROUTE_UPDATED: self.route_handler.handle_route_update_request,
        
        # Stop Events
        EventType.STOP_ACTIVATED: self.stop_handler.handle_stop_activation_request,
        EventType.STOP_DEACTIVATED: 
            self.stop_handler.handle_stop_deactivation_request
    }
    
    for event_type, handler in handlers_map.items():
        self.event_manager.register_handler(
            event_type=event_type,
            handler=handler,
            validation_rules=self._get_validation_rules(event_type)
        )
```
### Simulation Engine

The SimulationEngine is responsible for time progression and core simulation mechanics.

#### Key Responsibilities:
- Time progression management
- Event processing coordination
- Performance tracking
- State snapshots
- Simulation status management

#### Implementation Details:

```python
class SimulationEngine:
    def __init__(
        self,
        context: SimulationContext,
        state_manager: StateManager,
        event_manager: EventManager,
        time_limit: Optional[int] = None
    ):
        self.context = context
        self.state_manager = state_manager
        self.event_manager = event_manager
        self.time_limit = time_limit
        
        # Performance tracking
        self.total_events_processed = 0
        self.total_steps_executed = 0
        self.start_time: Optional[datetime] = None
        self.end_time: Optional[datetime] = None
        
    async def step(self) -> SimulationStep:
        """Execute one simulation step."""
        step_start_time = datetime.now()
        
        try:
            # Process current time events
            processed_events = await self.event_manager.process_events(
                self.context.current_time
            )
            events_processed = len(processed_events)
            
            # Update tracking
            self.total_events_processed += events_processed
            self.total_steps_executed += 1
            
            # Take state snapshot
            self.state_manager.take_snapshot(self.context.current_time)
            
            # Advance time and check completion
            self.context.advance_time()
            self._check_completion()
            
            # Return step results
            return SimulationStep(
                timestamp=self.context.current_time,
                events_processed=events_processed,
                state_snapshot=self.state_manager.get_current_state(),
                metrics=self._get_step_metrics(),
                status=self.context.status,
                execution_time=(datetime.now() - step_start_time).total_seconds()
            )
            
        except Exception as e:
            logger.error(f"Error during simulation step: {str(e)}")
            self.context.status = SimulationStatus.FAILED
            raise
```

### State Management

The state management provides transactional state updates and persistence.

#### Core Components:

1. **StateManager**: Coordinates multiple state workers
2. **StateWorker**: Base class for specialized workers
3. **State Containers**: Generic containers for state items

#### Implementation Details:

```python
@dataclass
class StateContainer(Generic[T]):
    """Generic container for state items with history tracking"""
    _items: Dict[str, T] = field(default_factory=dict)
    _history: Dict[datetime, Dict[str, T]] = field(
        default_factory=lambda: defaultdict(dict)
    )
    _snapshot_buffer: Dict[str, T] = field(default_factory=dict)
    
    def begin_transaction(self) -> None:
        """Begin state transaction"""
        self._snapshot_buffer = self._items.copy()
        
    def commit_transaction(self) -> None:
        """Commit current transaction"""
        self._snapshot_buffer.clear()
        
    def rollback_transaction(self) -> None:
        """Rollback current transaction"""
        self._items = self._snapshot_buffer.copy()
        self._snapshot_buffer.clear()
```

#### State Workers:

```python
class StateWorker(ABC):
    """Abstract base class for state workers"""
    
    @abstractmethod
    def initialize(self, config: Optional[Any] = None) -> None:
        """Initialize worker with config"""
        pass
        
    @abstractmethod
    def take_snapshot(self, timestamp: datetime) -> None:
        """Take state snapshot"""
        pass
        
    @abstractmethod
    def get_metrics(self) -> Dict[str, Any]:
        """Get worker metrics"""
        pass
        
    @abstractmethod
    def get_state(self) -> Dict[str, Any]:
        """Get current worker state"""
        pass
```

#### State Manager Implementation:

```python
class StateManager:
    """Coordinates multiple state workers and manages system-wide state"""
    
    def __init__(self, config: ScenarioConfig, sim_cfg: SimulationConfig):
        # Initialize state workers
        self.stop_worker = StopStateWorker()
        self.vehicle_worker = VehicleStateWorker()
        self.request_worker = RequestStateWorker()
        self.route_worker = RouteStateWorker()
        self.passenger_worker = PassengerStateWorker()
        
        self.workers: List[StateWorker] = [
            self.vehicle_worker,
            self.request_worker,
            self.route_worker,
            self.passenger_worker,
            self.stop_worker
        ]
        
        # Initialize base state
        self.state = SimulationState(
            current_time=datetime.fromisoformat(sim_cfg.start_time),
            status=SimulationStatus.INITIALIZED,
            vehicles={},
            requests={},
            routes={},
            passengers={},
            stops={}
        )
        
    def get_current_state(self) -> SimulationState:
        """Get current state from all workers"""
        if not self.state:
            raise RuntimeError("State not initialized")
            
        # Get component states
        vehicle_state = self.vehicle_worker.get_state()
        request_state = self.request_worker.get_state()
        passenger_state = self.passenger_worker.get_state()
        route_state = self.route_worker.get_state()
        stop_state = self.stop_worker.get_state()
        
        return SimulationState(
            current_time=self.state.current_time,
            status=self.state.status,
            vehicles=VehicleSystemState(**vehicle_state),
            requests=RequestSystemState(**request_state),
            passengers=PassengerSystemState(**passenger_state),
            routes=RouteSystemState(**route_state),
            stops=StopSystemState(**stop_state)
        )
```

### Event System

The event system manages all simulation events and their processing.

#### Key Components:

1. **EventManager**: Manages event queue and processing
2. **Event Types**: Comprehensive enumeration of all events
3. **Event Handlers**: Specialized handlers for each event type
4. **Event Validation**: Rules for event validation

#### Implementation Details:

```python
class EventManager:
    def __init__(self):
        self.event_queue: PriorityQueue = PriorityQueue()
        self.handlers: Dict[EventType, List[Callable]] = defaultdict(list)
        self.validation_rules: Dict[EventType, List[Callable]] = defaultdict(list)
        self.event_history: List[Event] = []
        
    def publish_event(self, event: Event) -> None:
        """Publish an event to the queue"""
        self.event_queue.put((event.priority, event))
        
    async def process_events(self, current_time: datetime) -> List[Event]:
        """Process all events for current time step"""
        processed_events = []
        
        while not self.event_queue.empty():
            _, event = self.event_queue.get()
            
            if event.timestamp > current_time:
                # Put back future events
                self.event_queue.put((event.priority, event))
                break
                
            try:
                # Validate event
                if not self._validate_event(event):
                    continue
                    
                # Process event
                handlers = self.handlers.get(event.type, [])
                for handler in handlers:
                    await handler(event)
                    
                processed_events.append(event)
                self.event_history.append(event)
                
            except Exception as e:
                logger.error(f"Error processing event {event}: {str(e)}")
                raise
                
        return processed_events
```
### Integration Points

#### MLflow Integration

The platform uses MLflow for experiment tracking, metric logging, and artifact management.

```python
class MLflowManager:
    def __init__(self, study_paths: StudyPaths):
        self.study_paths = study_paths
        mlflow.set_tracking_uri(str(study_paths.mlruns))
        
    def setup_study_tracking(self, study_config: StudyConfig) -> None:
        """Configure MLflow tracking for a study"""
        mlflow.set_experiment(study_config.metadata.name)
        mlflow.set_tags({
            "study_type": study_config.type.value,
            "study_version": study_config.metadata.version,
            "authors": ", ".join(study_config.metadata.authors),
            "tags": ", ".join(study_config.metadata.tags),
            "timestamp": datetime.now().isoformat()
        })
        
    def log_experiment_metrics(self, metrics: Dict[str, Any]) -> None:
        """Log experiment metrics to MLflow"""
        for metric_name, value in metrics.items():
            if isinstance(value, (int, float)):
                mlflow.log_metric(metric_name, value)
            elif isinstance(value, dict):
                for sub_name, sub_value in value.items():
                    if isinstance(sub_value, (int, float)):
                        mlflow.log_metric(f"{metric_name}.{sub_name}", sub_value)
```

#### Ray Integration for Distributed Execution

```python
@ray.remote
class RayScenarioRunner:
    """Ray actor wrapper for distributed scenario execution"""
    
    def __init__(self, scenario_cfg: ScenarioConfig, sim_cfg: SimulationConfig, paths: Path):
        self.runner = ScenarioRunner(
            cfg=scenario_cfg,
            sim_cfg=sim_cfg,
            paths=paths
        )
        
    async def run(self, replication: int) -> Dict[str, Any]:
        return await self.runner.run(replication)
        
    def cleanup(self):
        self.runner.cleanup()
```

### Configuration System

The configuration system uses a hierarchical structure with inheritance and validation.

#### Configuration Classes:

```python
@dataclass
class DataclassYAMLMixin:
    """Mixin providing YAML serialization for dataclasses"""
    
    @classmethod
    def from_yaml(cls, yaml_path: Path) -> 'DataclassYAMLMixin':
        with open(yaml_path) as f:
            data = yaml.safe_load(f)
        return cls(**data)
        
    def to_yaml(self, yaml_path: Path) -> None:
        with open(yaml_path, 'w') as f:
            yaml.dump(self.to_dict(), f)
            
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

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

        # Create study directories
        self.paths.create_study_dirs(self.metadata.name)
        
        # Set default name if not provided
        if not self.metadata.name:
            self.metadata.name = f"study_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

@dataclass
class ScenarioConfig(DataclassYAMLMixin):
    """Configuration for a specific simulation scenario"""
    name: str
    description: str = ""
    service: Union[ServiceConfig, Dict[str, Any]] = field(default_factory=ServiceConfig)
    route: Union[RouteConfig, Dict[str, Any]] = field(default_factory=RouteConfig)
    stop: Union[StopConfig, Dict[str, Any]] = field(default_factory=StopConfig)
    demand: Union[DemandConfig, Dict[str, Any]] = field(default_factory=DemandConfig)
    vehicle: Union[VehicleConfig, Dict[str, Any]] = field(default_factory=VehicleConfig)
    algorithm: Union[AlgorithmConfig, Dict[str, Any]] = field(default_factory=AlgorithmConfig)
    network: Union[NetworkConfig, Dict[str, Any]] = field(default_factory=NetworkConfig)
    metrics: Optional[Union[MetricsConfig, Dict[str, Any]]] = None

    def __post_init__(self):
        """Initialize nested configurations"""
        # Convert dictionary configs to proper dataclass instances
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
        if isinstance(self.network, dict):
            self.network = NetworkConfig(**self.network)
        if isinstance(self.metrics, dict):
            self.metrics = MetricsConfig(**self.metrics)
```

#### Configuration Validation:

```python
def validate_study_config(config: StudyConfig) -> None:
    """Validate study configuration"""
    if not config.metadata.name:
        raise ValueError("Study name is required in metadata")
        
    if config.type == StudyType.PARAMETER_SWEEP:
        if not config.parameter_sweep or not config.parameter_sweep.enabled:
            raise ValueError(
                "Parameter sweep configuration is required for parameter sweep studies"
            )
            
    elif not config.experiments:
        raise ValueError(
            "At least one experiment configuration is required for non-parameter sweep studies"
        )
            
def validate_scenario_config(config: ScenarioConfig) -> None:
    """Validate scenario configuration"""
    if not config.name:
        raise ValueError("Scenario name is required")
        
    if not config.algorithm.dispatch_strategy:
        raise ValueError("Dispatch strategy must be specified")
        
    if config.vehicle.fleet_size <= 0:
        raise ValueError("Fleet size must be positive")
```

### Best Practices and Usage

#### Code Organization

1. **Module Structure**:
   ```plaintext
   drt_sim/
   ├── core/
   │   ├── simulation/
   │   ├── state/
   │   ├── events/
   │   └── monitoring/
   ├── handlers/
   ├── models/
   ├── algorithms/
   ├── config/
   └── utils/
   ```

2. **Naming Conventions**:
   - Use descriptive names for classes and methods
   - Follow Python PEP 8 style guide
   - Use type hints consistently
   - Document all public interfaces

#### Custom Algorithm Implementation

```python
class CustomDispatchStrategy(BaseDispatchStrategy):
    """Example custom dispatch strategy implementation"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.optimization_params = config.get('optimization_params', {})
        
    async def optimize_assignments(
        self,
        requests: List[Request],
        vehicles: List[Vehicle]
    ) -> List[Assignment]:
        """Implement custom assignment logic"""
        assignments = []
        
        # Custom optimization logic here
        
        return assignments
```