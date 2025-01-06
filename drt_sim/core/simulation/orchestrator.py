from datetime import datetime, timedelta
from typing import Dict, Any, Optional
import logging
from pathlib import Path

from drt_sim.models.simulation import SimulationStatus

from .context import SimulationContext
from .engine import SimulationEngine
from drt_sim.models.simulation import SimulationState
from drt_sim.core.state.manager import StateManager
from drt_sim.core.events.manager import EventManager
from drt_sim.core.demand.manager import DemandManager
from drt_sim.config.config import ScenarioConfig, SimulationConfig
from drt_research_platform.drt_sim.handlers.request_handler import RequestHandler
from drt_sim.handlers.vehicle_handlers import VehicleHandler
from drt_sim.handlers.system_handlers import SystemHandler
from drt_sim.handlers.passenger_handler import PassengerHandler
from drt_sim.handlers.dispatch_handler import DispatchHandler
from drt_sim.handlers.route_handler import RouteHandler
from drt_sim.handlers.stop_handler import StopHandler
from drt_sim.models.event import EventType

logger = logging.getLogger(__name__)

class SimulationOrchestrator:
    """
    High-level coordinator for the simulation system. Manages the interactions between
    various subsystems and provides a unified interface for simulation control.
    """
    
    def __init__(
        self,
        cfg: ScenarioConfig,
        sim_cfg: SimulationConfig,
        output_dir: Optional[Path] = None
    ):
        """
        Initialize the simulation orchestrator.
        
        Args:
            cfg: Scenario configuration
            sim_cfg: Simulation configuration
            output_dir: Optional output directory for simulation artifacts
        """
        self.cfg = cfg
        self.sim_cfg = sim_cfg
        self.output_dir = output_dir or Path("simulation_output")
        
        # Core components - initialized in initialize()
        self.context: Optional[SimulationContext] = None
        self.engine: Optional[SimulationEngine] = None
        self.state_manager: Optional[StateManager] = None
        self.event_manager: Optional[EventManager] = None
        self.demand_manager: Optional[DemandManager] = None
        
        # Event handlers
        self.request_handler: Optional[RequestHandler] = None
        self.vehicle_handler: Optional[VehicleHandler] = None
        self.system_handler: Optional[SystemHandler] = None
        
        # Tracking
        self.initialized: bool = False
        self.step_count: int = 0
        
    def initialize(self) -> None:
        """Initialize all simulation components and prepare for execution."""
        if self.initialized:
            logger.warning("Simulation already initialized")
            return
            
        try:
            logger.info("Initializing simulation components")
            
            # Create context first as other components depend on it
            self.context = SimulationContext(
                start_time=datetime.fromisoformat(self.sim_cfg.start_time),
                end_time=datetime.fromisoformat(self.sim_cfg.end_time),
                time_step=timedelta(seconds=self.sim_cfg.time_step),
                warm_up_duration=timedelta(seconds=self.sim_cfg.warm_up_duration)
            )
            
            # Initialize state management
            self.state_manager = StateManager(config=self.cfg, sim_cfg=self.sim_cfg)
            
            # Initialize event system
            self.event_manager = EventManager()
            
            # Initialize demand management
            self.demand_manager = DemandManager(
                config=self.cfg.demand
            )
            
            # Initialize handlers
            self._initialize_handlers()
            
            # Initialize simulation engine last
            self.engine = SimulationEngine(
                context=self.context,
                state_manager=self.state_manager,
                event_manager=self.event_manager
            )
            
            # Register handlers with event manager
            self._register_handlers()

            self.initialized = True
            logger.info("Simulation initialization completed successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize simulation: {str(e)}")
            self.cleanup()
            raise
            
    def _initialize_handlers(self) -> None:
        """Initialize event handlers."""
        self.request_handler = RequestHandler(
            config=self.cfg,
            context=self.context,
            state_manager=self.state_manager
        )
        
        self.dispatch_handler = DispatchHandler(
            config=self.cfg,
            context=self.context,
            state_manager=self.state_manager
        )
        
        self.vehicle_handler = VehicleHandler(
            config=self.cfg,
            context=self.context,
            state_manager=self.state_manager
        )
        
        self.passenger_handler = PassengerHandler(
            config=self.cfg,
            context=self.context,
            state_manager=self.state_manager
        )
        
        self.route_handler = RouteHandler(
            config=self.cfg,
            context=self.context,
            state_manager=self.state_manager
        )
        
        self.stop_handler = StopHandler(
            config=self.cfg,
            context=self.context,
            state_manager=self.state_manager
        )
        
        self.system_handler = SystemHandler(
            config=self.cfg,
            context=self.context,
            state_manager=self.state_manager
        )

    def _register_handlers(self) -> None:
        """Register event handlers with the event manager."""
        if not self.event_manager:
            raise RuntimeError("Event manager not initialized")
            
        # Define handler mapping
        handlers_map = {
            # Request Lifecycle Events
            EventType.REQUEST_CREATED: self.request_handler.handle_request_created,
            EventType.REQUEST_VALIDATED: self.request_handler.handle_request_validated,
            EventType.REQUEST_REJECTED: self.request_handler.handle_request_rejected,
            EventType.REQUEST_ASSIGNED: self.request_handler.handle_request_assigned,
            EventType.REQUEST_REASSIGNED: self.request_handler.handle_request_reassigned,
            EventType.REQUEST_CANCELLED: self.request_handler.handle_request_cancelled,
            EventType.REQUEST_PICKUP_STARTED: self.request_handler.handle_request_pickup_started,
            EventType.REQUEST_PICKUP_COMPLETED: self.request_handler.handle_request_pickup_completed,
            EventType.REQUEST_DROPOFF_STARTED: self.request_handler.handle_request_dropoff_started,
            EventType.REQUEST_DROPOFF_COMPLETED: self.request_handler.handle_request_dropoff_completed,
            EventType.REQUEST_EXPIRED: self.request_handler.handle_request_expired,
            EventType.REQUEST_NO_VEHICLE: self.request_handler.handle_request_no_vehicle,

            # Dispatch Events
            EventType.DISPATCH_REQUESTED: self.dispatch_handler.handle_dispatch_requested,
            EventType.DISPATCH_ASSIGNED: self.dispatch_handler.handle_dispatch_assigned,
            EventType.DISPATCH_REJECTED: self.dispatch_handler.handle_dispatch_rejected,
            
            # Passenger Journey Events
            EventType.PASSENGER_WALKING_TO_PICKUP: self.passenger_handler.handle_passenger_walking_to_pickup,
            EventType.PASSENGER_ARRIVED_AT_PICKUP_STOP: self.passenger_handler.handle_passenger_arrived_at_pickup_stop,
            EventType.PASSENGER_IN_VEHICLE: self.passenger_handler.handle_passenger_in_vehicle,
            EventType.PASSENGER_ARRIVED_AT_DESTINATION_STOP: self.passenger_handler.handle_passenger_arrived_at_destination_stop,
            EventType.PASSENGER_WALKING_TO_DESTINATION: self.passenger_handler.handle_passenger_walking_to_destination,
            EventType.PASSENGER_ARRIVED_AT_DESTINATION: self.passenger_handler.handle_passenger_arrived_at_destination,
            
            # Vehicle Events
            EventType.VEHICLE_CREATED: self.vehicle_handler.handle_vehicle_created,
            EventType.VEHICLE_ACTIVATED: self.vehicle_handler.handle_vehicle_activated,
            EventType.VEHICLE_DEPARTED: self.vehicle_handler.handle_vehicle_departed,
            EventType.VEHICLE_ARRIVED: self.vehicle_handler.handle_vehicle_arrived,
            EventType.VEHICLE_REROUTED: self.vehicle_handler.handle_vehicle_rerouted,
            EventType.VEHICLE_BREAKDOWN: self.vehicle_handler.handle_vehicle_breakdown,
            EventType.VEHICLE_AT_CAPACITY: self.vehicle_handler.handle_vehicle_at_capacity,
            
            # Route Events
            EventType.ROUTE_CREATED: self.route_handler.handle_route_created,
            EventType.ROUTE_UPDATED: self.route_handler.handle_route_updated,
            EventType.ROUTE_COMPLETED: self.route_handler.handle_route_completed,
            EventType.ROUTE_DELAYED: self.route_handler.handle_route_delayed,
            EventType.ROUTE_DETOUR_NEEDED: self.route_handler.handle_route_detour_needed,
            EventType.ROUTE_OPTIMIZATION_NEEDED: self.route_handler.handle_route_optimization_needed,
            EventType.VEHICLE_ROUTE_STARTED: self.route_handler.handle_route_started,
            EventType.ROUTE_STOP_REACHED: self.route_handler.handle_route_stop_reached,
            
            # Stop Events
            EventType.STOP_ACTIVATED: self.stop_handler.handle_stop_activated,
            EventType.STOP_DEACTIVATED: self.stop_handler.handle_stop_deactivated,
            EventType.STOP_CONGESTED: self.stop_handler.handle_stop_congested,
            EventType.STOP_CAPACITY_EXCEEDED: self.stop_handler.handle_stop_capacity_exceeded,
            
            # Passenger Events
            EventType.PASSENGER_WALKING_TO_PICKUP: self.passenger_handler.handle_passenger_walking_to_pickup,
            EventType.PASSENGER_ARRIVED_AT_PICKUP_STOP: self.passenger_handler.handle_passenger_arrived_at_pickup_stop,
            EventType.PASSENGER_IN_VEHICLE: self.passenger_handler.handle_passenger_in_vehicle,
            EventType.PASSENGER_ARRIVED_AT_DESTINATION_STOP: self.passenger_handler.handle_passenger_arrived_at_destination_stop,
            EventType.PASSENGER_WALKING_TO_DESTINATION: self.passenger_handler.handle_passenger_walking_to_destination,
            EventType.PASSENGER_ARRIVED_AT_DESTINATION: self.passenger_handler.handle_passenger_arrived_at_destination,
            
            # System Events
            EventType.SIMULATION_START: self.system_handler.handle_simulation_initialized,
            EventType.SIMULATION_END: self.system_handler.handle_simulation_completed,
            EventType.SIMULATION_ERROR: self.system_handler.handle_system_error,
            EventType.WARMUP_COMPLETED: self.system_handler.handle_warmup_completed
        }
        
        # Register all handlers
        for event_type, handler in handlers_map.items():
            if not callable(handler):
                raise ValueError(f"Invalid handler for event type {event_type}")
            self.event_manager.register_handler(event_type, handler)
            logger.debug(f"Registered handler for {event_type.value}")
        
    def _setup_initial_state(self) -> None:
        """Set up the initial simulation state."""
        if not self.state_manager:
            raise RuntimeError("State manager not initialized")
        
        # Set initial system state
        initial_state = SimulationState(
            current_time=self.context.current_time,
            status=SimulationStatus.INITIALIZED
        )
        
        self.state_manager.set_state(initial_state)
        logger.info("Initial simulation state configured")
            
    def step(self) -> Dict[str, Any]:
        """
        Execute one simulation step.
        
        Returns:
            Dict containing step results
        """
        if not self.initialized:
            raise RuntimeError("Simulation not initialized")
            
        try:
            # Process demand for current timestep
            self._process_demand()
            # Execute simulation step
            step_results = self.engine.step()
            # Update step count
            self.step_count += 1
            
            return step_results
            
        except Exception as e:
            logger.error(f"Error during simulation step: {str(e)}")
            self.context.status = SimulationStatus.FAILED
            raise
            
    def _process_demand(self) -> None:
        """Process demand for the current time step."""
        if not self.demand_manager:
            raise RuntimeError("Demand manager not initialized")
        if not self.event_manager:
            raise RuntimeError("Event manager not initialized")
            
        # Generate new events and requests for current timestep
        events, requests = self.demand_manager.generate_demand(
            self.context.current_time,
            self.context.time_step
        )
        
        # Process each event through the event manager
        for event in events:
            self.event_manager.process_event(event)
            
        logger.debug(
            f"Processed {len(events)} demand events "
            f"containing {len(requests)} requests for timestep"
        )
            
    def get_state(self) -> SimulationState:
        """
        Get current simulation state.
        
        Returns:
            Dict containing current state
        """
        if not self.state_manager:
            raise RuntimeError("State manager not initialized")
            
        return self.state_manager.get_state()
        
    def save_state(self, path: Optional[Path] = None) -> None:
        """
        Save current simulation state to file.
        
        Args:
            path: Optional path to save state file
        """
        if not self.state_manager:
            raise RuntimeError("State manager not initialized")
            
        self.state_manager.save_state(path or self.output_dir / "states")
        
    def load_state(self, path: Path) -> None:
        """
        Load simulation state from file.
        
        Args:
            path: Path to state file
        """
        if not self.state_manager:
            raise RuntimeError("State manager not initialized")
            
        self.state_manager.load_state(path)
        
    def cleanup(self) -> None:
        """Clean up simulation resources."""
        logger.info("Cleaning up simulation resources")
        
        try:
            # Clean up components in reverse order of initialization
            if self.engine:
                self.engine.cleanup()
                
            if self.demand_manager:
                self.demand_manager.cleanup()
                
            if self.event_manager:
                self.event_manager.cleanup()
                
            if self.state_manager:
                self.state_manager.cleanup()
                
            # Reset flags and counters
            self.initialized = False
            self.step_count = 0
            
            logger.info("Simulation cleanup completed")
            
        except Exception as e:
            logger.error(f"Error during cleanup: {str(e)}")
            raise