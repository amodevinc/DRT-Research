from datetime import datetime, timedelta
from typing import Dict, Any, Optional
import logging
from pathlib import Path

from drt_sim.models.simulation import SimulationStatus
from drt_sim.core.monitoring.metrics_collector import MetricsCollector
from drt_sim.core.monitoring.types.metrics import MetricName
from drt_sim.core.simulation.context import SimulationContext
from drt_sim.core.simulation.engine import SimulationEngine
from drt_sim.models.simulation import SimulationState
from drt_sim.core.state.manager import StateManager
from drt_sim.core.events.manager import EventManager
from drt_sim.core.demand.manager import DemandManager
from drt_sim.core.user.manager import UserProfileManager
from drt_sim.config.config import ScenarioConfig, SimulationConfig
from drt_sim.handlers.request_handler import RequestHandler
from drt_sim.handlers.vehicle_handler import VehicleHandler
from drt_sim.handlers.passenger_handler import PassengerHandler
from drt_sim.handlers.route_handler import RouteHandler
from drt_sim.handlers.stop_handler import StopHandler
from drt_sim.handlers.matching_handler import MatchingHandler
from drt_sim.core.services.route_service import RouteService
from drt_sim.models.event import EventType
from drt_sim.core.logging_config import setup_logger
from drt_sim.network.manager import NetworkManager
from drt_sim.models.base import SimulationEncoder
import traceback
import json
logger = setup_logger(__name__)

class SimulationOrchestrator:
    """
    High-level coordinator for the simulation system. Manages the interactions between
    various subsystems and provides a unified interface for simulation control.
    """
    
    def __init__(
        self,
        cfg: ScenarioConfig,
        sim_cfg: SimulationConfig,
        output_dir: Optional[Path] = None,
        metrics_collector: Optional[MetricsCollector] = None
    ):
        """
        Initialize the simulation orchestrator.
        
        Args:
            cfg: Scenario configuration
            sim_cfg: Simulation configuration
            output_dir: Optional output directory for simulation artifacts
            metrics_collector: Optional metrics collector from parent scenario
        """
        self.cfg = cfg
        self.sim_cfg = sim_cfg
        self.output_dir = output_dir
        self.metrics_collector = metrics_collector
        
        # Core components - initialized in initialize()
        self.context: Optional[SimulationContext] = None
        self.engine: Optional[SimulationEngine] = None
        self.state_manager: Optional[StateManager] = None
        self.event_manager: Optional[EventManager] = None
        self.demand_manager: Optional[DemandManager] = None
        self.network_manager: Optional[NetworkManager] = None
        self.user_profile_manager: Optional[UserProfileManager] = None
        self.route_service: Optional[RouteService] = None
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
            
            
            # Initialize state management
            self.state_manager = StateManager(config=self.cfg, sim_cfg=self.sim_cfg)
            
            # Initialize event system
            self.event_manager = EventManager()

            # Create context first as other components depend on it
            self.context = SimulationContext(
                start_time=datetime.fromisoformat(self.sim_cfg.start_time),
                end_time=datetime.fromisoformat(self.sim_cfg.end_time),
                time_step=timedelta(seconds=self.sim_cfg.time_step),
                warm_up_duration=timedelta(seconds=self.sim_cfg.warm_up_duration),
                event_manager=self.event_manager,
                metrics_collector=self.metrics_collector
            )
            
            # Initialize demand management
            self.demand_manager = DemandManager(
                config=self.cfg.demand
            )

            # Initialize network manager
            self.network_manager = NetworkManager(config=self.cfg.network)

            # Initialize user profile manager
            self.user_profile_manager = UserProfileManager()

            # Initialize simulation engine last
            self.engine = SimulationEngine(
                context=self.context,
                state_manager=self.state_manager,
                event_manager=self.event_manager
            )
            self.route_service = RouteService(
                network_manager=self.network_manager,
                sim_context=self.context,
                config=self.cfg,
            )
            
            # Initialize handlers
            self._initialize_handlers()
            
            # Register handlers with event manager
            self._register_handlers()

            # Schedule all demand before starting simulation
            self._schedule_all_demand()

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
            state_manager=self.state_manager,
            network_manager=self.network_manager
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
            state_manager=self.state_manager,
            network_manager=self.network_manager
        )
        
        self.stop_handler = StopHandler(
            config=self.cfg,
            context=self.context,
            state_manager=self.state_manager,
            network_manager=self.network_manager
        )
        
        self.matching_handler = MatchingHandler(
            config=self.cfg,
            context=self.context,
            state_manager=self.state_manager,
            network_manager=self.network_manager,
            user_profile_manager=self.user_profile_manager,
            route_service=self.route_service
        )

    def _register_handlers(self) -> None:
        """Register event handlers with the event manager."""
        if not self.event_manager:
            raise RuntimeError("Event manager not initialized")
            
        # Define handler mapping
        handlers_map = {
            # Request Events
            EventType.REQUEST_RECEIVED: self.request_handler.handle_request_received,
            EventType.REQUEST_REJECTED: self.request_handler.handle_request_rejected,
            EventType.REQUEST_VALIDATION_FAILED: self.request_handler.handle_request_validation_failed,
            EventType.REQUEST_ASSIGNED: self.request_handler.handle_request_assigned,
            EventType.REQUEST_CANCELLED: self.request_handler.handle_request_cancelled,
            EventType.REQUEST_EXPIRED: self.request_handler.handle_request_expired,
            
            # Matching Events
            EventType.MATCH_REQUEST_TO_VEHICLE: self.matching_handler.handle_match_request_to_vehicle,
            
            # Vehicle Events
            EventType.VEHICLE_ARRIVED_STOP: self.vehicle_handler.handle_vehicle_arrived_stop,
            EventType.VEHICLE_DISPATCH_REQUEST: self.vehicle_handler.handle_vehicle_dispatch_request,
            EventType.VEHICLE_ACTIVE_ROUTE_UPDATE: self.vehicle_handler.handle_vehicle_active_route_id_update,
            EventType.VEHICLE_REROUTE_REQUEST: self.vehicle_handler.handle_vehicle_reroute_request,
            EventType.VEHICLE_EN_ROUTE: self.vehicle_handler.handle_vehicle_en_route,
            EventType.VEHICLE_REBALANCING_REQUIRED: self.vehicle_handler.handle_vehicle_rebalancing_required,
            EventType.VEHICLE_WAIT_TIMEOUT: self.vehicle_handler.handle_vehicle_wait_timeout,
            # Passenger Events
            EventType.START_PASSENGER_JOURNEY: self.passenger_handler.handle_start_passenger_journey,
            EventType.PASSENGER_WALKING_TO_PICKUP: self.passenger_handler.handle_passenger_walking_to_pickup,
            EventType.PASSENGER_ARRIVED_PICKUP: self.passenger_handler.handle_passenger_arrived_pickup,
            EventType.PASSENGER_BOARDING_COMPLETED: self.passenger_handler.handle_boarding_completed,
            EventType.PASSENGER_ALIGHTING_COMPLETED: self.passenger_handler.handle_alighting_completed,
            EventType.PASSENGER_ARRIVED_DESTINATION: self.passenger_handler.handle_passenger_arrived_destination,
            EventType.PASSENGER_NO_SHOW: self.passenger_handler.handle_passenger_no_show,
            EventType.SERVICE_LEVEL_VIOLATION: self.passenger_handler.handle_service_level_violation,
            
            # Route Events
            EventType.ROUTE_UPDATE_REQUEST: self.route_handler.handle_route_update_request,
            
            # Stop Events
            EventType.DETERMINE_VIRTUAL_STOPS: self.stop_handler.handle_determine_virtual_stops,
            EventType.STOP_SELECTION_TICK: self.stop_handler.handle_stop_selection_tick,
            EventType.STOP_ACTIVATED: self.stop_handler.handle_stop_activation_request,
            EventType.STOP_DEACTIVATED: self.stop_handler.handle_stop_deactivation_request,
        }
        
        # Register validation rules
        validation_rules = {
            EventType.REQUEST_RECEIVED: [
                lambda e: hasattr(e, 'request_id'),
                lambda e: hasattr(e, 'passenger_id'),
            ],
            EventType.VEHICLE_ASSIGNMENT: [
                lambda e: hasattr(e, 'vehicle_id'),
                lambda e: hasattr(e, 'request_id'),
            ],
            # Add other validation rules as needed
        }
        
        # Register all handlers with their validation rules and error handlers
        for event_type, handler in handlers_map.items():
            if not callable(handler):
                raise ValueError(f"Invalid handler for event type {event_type}")
                
            self.event_manager.register_handler(
                event_type=event_type,
                handler=handler,
                validation_rules=validation_rules.get(event_type)
            )
            
            logger.debug(f"Registered handler for {event_type}")
        
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

    def save_event_history(self, output_path: Path) -> None:
        """Save the complete event history"""
        if not self.event_manager:
            logger.warning("No event manager available")
            return
            
        try:
            events = self.event_manager.get_serializable_history()
            with open(output_path, 'w') as f:
                json.dump(events, f, cls=SimulationEncoder, indent=2)
            logger.info(f"Saved event history to {output_path}")
        except Exception as e:
            logger.error(f"Failed to save event history: {str(e)}\n{traceback.format_exc()}")
            
    async def step(self) -> Dict[str, Any]:
        """Execute one simulation step."""
        if not self.initialized:
            raise RuntimeError("Simulation not initialized")
            
        try:
            step_start_time = datetime.now()
            
            # Execute simulation step
            step_metrics = await self.engine.step()
            
            # Log step execution time
            step_duration = (datetime.now() - step_start_time).total_seconds()
            # if self.metrics_collector:
            #     self.metrics_collector.log(
            #         MetricName.SIMULATION_STEP_DURATION,
            #         step_duration,
            #         {
            #             'step': self.step_count,
            #             'current_time': self.context.current_time.isoformat()
            #         }
            #     )
            
            self.step_count += 1
            return step_metrics
            
        except Exception as e:
            logger.error(f"Error during simulation step: {str(e)}")
            raise

    def _schedule_all_demand(self) -> None:
        """Schedule all demand events for the entire simulation period."""
        if not self.demand_manager or not self.engine:
            raise RuntimeError("Required components not initialized")
            
        # Generate all demand events for the entire simulation period
        events, requests = self.demand_manager.generate_demand(
            self.context.start_time,
            self.context.end_time - self.context.start_time
        )
        
        # Schedule all events through the engine
        for event in events:
            self.engine.schedule_event(event)
            
        logger.info(
            f"Scheduled {len(events)} demand events "
            f"containing {len(requests)} requests for entire simulation"
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
        """Clean up simulation resources and export metrics."""
        try:
            if self.initialized:
                # Export metrics if collector exists and output directory is set
                if self.metrics_collector:
                    self.metrics_collector.flush_hierarchical()
                    
                # Clean up other components
                if self.state_manager:
                    self.state_manager.cleanup()
                if self.event_manager:
                    self.event_manager.cleanup()
                if self.demand_manager:
                    self.demand_manager.cleanup()
                    
                self.initialized = False
                logger.info("Simulation cleanup completed")
                
        except Exception as e:
            logger.error(f"Error during simulation cleanup: {str(e)}")
            raise