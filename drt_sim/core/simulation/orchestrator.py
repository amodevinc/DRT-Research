from datetime import datetime, timedelta
from typing import Dict, Any, Optional
import logging
from pathlib import Path
import asyncio

from drt_sim.models.state import SimulationStatus
from drt_sim.core.monitoring.metrics.collector import MetricsCollector
from drt_sim.core.monitoring.visualization.manager import VisualizationManager
from drt_sim.core.simulation.context import SimulationContext
from drt_sim.core.simulation.engine import SimulationEngine, SimulationStep
from drt_sim.models.state import SimulationState
from drt_sim.core.state.manager import StateManager
from drt_sim.core.events.manager import EventManager
from drt_sim.core.demand.manager import DemandManager
from drt_sim.core.user.user_profile_manager import UserProfileManager
from drt_sim.core.user.user_acceptance_manager import UserAcceptanceManager
from drt_sim.config.config import ParameterSet, SimulationConfig
from drt_sim.handlers.request_handler import RequestHandler
from drt_sim.handlers.vehicle_handler import VehicleHandler
from drt_sim.handlers.passenger_handler import PassengerHandler
from drt_sim.handlers.route_handler import RouteHandler
from drt_sim.handlers.stop_handler import StopHandler
from drt_sim.handlers.matching_handler import MatchingHandler
from drt_sim.core.services.route_service import RouteService
from drt_sim.models.event import EventType
from drt_sim.network.manager import NetworkManager
from drt_sim.models.base import SimulationEncoder
from drt_sim.integration.traffic_sim_integration import SUMOIntegration
import traceback
import json
import logging
logger = logging.getLogger(__name__)
class SimulationOrchestrator:
    """
    High-level coordinator for the simulation system. Manages the interactions between
    various subsystems and provides a unified interface for simulation control.
    """
    
    def __init__(
        self,
        cfg: ParameterSet,
        sim_cfg: SimulationConfig,
        output_dir: Optional[Path] = None,
        metrics_collector: Optional[MetricsCollector] = None,
        replication_id: Optional[str] = None,
        artifact_dir: Optional[Path] = None
    ):
        """
        Initialize the simulation orchestrator.
        
        Args:
            cfg: Scenario configuration
            sim_cfg: Simulation configuration
            output_dir: Optional output directory for simulation artifacts
            metrics_collector: Optional metrics collector from parent scenario
            replication_id: Optional replication ID for logging
            artifact_dir: Optional artifact directory for MLflow logging
        """
        self.cfg = cfg
        self.sim_cfg = sim_cfg
        self.output_dir = output_dir
        self.metrics_collector = metrics_collector
        self.replication_id = replication_id
        self.artifact_dir = artifact_dir
    
        # Core components - initialized in initialize()
        self.context: Optional[SimulationContext] = None
        self.engine: Optional[SimulationEngine] = None
        self.state_manager: Optional[StateManager] = None
        self.event_manager: Optional[EventManager] = None
        self.demand_manager: Optional[DemandManager] = None
        self.network_manager: Optional[NetworkManager] = None
        self.user_profile_manager: Optional[UserProfileManager] = None
        self.user_acceptance_manager: Optional[UserAcceptanceManager] = None
        self.route_service: Optional[RouteService] = None
        self.visualization_manager: Optional[VisualizationManager] = None
        self.sumo_integration: Optional[SUMOIntegration] = None
        
        # Tracking
        self.initialized: bool = False
        self.step_count: int = 0
        
    async def initialize(self) -> None:
        """Initialize all simulation components and prepare for execution."""
        if self.initialized:
            logger.warning("Simulation already initialized")
            return
            
        try:
            logger.info("Initializing simulation components")
            
            # Initialize visualization manager if output directory is provided
            if self.output_dir:
                self.visualization_manager = VisualizationManager(self.output_dir)
            
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
            self.user_profile_manager = UserProfileManager(self.cfg.user_acceptance)

            # Initialize user acceptance manager
            self.user_acceptance_manager = UserAcceptanceManager(
                config=self.cfg.user_acceptance,
                user_profile_manager=self.user_profile_manager
            )

            # Initialize SUMO integration if enabled
            if self.sim_cfg.sumo.enabled:
                logger.info("Initializing SUMO integration")
                self.sumo_integration = SUMOIntegration(config=self.sim_cfg.sumo)
                await self.sumo_integration.initialize(drt_network_config=self.cfg.network)

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

            # Start SUMO if enabled
            if self.sumo_integration and self.sim_cfg.sumo.enabled:
                await self.sumo_integration.start()

            self.initialized = True
            logger.info("Simulation initialization completed successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize simulation: {str(e)}")
            self.cleanup()
            raise
            
    def _initialize_handlers(self) -> None:
        """Initialize all event handlers."""
        logger.info("Initializing handlers")
        
        # Initialize user profile manager
        self.user_profile_manager = UserProfileManager(self.cfg.user_acceptance)
        
        # Initialize user acceptance manager
        self.user_acceptance_manager = UserAcceptanceManager(
            config=self.cfg.user_acceptance,
            user_profile_manager=self.user_profile_manager
        )
        
        # Initialize route service
        self.route_service = RouteService(
            self.network_manager,
            self.context,
            self.state_manager
        )
        
        # Initialize request handler
        self.request_handler = RequestHandler(
            self.cfg,
            self.context,
            self.state_manager,
            self.network_manager
        )
        
        # Initialize vehicle handler
        self.vehicle_handler = VehicleHandler(
            self.cfg,
            self.context,
            self.state_manager,
        )
        
        # Initialize passenger handler
        self.passenger_handler = PassengerHandler(
            self.cfg,
            self.context,
            self.state_manager,
            self.network_manager
        )
        
        # Initialize route handler
        self.route_handler = RouteHandler(
            self.cfg,
            self.context,
            self.state_manager,
            self.network_manager
        )
        
        # Initialize stop handler
        self.stop_handler = StopHandler(
            self.cfg,
            self.context,
            self.state_manager,
            self.network_manager
        )
        
        # Initialize matching handler
        self.matching_handler = MatchingHandler(
            self.cfg,
            self.context,
            self.state_manager,
            self.network_manager,
            self.user_profile_manager,
            self.route_service,
            self.user_acceptance_manager
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
            EventType.VEHICLE_REROUTE_REQUEST: self.vehicle_handler.handle_vehicle_reroute_request,
            EventType.VEHICLE_POSITION_UPDATE: self.vehicle_handler.handle_vehicle_position_update,
            EventType.VEHICLE_REBALANCING_REQUIRED: self.vehicle_handler.handle_vehicle_rebalancing_required,
            EventType.VEHICLE_WAIT_TIMEOUT: self.vehicle_handler.handle_vehicle_wait_timeout,
            EventType.VEHICLE_STOP_OPERATIONS_COMPLETED: self.vehicle_handler.handle_stop_operations_completed,
            
            # Passenger Events
            EventType.START_PASSENGER_JOURNEY: self.passenger_handler.handle_start_passenger_journey,
            EventType.PASSENGER_WALKING_TO_PICKUP: self.passenger_handler.handle_passenger_walking_to_pickup,
            EventType.PASSENGER_ARRIVED_PICKUP: self.passenger_handler.handle_passenger_arrived_pickup,
            EventType.PASSENGER_BOARDING_COMPLETED: self.passenger_handler.handle_boarding_completed,
            EventType.PASSENGER_ALIGHTING_COMPLETED: self.passenger_handler.handle_alighting_completed,
            EventType.PASSENGER_ARRIVED_DESTINATION: self.passenger_handler.handle_passenger_arrived_destination,
            EventType.PASSENGER_NO_SHOW: self.passenger_handler.handle_passenger_no_show,
            EventType.SERVICE_LEVEL_VIOLATION: self.passenger_handler.handle_service_level_violation,
            EventType.PASSENGER_READY_FOR_BOARDING: self.vehicle_handler.handle_passenger_ready_for_boarding,
            
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
            EventType.VEHICLE_STOP_OPERATIONS_COMPLETED: [
                lambda e: hasattr(e, 'vehicle_id'),
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
            
    async def step(self) -> SimulationStep:
        """Execute one simulation step."""
        if not self.initialized:
            raise RuntimeError("Simulation not initialized")
            
        try:
            step_start_time = datetime.now()
            
            # Execute simulation step
            simulation_step = await self.engine.step()
            
            # Synchronize with SUMO if enabled
            if self.sumo_integration and self.sim_cfg.sumo.enabled:
                await self._synchronize_with_sumo()
                await self.sumo_integration.step(self.sim_cfg.time_step)
            
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
            return simulation_step
            
        except Exception as e:
            logger.error(f"Error during simulation step: {str(e)}")
            raise

    async def _synchronize_with_sumo(self) -> None:
        """Synchronize DRT simulation state with SUMO."""
        if not self.sumo_integration or not self.state_manager:
            return
            
        try:
            # Get current vehicle states
            state = self.state_manager.get_state()
            if not hasattr(state, 'vehicles') or not state.vehicles:
                return
                
            # Update each vehicle in SUMO
            for vehicle_id, vehicle in state.vehicles.items():
                if vehicle.route and vehicle.route.current_segment:
                    # Update vehicle position in SUMO
                    if vehicle.position:
                        await self.sumo_integration.update_vehicle_position(
                            vehicle_id=vehicle_id,
                            position=vehicle.position
                        )
                    
                    # If vehicle has a new route, update it in SUMO
                    if vehicle.route and vehicle.route.waypoints:
                        await self.sumo_integration.update_vehicle_route(
                            vehicle_id=vehicle_id,
                            route=vehicle.route.waypoints
                        )
        except Exception as e:
            logger.error(f"Error synchronizing with SUMO: {str(e)}")

    def _schedule_all_demand(self) -> None:
        """Schedule all demand events for the entire simulation period."""
        if not self.demand_manager or not self.engine:
            raise RuntimeError("Required components not initialized")
            
        # Generate all demand events for the entire simulation period
        events, requests = self.demand_manager.generate_demand(
            self.context.start_time,
            self.context.end_time - self.context.start_time
        )
        self.state_manager.request_worker.load_historical_requests(requests)
        
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
        """Clean up simulation resources."""
        try:
            if self.initialized:
                # Clean up SUMO integration if enabled
                if self.sumo_integration:
                    asyncio.create_task(self.sumo_integration.stop())
                
                # Clean up components
                if self.state_manager:
                    self.state_manager.cleanup()
                if self.event_manager:
                    self.event_manager.cleanup()
                if self.demand_manager:
                    self.demand_manager.cleanup()
                if self.visualization_manager:
                    self.visualization_manager.cleanup()
                    
                self.initialized = False
                logger.info("Simulation cleanup completed")
                
        except Exception as e:
            logger.error(f"Error during simulation cleanup: {str(e)}")
            raise