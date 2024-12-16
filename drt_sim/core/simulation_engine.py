# drt_sim/core/simulation_engine.py
from typing import Dict
from datetime import datetime
import logging
from dataclasses import dataclass
from enum import Enum, auto
from ..config.parameters import SimulationParameters, ScenarioParameters
from .event_manager import EventManager, SimulationEvent, EventType
from .state_management import StateManager
from .hooks import HookManager

class SimulationStatus(Enum):
    """Possible states of the simulation"""
    INITIALIZED = auto()
    RUNNING = auto()
    PAUSED = auto()
    COMPLETED = auto()
    ERROR = auto()

@dataclass
class SimulationContext:
    """Context object passed to event handlers"""
    current_time: datetime
    state: StateManager
    config: SimulationParameters
    event_manager: EventManager
    hook_manager: HookManager

class SimulationEngine:
    """Core simulation engine for DRT system"""
    
    def __init__(self, config: SimulationParameters, scenario: ScenarioParameters):
        self.config = config
        self.scenario = scenario
        self.logger = logging.getLogger(__name__)
        
        # Initialize components
        self.state = StateManager(scenario)
        self.event_manager = EventManager()
        self.hook_manager = HookManager()
        
        self.status = SimulationStatus.INITIALIZED
        self.current_time = config.start_time
        self._simulation_context = None
        
    def initialize(self) -> None:
        """Initialize the simulation"""
        self.logger.info("Initializing simulation...")
        
        # Create simulation context
        self._simulation_context = SimulationContext(
            current_time=self.current_time,
            state=self.state,
            config=self.config,
            event_manager=self.event_manager,
            hook_manager=self.hook_manager
        )
        
        try:
            # Initialize state with config
            self.state.initialize(self.config)
            
            # Schedule initial events
            self._schedule_initial_events()
            
            # Call initialization hooks
            self.hook_manager.call_hooks('simulation_init', self._simulation_context)
            
            self.logger.info("Simulation initialized successfully")
            
        except Exception as e:
            self.status = SimulationStatus.ERROR
            self.logger.error(f"Initialization failed: {str(e)}")
            raise
            
    def _schedule_initial_events(self) -> None:
        """Schedule events that should exist at simulation start"""
        # Schedule warm-up end event
        warm_up_end = self.config.start_time + self.config.warm_up_period
        self.event_manager.schedule_event(
            SimulationEvent(
                event_type=EventType.WARM_UP_END,
                timestamp=warm_up_end,
                priority=1,
                data={}
            )
        )
        
        # Schedule cool-down start event
        cool_down_start = self.config.end_time - self.config.cool_down_period
        self.event_manager.schedule_event(
            SimulationEvent(
                event_type=EventType.COOL_DOWN_START,
                timestamp=cool_down_start,
                priority=1,
                data={}
            )
        )
        
        # Schedule simulation end event
        self.event_manager.schedule_event(
            SimulationEvent(
                event_type=EventType.SIMULATION_END,
                timestamp=self.config.end_time,
                priority=0,
                data={}
            )
        )
        
    def run(self) -> None:
        """Run the simulation until completion"""
        self.logger.info(f"Starting simulation at {self.current_time}")
        self.status = SimulationStatus.RUNNING
        
        try:
            while self.status == SimulationStatus.RUNNING:
                # Get next event
                event = self.event_manager.get_next_event()
                if not event:
                    break
                    
                # Update simulation time
                self.current_time = event.timestamp
                self._simulation_context.current_time = self.current_time
                
                # Process event
                self._process_event(event)
                
                # Check for simulation end
                if self.current_time >= self.config.end_time:
                    break
                    
            self.status = SimulationStatus.COMPLETED
            self.logger.info("Simulation completed successfully")
            
        except Exception as e:
            self.status = SimulationStatus.ERROR
            self.logger.error(f"Simulation failed: {str(e)}")
            raise
            
    def _process_event(self, event: SimulationEvent) -> None:
        """Process a single simulation event"""
        self.logger.debug(f"Processing event: {event.event_type} at {event.timestamp}")
        
        # Pre-event hooks
        self.hook_manager.call_hooks('pre_event', self._simulation_context, event)
        
        # Process event based on type
        if event.event_type == EventType.SIMULATION_END:
            self.status = SimulationStatus.COMPLETED
        else:
            # Handle other event types
            self.event_manager.process_event(event, self._simulation_context)
        
        # Post-event hooks
        self.hook_manager.call_hooks('post_event', self._simulation_context, event)
        
    def pause(self) -> None:
        """Pause the simulation"""
        if self.status == SimulationStatus.RUNNING:
            self.status = SimulationStatus.PAUSED
            self.logger.info("Simulation paused")
            
    def resume(self) -> None:
        """Resume the simulation"""
        if self.status == SimulationStatus.PAUSED:
            self.status = SimulationStatus.RUNNING
            self.logger.info("Simulation resumed")
            
    def get_statistics(self) -> Dict:
        """Get current simulation statistics"""
        return {
            'current_time': self.current_time,
            'status': self.status,
            'processed_events': self.event_manager.processed_count,
            'pending_events': self.event_manager.pending_count,
            'metrics': self.state.get_metrics()
        }