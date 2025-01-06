from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List
import logging
import heapq
from dataclasses import dataclass

from drt_sim.core.simulation.context import SimulationContext
from drt_sim.core.events.manager import EventManager
from drt_sim.models.simulation import SimulationStatus
from drt_sim.core.state.manager import StateManager
from drt_sim.models.event import Event
from drt_sim.config.config import DataclassYAMLMixin
logger = logging.getLogger(__name__)

@dataclass
class SimulationStep(DataclassYAMLMixin):
    """Represents the result of a single simulation step"""
    timestamp: datetime
    events_processed: int
    state_snapshot: Dict[str, Any]
    metrics: Dict[str, float]
    status: SimulationStatus
    execution_time: float  # in seconds

class SimulationEngine:
    """
    Core simulation engine responsible for time progression and event processing.
    Coordinates between different components to execute the simulation.
    """
    
    def __init__(
        self,
        context: SimulationContext,
        state_manager: StateManager,
        event_manager: EventManager,
        time_limit: Optional[int] = None  # in seconds
    ):
        """
        Initialize simulation engine.
        
        Args:
            context: Simulation context managing time and status
            state_manager: Manager for simulation state
            event_manager: Manager for events
            time_limit: Optional time limit for simulation run
        """
        self.context = context
        self.state_manager = state_manager
        self.event_manager = event_manager
        self.time_limit = time_limit
        
        # Performance tracking
        self.total_events_processed = 0
        self.total_steps_executed = 0
        self.start_time: Optional[datetime] = None
        self.end_time: Optional[datetime] = None
        
        # Event queue for future events
        self.future_events: List[Event] = []
        heapq.heapify(self.future_events)
    
    def initialize(self) -> None:
        """Initialize the simulation engine"""
        logger.info("Initializing simulation engine")
        
        try:
            # Record start time
            self.start_time = datetime.now()
            
            # Reset counters
            self.total_events_processed = 0
            self.total_steps_executed = 0
            
            # Clear future events
            self.future_events.clear()
            
            # Set initial simulation status
            if self.context.warm_up_duration.total_seconds() > 0:
                self.context.status = SimulationStatus.WARMING_UP
            else:
                self.context.status = SimulationStatus.RUNNING
            
            logger.info("Simulation engine initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize simulation engine: {str(e)}")
            self.context.status = SimulationStatus.FAILED
            raise
    
    def step(self) -> SimulationStep:
        """
        Execute one simulation step.
        
        Returns:
            SimulationStep containing results of the step
        """
        step_start_time = datetime.now()
        events_processed = 0
        
        try:
            # Process all events for current time step
            current_events = self._get_current_events()
            for event in current_events:
                self.event_manager.process_event(event)
                events_processed += 1
            
            # Update total counts
            self.total_events_processed += events_processed
            self.total_steps_executed += 1
            
            # Take state snapshot
            self.state_manager.take_snapshot(self.context.current_time)
            
            # Advance simulation time
            self.context.advance_time()
            
            # Check for simulation completion
            self._check_completion()
            
            # Calculate step execution time
            execution_time = (datetime.now() - step_start_time).total_seconds()
            
            # Create step result
            step_result = SimulationStep(
                timestamp=self.context.current_time,
                events_processed=events_processed,
                state_snapshot=self.state_manager.get_current_state(),
                metrics=self._get_step_metrics(),
                status=self.context.status,
                execution_time=execution_time
            )
            
            return step_result
            
        except Exception as e:
            logger.error(f"Error during simulation step: {str(e)}")
            self.context.status = SimulationStatus.FAILED
            raise
    
    def _get_current_events(self) -> List[Event]:
        """Get all events for current time step"""
        current_events = []
        
        # Get events from future events queue
        while (self.future_events and 
               self.future_events[0].timestamp <= self.context.current_time):
            event = heapq.heappop(self.future_events)
            current_events.append(event)
        
        # Sort events by priority
        current_events.sort(key=lambda x: x.priority)
        
        return current_events
    
    def schedule_event(self, event: Event) -> None:
        """
        Schedule an event for future processing
        
        Args:
            event: Event to schedule
        """
        if event.timestamp < self.context.current_time:
            raise ValueError("Cannot schedule event in the past")
        
        heapq.heappush(self.future_events, event)
    
    def _check_completion(self) -> None:
        """Check if simulation should complete"""
        # Check time limit
        if self.time_limit:
            elapsed = (datetime.now() - self.start_time).total_seconds()
            if elapsed >= self.time_limit:
                logger.info("Simulation time limit reached")
                self.context.status = SimulationStatus.COMPLETED
                return
        
        # Check simulation end time
        if self.context.current_time >= self.context.end_time:
            logger.info("Simulation end time reached")
            self.context.status = SimulationStatus.COMPLETED
            return
        
        # Check for no more events
        if not self.future_events and self.context.status == SimulationStatus.RUNNING:
            logger.info("No more events to process")
            self.context.status = SimulationStatus.COMPLETED
    
    def _get_step_metrics(self) -> Dict[str, float]:
        """Get metrics for current step"""
        return {
            'events_processed': self.total_events_processed,
            'steps_executed': self.total_steps_executed,
            'events_per_step': (self.total_events_processed / self.total_steps_executed 
                              if self.total_steps_executed > 0 else 0),
            'future_events': len(self.future_events)
        }
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get engine performance metrics"""
        elapsed = ((self.end_time or datetime.now()) - 
                  (self.start_time or datetime.now())).total_seconds()
        
        return {
            'total_events_processed': self.total_events_processed,
            'total_steps_executed': self.total_steps_executed,
            'events_per_second': self.total_events_processed / elapsed if elapsed > 0 else 0,
            'steps_per_second': self.total_steps_executed / elapsed if elapsed > 0 else 0,
            'average_events_per_step': (self.total_events_processed / self.total_steps_executed 
                                      if self.total_steps_executed > 0 else 0),
            'elapsed_time': elapsed
        }
    
    def cleanup(self) -> None:
        """Clean up engine resources"""
        try:
            # Record end time
            self.end_time = datetime.now()
            
            # Clear future events
            self.future_events.clear()
            
            # Log final statistics
            logger.info("Simulation engine cleanup complete")
            logger.info(f"Total events processed: {self.total_events_processed}")
            logger.info(f"Total steps executed: {self.total_steps_executed}")
            
        except Exception as e:
            logger.error(f"Error during engine cleanup: {str(e)}")
            raise