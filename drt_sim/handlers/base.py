# drt_sim/handlers/base.py

from typing import Dict, Any, Optional, List, Callable
from datetime import datetime
import logging
import traceback

from drt_sim.models.event import Event, EventType, EventStatus, EventPriority
from drt_sim.core.simulation.context import SimulationContext
from drt_sim.core.state.manager import StateManager
from drt_sim.config.config import ScenarioConfig

class BaseHandler:
    """
    Base class for event handlers providing common functionality.
    All specific handlers should inherit from this class.
    """
    
    def __init__(
        self,
        config: ScenarioConfig,
        context: SimulationContext,
        state_manager: StateManager
    ):
        """
        Initialize base handler.
        
        Args:
            config: Scenario configuration
            context: Simulation context
            state_manager: State manager instance
        """
        self.config = config
        self.context = context
        self.state_manager = state_manager
        
        # Initialize logger with handler class name
        self.logger = logging.getLogger(f"{self.__class__.__name__}")
        
        # Handler registration mapping
        self._handlers: Dict[EventType, Callable[[Event], None]] = {}
        
        # Handler statistics
        self.stats: Dict[str, int] = {
            'events_processed': 0,
            'events_failed': 0,
            'total_processing_time': 0
        }
        
        # Register handler methods
        self._register_handlers()
    
    def _register_handlers(self) -> None:
        """
        Register handler methods based on method naming convention.
        Methods starting with 'handle_' are automatically registered.
        """
        for attr_name in dir(self):
            if attr_name.startswith('handle_'):
                event_type = attr_name[7:].upper()  # Convert handle_event_name to EVENT_NAME
                if hasattr(EventType, event_type):
                    handler_method = getattr(self, attr_name)
                    self._handlers[EventType[event_type]] = handler_method
    
    def handle_event(self, event: Event) -> Optional[List[Event]]:
        """
        Main entry point for handling events.
        
        Args:
            event: Event to handle
            
        Returns:
            Optional list of follow-up events
        """
        start_time = datetime.now()
        follow_up_events: List[Event] = []
        
        try:
            # Check if we have a handler for this event type
            handler = self._handlers.get(event.event_type)
            if not handler:
                self.logger.warning(f"No handler registered for event type: {event.event_type}")
                return None
            
            # Validate event before processing
            if not self._validate_event(event):
                self.logger.error(f"Event validation failed: {event.id}")
                return None
            
            # Mark event as processing
            event.mark_processing()
            
            # Execute handler
            result = handler(event)
            
            # Handle any follow-up events returned by the handler
            if result:
                if isinstance(result, list):
                    follow_up_events.extend(result)
                else:
                    follow_up_events.append(result)
            
            # Mark event as completed
            event.mark_completed()
            
            # Update stats
            self.stats['events_processed'] += 1
            processing_time = (datetime.now() - start_time).total_seconds()
            self.stats['total_processing_time'] += processing_time
            
            return follow_up_events
            
        except Exception as e:
            # Mark event as failed
            event.mark_failed(str(e))
            
            # Update stats
            self.stats['events_failed'] += 1
            
            # Log error with stack trace
            self.logger.error(
                f"Error handling event {event.id} of type {event.event_type}: {str(e)}\n"
                f"Stack trace:\n{traceback.format_exc()}"
            )
            
            # Create error event if needed
            error_event = self._create_error_event(event, e)
            if error_event:
                follow_up_events.append(error_event)
            
            return follow_up_events
    
    def _validate_event(self, event: Event) -> bool:
        """
        Validate event before processing.
        
        Args:
            event: Event to validate
            
        Returns:
            bool indicating if event is valid
        """
        try:
            # Check required fields
            if not event.id or not event.event_type:
                self.logger.error(f"Missing required fields in event: {event.id}")
                return False
            
            # Check timestamp
            if event.timestamp > self.context.current_time:
                self.logger.error(f"Event timestamp is in the future: {event.id}")
                return False
            
            # Check status
            if event.status != EventStatus.PENDING:
                self.logger.error(f"Event has already been processed: {event.id}")
                return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error validating event: {str(e)}")
            return False
    
    def _create_error_event(self, original_event: Event, error: Exception) -> Optional[Event]:
        """
        Create error event from failed event.
        
        Args:
            original_event: Failed event
            error: Exception that occurred
            
        Returns:
            New error event
        """
        try:
            return Event(
                event_type=EventType.SIMULATION_ERROR,
                timestamp=self.context.current_time,
                priority=EventPriority.CRITICAL,
                data={
                    'original_event_id': original_event.id,
                    'original_event_type': original_event.event_type.value,
                    'error_message': str(error),
                    'error_type': error.__class__.__name__
                }
            )
        except Exception as e:
            self.logger.error(f"Error creating error event: {str(e)}")
            return None
    
    def get_stats(self) -> Dict[str, Any]:
        """Get handler statistics"""
        if self.stats['events_processed'] > 0:
            avg_processing_time = (
                self.stats['total_processing_time'] / self.stats['events_processed']
            )
        else:
            avg_processing_time = 0
            
        return {
            **self.stats,
            'average_processing_time': avg_processing_time,
            'success_rate': (
                (self.stats['events_processed'] - self.stats['events_failed']) /
                self.stats['events_processed'] if self.stats['events_processed'] > 0 else 0
            )
        }
    
    def reset_stats(self) -> None:
        """Reset handler statistics"""
        self.stats = {
            'events_processed': 0,
            'events_failed': 0,
            'total_processing_time': 0
        }