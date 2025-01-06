from typing import Dict, List, Callable, Optional
from datetime import datetime
import logging
from queue import PriorityQueue
import traceback

from drt_sim.models.event import Event, EventType

logger = logging.getLogger(__name__)

class EventManager:
    """
    Manages event registration, dispatch, and processing throughout the simulation.
    Includes validation, error handling, and event history tracking.
    """
    
    def __init__(self):
        self.handlers: Dict[EventType, List[Callable[[Event], None]]] = {}
        self.event_queue: PriorityQueue[Event] = PriorityQueue()
        self.event_history: List[Event] = []
        self.validation_rules: Dict[EventType, List[Callable[[Event], bool]]] = {}
        self.error_handlers: Dict[EventType, Callable[[Event, Exception], None]] = {}
    
    def register_handler(
        self,
        event_type: EventType,
        handler: Callable[[Event], None],
        validation_rules: Optional[List[Callable[[Event], bool]]] = None
    ) -> None:
        """Register an event handler with optional validation rules."""
        if event_type not in self.handlers:
            self.handlers[event_type] = []
        self.handlers[event_type].append(handler)
        
        if validation_rules:
            if event_type not in self.validation_rules:
                self.validation_rules[event_type] = []
            self.validation_rules[event_type].extend(validation_rules)
            
        logger.debug(f"Registered handler for event type: {event_type.value}")
    
    def process_event(self, event: Event) -> bool:
        """
        Process a single event immediately.
        
        Args:
            event: Event to process
            
        Returns:
            bool: True if event was successfully processed
        """
        if not self._validate_event(event):
            return False
            
        return self._process_event(event)
    
    def dispatch_event(self, event: Event) -> bool:
        """
        Queue an event for later processing.
        
        Args:
            event: Event to dispatch
            
        Returns:
            bool: True if event was successfully queued
        """
        try:
            # Validate event
            if not self._validate_event(event):
                return False
            
            # Add to queue
            self.event_queue.put(event)
            logger.debug(f"Queued event: {event.id} of type {event.event_type.value}")
            return True
            
        except Exception as e:
            logger.error(f"Error queuing event {event.id}: {str(e)}")
            return False
    
    def process_events(self, current_time: datetime) -> List[Event]:
        """
        Process all queued events up to current time.
        
        Args:
            current_time: Current simulation time
            
        Returns:
            List of processed events
        """
        processed_events = []
        
        while not self.event_queue.empty():
            # Peek at next event
            event = self.event_queue.queue[0]
            
            # Stop if event is in future
            if event.timestamp > current_time:
                break
            
            # Get and process event
            event = self.event_queue.get()
            if self._process_event(event):
                processed_events.append(event)
                
            # Add to history
            self.event_history.append(event)
        
        return processed_events
    
    def _process_event(self, event: Event) -> bool:
        """Process a single event through its handlers."""
        try:
            # Update event status
            event.mark_processing()
            
            # Get handlers for event type
            handlers = self.handlers.get(event.event_type, [])
            if not handlers:
                logger.warning(f"No handlers registered for event type: {event.event_type.value}")
                return False
            
            # Execute handlers
            for handler in handlers:
                try:
                    handler(event)
                except Exception as e:
                    self._handle_error(event, e)
                    return False
            
            event.mark_completed()
            return True
            
        except Exception as e:
            self._handle_error(event, e)
            return False
    
    def _validate_event(self, event: Event) -> bool:
        """Validate an event using registered validation rules."""
        try:
            # Basic validation
            if not event.event_type or not isinstance(event.event_type, EventType):
                logger.warning(f"Invalid event type for event {event.id}")
                return False
            
            if not event.timestamp:
                logger.warning(f"Missing timestamp for event {event.id}")
                return False
            
            # Check custom validation rules
            rules = self.validation_rules.get(event.event_type, [])
            for rule in rules:
                try:
                    if not rule(event):
                        logger.warning(f"Event {event.id} failed validation rule")
                        return False
                except Exception as e:
                    logger.error(f"Error in validation rule for event {event.id}: {str(e)}")
                    return False
            
            return True
            
        except Exception as e:
            logger.error(f"Error validating event {event.id}: {str(e)}")
            return False
    
    def _handle_error(self, event: Event, error: Exception) -> None:
        """Handle event processing error"""
        error_handler = self.error_handlers.get(event.event_type)
        if error_handler:
            try:
                error_handler(event, error)
            except Exception as e:
                logger.error(f"Error handler failed for event {event.id}: {str(e)}")
        
        event.mark_failed(str(error))
        logger.error(f"Error processing event {event.id}: {str(error)}\n{traceback.format_exc()}")
    
    def get_event_history(
        self,
        event_type: Optional[EventType] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None
    ) -> List[Event]:
        """Get filtered event history"""
        events = self.event_history
        
        if event_type:
            events = [e for e in events if e.event_type == event_type]
        if start_time:
            events = [e for e in events if e.timestamp >= start_time]
        if end_time:
            events = [e for e in events if e.timestamp <= end_time]
            
        return events
    
    def cleanup(self) -> None:
        """Clean up event manager resources"""
        while not self.event_queue.empty():
            self.event_queue.get()
        self.event_history.clear()
        self.handlers.clear()
        self.validation_rules.clear()
        self.error_handlers.clear()