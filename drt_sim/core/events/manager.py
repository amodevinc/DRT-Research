from typing import Dict, List, Callable, Optional, Union, Coroutine, Any
from datetime import datetime
from queue import PriorityQueue
import traceback
import asyncio
from copy import deepcopy
import threading
from drt_sim.models.event import Event, EventType, EventPriority, EventStatus
import logging
logger = logging.getLogger(__name__)
HandlerType = Union[Callable[[Event], None], Callable[[Event], Coroutine[Any, Any, None]]]

class EventManager:
    """
    Manages event registration, dispatch, and processing throughout the simulation.
    Updated to work with immutable Event objects by creating new instances for status updates.
    Thread-safe implementation for concurrent access.
    """
    
    def __init__(self, max_history_size: int = 10000, continue_on_handler_error: bool = False):
        self.handlers: Dict[EventType, List[Callable[[Event], None]]] = {}
        self.event_queue: PriorityQueue[Event] = PriorityQueue()
        self.event_history: List[Event] = []
        self.validation_rules: Dict[EventType, List[Callable[[Event], bool]]] = {}
        self.error_handlers: Dict[EventType, Callable[[Event, Exception], None]] = {}
        self.lock = threading.RLock()  # Reentrant lock for thread safety
        self.max_history_size = max_history_size
        self.continue_on_handler_error = continue_on_handler_error

    def get_queue_size(self) -> int:
        """Get the current size of the event queue."""
        with self.lock:
            return len(self.event_queue.queue)
    
    def register_handler(
        self,
        event_type: EventType,
        handler: HandlerType,
        validation_rules: Optional[List[Callable[[Event], bool]]] = None
    ) -> None:
        """Register an event handler with optional validation rules."""
        with self.lock:
            if event_type not in self.handlers:
                self.handlers[event_type] = []
            self.handlers[event_type].append(handler)
            
            if validation_rules:
                if event_type not in self.validation_rules:
                    self.validation_rules[event_type] = []
                self.validation_rules[event_type].extend(validation_rules)
                
            logger.debug(
                f"Registered {'async' if asyncio.iscoroutinefunction(handler) else 'sync'} "
                f"handler for event type: {event_type.value}"
            )
    
    def register_error_handler(
        self,
        event_type: EventType,
        error_handler: Callable[[Event, Exception], None]
    ) -> None:
        """Register an error handler for a specific event type."""
        with self.lock:
            self.error_handlers[event_type] = error_handler
            logger.debug(f"Registered error handler for event type: {event_type.value}")

    def schedule_recurring_event(
        self,
        event_type: EventType,
        start_time: datetime,
        interval_seconds: float,
        end_time: Optional[datetime] = None,
        priority: EventPriority = EventPriority.LOW,
        data: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Event:
        """Helper method to easily schedule recurring events."""
        event = Event(
            event_type=event_type,
            timestamp=start_time,
            priority=priority,
            is_recurring=True,
            recurrence_interval=interval_seconds,
            recurrence_end=end_time,
            data=data or {},
            metadata=metadata or {}
        )
        
        self.publish_event(event)
        logger.info(
            f"Scheduled recurring event {event_type.value} "
            f"every {interval_seconds} seconds starting at {start_time}"
        )
        return event
    
    def _create_status_updated_event(
        self,
        original_event: Event,
        new_status: EventStatus,
        error_message: Optional[str] = None
    ) -> Event:
        """Create a new event with updated status while preserving other attributes."""
        # Create deep copies of the immutable mappings
        data_copy = deepcopy(dict(original_event.data))
        metadata_copy = deepcopy(dict(original_event.metadata))
        service_metrics_copy = deepcopy(dict(original_event.service_metrics))
        location_copy = deepcopy(dict(original_event.location)) if original_event.location else None
        
        if error_message:
            data_copy['error_message'] = error_message
            
        return Event(
            id=original_event.id,  # Keep same ID to track event lifecycle
            event_type=original_event.event_type,
            priority=original_event.priority,
            timestamp=original_event.timestamp,
            status=new_status,
            vehicle_id=original_event.vehicle_id,
            request_id=original_event.request_id,
            passenger_id=original_event.passenger_id,
            route_id=original_event.route_id,
            stop_id=original_event.stop_id,
            scheduled_time=original_event.scheduled_time,
            actual_time=original_event.actual_time,
            created_at=original_event.created_at,
            processed_at=datetime.now() if new_status == EventStatus.PROCESSING else original_event.processed_at,
            completed_at=datetime.now() if new_status == EventStatus.COMPLETED else original_event.completed_at,
            waiting_time=original_event.waiting_time,
            ride_time=original_event.ride_time,
            walking_distance=original_event.walking_distance,
            deviation_minutes=original_event.deviation_minutes,
            service_metrics=service_metrics_copy,
            location=location_copy,
            data=data_copy,
            metadata=metadata_copy,
            is_recurring=original_event.is_recurring,
            recurrence_interval=original_event.recurrence_interval,
            recurrence_end=original_event.recurrence_end
        )
    
    async def _process_event(self, event: Event) -> bool:
        """Process a single event through its handlers."""
        try:
            # Create new event with PROCESSING status
            processing_event = self._create_status_updated_event(event, EventStatus.PROCESSING)
            self._add_to_history(processing_event)
            
            handlers = self.handlers.get(event.event_type, [])
            if not handlers:
                logger.warning(f"No handlers registered for event type: {event.event_type.value}")
                return False
            
            success = True
            for handler in handlers:
                try:
                    if asyncio.iscoroutinefunction(handler):
                        await handler(event)
                    else:
                        handler(event)
                except Exception as e:
                    self._handle_error(event, e)
                    success = False
                    if not self.continue_on_handler_error:
                        return False
            
            if success:
                # Create new event with COMPLETED status
                completed_event = self._create_status_updated_event(event, EventStatus.COMPLETED)
                self._add_to_history(completed_event)
                
                # Handle recurring events
                if event.is_recurring:
                    next_event = event.create_next_recurrence()
                    if next_event:
                        self.publish_event(next_event)
                        logger.info(
                            f"Scheduled next recurrence of event {event.event_type.value} "
                            f"for {next_event.timestamp}"
                        )
                
                return True
            return False
            
        except Exception as e:
            self._handle_error(event, e)
            return False

    def _handle_error(self, event: Event, error: Exception) -> None:
        """Handle event processing error by creating new failed event"""
        error_handler = self.error_handlers.get(event.event_type)
        if error_handler:
            try:
                error_handler(event, error)
            except Exception as e:
                logger.error(f"Error handler failed for event {event.id}: {str(e)}")
        
        # Create new event with FAILED status and error message
        failed_event = self._create_status_updated_event(event, EventStatus.FAILED, str(error))
        self._add_to_history(failed_event)
        logger.error(f"Error processing event {event.id}: {str(error)}\n{traceback.format_exc()}")

    def _add_to_history(self, event: Event) -> None:
        """Add event to history with size limit enforcement"""
        with self.lock:
            self.event_history.append(event)
            # Trim history if it exceeds the maximum size
            if len(self.event_history) > self.max_history_size:
                self.event_history = self.event_history[-self.max_history_size:]

    async def process_events(self, current_time: datetime) -> List[Event]:
        """Process all queued events up to current time."""
        processed_events = []
        
        with self.lock:
            events_to_process = []
            while not self.event_queue.empty():
                # Peek at the next event
                next_event = self.event_queue.queue[0]
                if next_event.timestamp > current_time:
                    break
                
                # Get the event for processing
                event = self.event_queue.get()
                events_to_process.append(event)
        
        # Process events outside the lock to avoid deadlocks
        for event in events_to_process:
            if await self._process_event(event):
                processed_events.append(event)
        
        return processed_events
    
    def get_all_events(self) -> List[Event]:
        """Get all events from the event queue."""
        with self.lock:
            return list(self.event_queue.queue)
    
    def publish_event(self, event: Event) -> bool:
        """Queue an event for later processing."""
        try:
            validation_result = self._validate_event(event)
            if not validation_result[0]:
                # Create a failed event for validation failure
                error_message = validation_result[1]
                failed_event = self._create_status_updated_event(
                    event, 
                    EventStatus.FAILED, 
                    f"Validation failed: {error_message}"
                )
                self._add_to_history(failed_event)
                return False
            
            with self.lock:
                self.event_queue.put(event)
            return True
            
        except Exception as e:
            logger.error(f"Error queuing event {event.id}: {str(e)}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            return False
    
    def peek_next_event_time(self) -> Optional[datetime]:
        """Get timestamp of next event without removing it from queue."""
        with self.lock:
            if self.event_queue.empty():
                return None
            return self.event_queue.queue[0].timestamp
    
    def _validate_event(self, event: Event) -> tuple[bool, Optional[str]]:
        """
        Validate an event using registered validation rules.
        
        Returns:
            tuple: (is_valid, error_message)
        """
        try:
            # Basic validation
            if not event.event_type or not isinstance(event.event_type, EventType):
                return False, "Invalid event type"
            
            if not event.timestamp:
                return False, "Missing timestamp"
            
            # Check custom validation rules
            rules = self.validation_rules.get(event.event_type, [])
            for rule in rules:
                try:
                    if not rule(event):
                        return False, "Failed custom validation rule"
                except Exception as e:
                    return False, f"Error in validation rule: {str(e)}"
            
            return True, None
            
        except Exception as e:
            return False, f"Validation error: {str(e)}"
    
    def get_event_history(
        self,
        event_type: Optional[EventType] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None
    ) -> List[Event]:
        """Get filtered event history"""
        with self.lock:
            events = self.event_history.copy()
        
        if event_type:
            events = [e for e in events if e.event_type == event_type]
        if start_time:
            events = [e for e in events if e.timestamp >= start_time]
        if end_time:
            events = [e for e in events if e.timestamp <= end_time]
            
        return events
    
    def get_serializable_history(self) -> List[Dict[str, Any]]:
        """Convert event history to serializable format with explicit dict conversion"""
        with self.lock:
            events_to_serialize = self.event_history.copy()
            
        history = []
        for event in events_to_serialize:
            try:
                event_dict = event.to_dict()
                # Double-check all nested dictionaries are converted from MappingProxyType
                event_dict['service_metrics'] = dict(event.service_metrics) if event.service_metrics else {}
                event_dict['location'] = dict(event.location) if event.location else None
                event_dict['data'] = dict(event.data) if event.data else {}
                event_dict['metadata'] = dict(event.metadata) if event.metadata else {}
                history.append(event_dict)
            except Exception as e:
                logger.error(f"Error serializing event {event.id}: {str(e)}\n"
                           f"{traceback.format_exc()}")
                # Add debug info
                logger.error(f"Event data types: service_metrics: {type(event.service_metrics)}, "
                           f"location: {type(event.location)}, "
                           f"data: {type(event.data)}, "
                           f"metadata: {type(event.metadata)}")
        return history
        
    def cleanup(self) -> None:
        """Clean up event manager resources"""
        with self.lock:
            while not self.event_queue.empty():
                self.event_queue.get()
            self.event_history.clear()
            self.handlers.clear()
            self.validation_rules.clear()
            self.error_handlers.clear()

    def cancel_event(self, event_id: str) -> bool:
        """
        Cancel a pending event by its ID.
        
        Args:
            event_id: The ID of the event to cancel
            
        Returns:
            bool: True if event was found and canceled, False otherwise
        """
        try:
            logger.info(f"Attempting to cancel event with ID: {event_id}")
            
            with self.lock:
                # Use a more efficient approach with a temporary queue
                temp_queue = PriorityQueue()
                event_to_cancel = None
                event_count = 0
                
                # Move events to temp queue while searching for the one to cancel
                while not self.event_queue.empty():
                    event = self.event_queue.get()
                    event_count += 1
                    
                    if event.id == event_id:
                        event_to_cancel = event
                    else:
                        temp_queue.put(event)
                
                # If we didn't find the event, put everything back and return
                if not event_to_cancel:
                    logger.warning(f"No pending event found with ID {event_id}")
                    # Restore the original queue
                    while not temp_queue.empty():
                        self.event_queue.put(temp_queue.get())
                    return False
                
                # Log event details before canceling
                logger.info(f"Found event to cancel:")
                logger.info(f"  Event Type: {event_to_cancel.event_type.value}")
                logger.info(f"  Timestamp: {event_to_cancel.timestamp}")
                logger.info(f"  Priority: {event_to_cancel.priority}")
                logger.info(f"  Vehicle ID: {event_to_cancel.vehicle_id if event_to_cancel.vehicle_id else 'None'}")
                logger.info(f"  Request ID: {event_to_cancel.request_id if event_to_cancel.request_id else 'None'}")
                logger.info(f"  Passenger ID: {event_to_cancel.passenger_id if event_to_cancel.passenger_id else 'None'}")
                logger.info(f"  Data: {dict(event_to_cancel.data)}")
                
                # Restore the queue without the canceled event
                while not temp_queue.empty():
                    self.event_queue.put(temp_queue.get())
                
                logger.debug(f"Removed event from queue. New queue size: {event_count - 1}")
                
                # Create canceled event for history
                canceled_event = self._create_status_updated_event(
                    event_to_cancel,
                    EventStatus.CANCELLED,
                    "Event explicitly canceled"
                )
                self._add_to_history(canceled_event)
                
                logger.info(f"Successfully canceled event {event_id} of type {event_to_cancel.event_type.value}")
                return True
                
        except Exception as e:
            logger.error(f"Error canceling event {event_id}: {str(e)}\n{traceback.format_exc()}")
            return False