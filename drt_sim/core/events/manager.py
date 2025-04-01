from typing import Dict, List, Callable, Optional, Union, Coroutine, Any
from datetime import datetime
from queue import PriorityQueue
import traceback
import asyncio
from copy import deepcopy
import threading
from drt_sim.models.event import Event, EventType, EventPriority, EventStatus
import logging
import os

from drt_sim.core.events.store import EventHistoryStore

logger = logging.getLogger(__name__)
HandlerType = Union[Callable[[Event], None], Callable[[Event], Coroutine[Any, Any, None]]]

class EventManager:
    """
    Manages event registration, dispatch, and processing throughout the simulation.
    Updated to work with immutable Event objects by creating new instances for status updates.
    Thread-safe implementation for concurrent access.
    Enhanced with advanced event history storage for debugging and visualization.
    """
    
    def __init__(
        self, 
        output_dir: str = "output",
        max_memory_events: int = 10000, 
        continue_on_handler_error: bool = False,
        auto_save_frequency: int = 1000
    ):
        self.handlers: Dict[EventType, List[Callable[[Event], None]]] = {}
        self.event_queue: PriorityQueue[Event] = PriorityQueue()
        
        # Enhanced history storage
        os.makedirs(output_dir, exist_ok=True)
        debug_file_path = os.path.join(output_dir, "debug_events.jsonl")
        viz_file_path = os.path.join(output_dir, "visualization_events.json")
        
        self.event_history = EventHistoryStore(
            max_memory_events=max_memory_events,
            auto_save_events=True,
            debug_file_path=debug_file_path,
            viz_file_path=viz_file_path,
            save_interval=auto_save_frequency
        )
        
        self.validation_rules: Dict[EventType, List[Callable[[Event], bool]]] = {}
        self.error_handlers: Dict[EventType, Callable[[Event, Exception], None]] = {}
        self.lock = threading.RLock()  # Reentrant lock for thread safety
        self.max_history_size = max_memory_events
        self.continue_on_handler_error = continue_on_handler_error
        self.output_dir = output_dir

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
        """Add event to enhanced history store"""
        self.event_history.add_event(event)

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
        if event_type:
            events = self.event_history.get_events_by_type(event_type)
        else:
            events = self.event_history.events_by_timestamp.copy()
        
        if start_time:
            events = [e for e in events if e.timestamp >= start_time]
        if end_time:
            events = [e for e in events if e.timestamp <= end_time]
            
        return events
    
    def get_entity_events(
        self,
        entity_type: str,
        entity_id: str
    ) -> List[Event]:
        """Get all events for a specific entity (vehicle, passenger, request, etc.)"""
        return self.event_history.get_events_by_entity(entity_type, entity_id)
    
    def get_entity_timeline(
        self,
        entity_type: str,
        entity_id: str
    ) -> List[Dict]:
        """Get visualization timeline for a specific entity"""
        return self.event_history.get_entity_timeline(entity_type, entity_id)
    
    def save_event_history(self) -> None:
        """Manually save event history to files"""
        self.event_history.save_to_files()
        
    def load_event_history(self) -> None:
        """Load event history from files"""
        self.event_history.load_from_files()
    
    def get_visualization_data(self) -> Dict:
        """Get all entity timelines for visualization"""
        timelines = self.event_history.get_all_entity_timelines()
        
        return {
            "metadata": {
                "event_count": self.event_history.event_count,
                "generated_at": datetime.now().isoformat(),
                "version": "1.0"
            },
            "timelines": timelines
        }
    
    def cleanup(self) -> None:
        """Clean up event manager resources"""
        with self.lock:
            # Save history before cleaning up
            self.save_event_history()
            
            # Clear queue and all handlers
            while not self.event_queue.empty():
                self.event_queue.get()
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