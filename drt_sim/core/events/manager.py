from typing import Dict, List, Callable, Optional, Union, Coroutine, Any
from datetime import datetime
from queue import PriorityQueue
import traceback
import asyncio
from copy import deepcopy
from drt_sim.models.event import Event, EventType, EventPriority, EventStatus
import logging
logger = logging.getLogger(__name__)
HandlerType = Union[Callable[[Event], None], Callable[[Event], Coroutine[Any, Any, None]]]

class EventManager:
    """
    Manages event registration, dispatch, and processing throughout the simulation.
    Updated to work with immutable Event objects by creating new instances for status updates.
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
        handler: HandlerType,
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
            
        logger.debug(
            f"Registered {'async' if asyncio.iscoroutinefunction(handler) else 'sync'} "
            f"handler for event type: {event_type.value}"
        )

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
            self.event_history.append(processing_event)
            
            handlers = self.handlers.get(event.event_type, [])
            if not handlers:
                logger.warning(f"No handlers registered for event type: {event.event_type.value}")
                return False
            
            for handler in handlers:
                try:
                    if asyncio.iscoroutinefunction(handler):
                        await handler(event)
                    else:
                        handler(event)
                except Exception as e:
                    self._handle_error(event, e)
                    return False
            
            # Create new event with COMPLETED status
            completed_event = self._create_status_updated_event(event, EventStatus.COMPLETED)
            self.event_history.append(completed_event)
            
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
        self.event_history.append(failed_event)
        logger.error(f"Error processing event {event.id}: {str(error)}\n{traceback.format_exc()}")

    async def process_events(self, current_time: datetime) -> List[Event]:
        """Process all queued events up to current time."""
        processed_events = []
        
        while not self.event_queue.empty():
            next_event = self.event_queue.queue[0]  # Peek without removing
            if next_event.timestamp > current_time:
                break
            
            event = self.event_queue.get()
            if await self._process_event(event):
                processed_events.append(event)
        
        return processed_events
    
    def publish_event(self, event: Event) -> bool:
        """Queue an event for later processing."""
        try:
            if not self._validate_event(event):
                return False
            
            self.event_queue.put(event)
            return True
            
        except Exception as e:
            logger.error(f"Error queuing event {event.id}: {str(e)}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            return False
    
    def peek_next_event_time(self) -> Optional[datetime]:
        """Get timestamp of next event without removing it from queue."""
        if self.event_queue.empty():
            return None
        return self.event_queue.queue[0].timestamp
    
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
    
    def get_serializable_history(self) -> List[Dict[str, Any]]:
        """Convert event history to serializable format with explicit dict conversion"""
        history = []
        for event in self.event_history:
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
        while not self.event_queue.empty():
            self.event_queue.get()
        self.event_history.clear()
        self.handlers.clear()
        self.validation_rules.clear()
        self.error_handlers.clear()