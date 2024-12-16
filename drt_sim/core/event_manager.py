# drt_sim/core/event_manager.py
from typing import Optional, Dict, List, Callable, Any
from datetime import datetime
import heapq
from dataclasses import dataclass, field
from enum import Enum, auto
import logging
import uuid

class EventType(Enum):
    """Types of events in the DRT simulation"""
    # Simulation control events
    SIMULATION_START = auto()
    SIMULATION_END = auto()
    WARM_UP_END = auto()
    COOL_DOWN_START = auto()
    
    # Vehicle events
    VEHICLE_DEPARTURE = auto()
    VEHICLE_ARRIVAL = auto()
    VEHICLE_BREAK_START = auto()
    VEHICLE_BREAK_END = auto()
    VEHICLE_SHIFT_END = auto()
    VEHICLE_MAINTENANCE = auto()
    
    # Passenger events
    REQUEST_ARRIVAL = auto()
    PASSENGER_PICKUP = auto()
    PASSENGER_DROPOFF = auto()
    REQUEST_CANCELLATION = auto()
    REQUEST_TIMEOUT = auto()
    
    # System events
    DISPATCH_OPTIMIZATION = auto()
    ROUTE_RECALCULATION = auto()
    REBALANCING_TRIGGER = auto()
    TRAFFIC_UPDATE = auto()
    WEATHER_UPDATE = auto()
    
    # Analysis events
    METRICS_COLLECTION = auto()
    STATE_SNAPSHOT = auto()

@dataclass(order=True)
class SimulationEvent:
    """Event class for the DRT simulation"""
    timestamp: datetime
    event_type: EventType
    priority: int
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    data: Dict = field(default_factory=dict)
    dependencies: List[str] = field(default_factory=list)
    handler: Optional[Callable] = field(default=None, compare=False)
    processed: bool = field(default=False)
    
    def __post_init__(self):
        """Ensure all required fields are properly initialized"""
        if not isinstance(self.event_type, EventType):
            raise ValueError(f"Invalid event type: {self.event_type}")
        if not isinstance(self.priority, int) or not (0 <= self.priority <= 10):
            raise ValueError(f"Priority must be an integer between 0 and 10")

class EventManager:
    """Manages events for the DRT simulation"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self._event_queue: List[SimulationEvent] = []
        self._event_map: Dict[str, SimulationEvent] = {}
        self._dependency_map: Dict[str, List[str]] = {}
        self._default_handlers: Dict[EventType, Callable] = {}
        self.processed_count = 0
        
    @property
    def pending_count(self) -> int:
        """Get number of pending events"""
        return len(self._event_queue)
    
    def register_handler(self, event_type: EventType, handler: Callable) -> None:
        """Register a default handler for an event type"""
        self._default_handlers[event_type] = handler
        self.logger.debug(f"Registered handler for {event_type}")
    
    def schedule_event(self, event: SimulationEvent) -> None:
        """Schedule a new event"""
        # Validate event
        if event.id in self._event_map:
            raise ValueError(f"Event with ID {event.id} already exists")
            
        # Add to queue and map
        heapq.heappush(self._event_queue, event)
        self._event_map[event.id] = event
        
        # Update dependency tracking
        for dep_id in event.dependencies:
            if dep_id not in self._dependency_map:
                self._dependency_map[dep_id] = []
            self._dependency_map[dep_id].append(event.id)
            
        self.logger.debug(
            f"Scheduled {event.event_type} event at {event.timestamp}"
        )
    
    def cancel_event(self, event_id: str) -> bool:
        """Cancel a scheduled event"""
        if event_id not in self._event_map:
            return False
            
        # Remove from map and mark as processed
        event = self._event_map.pop(event_id)
        event.processed = True
        
        # Clean up dependencies
        self._cleanup_dependencies(event_id)
        
        self.logger.debug(f"Cancelled event {event_id}")
        return True
    
    def get_next_event(self) -> Optional[SimulationEvent]:
        """Get the next event that's ready to be processed"""
        while self._event_queue:
            # Get next event
            event = heapq.heappop(self._event_queue)
            
            # Skip if already processed
            if event.processed:
                continue
                
            # Check dependencies
            if not self._check_dependencies(event):
                heapq.heappush(self._event_queue, event)
                continue
                
            return event
            
        return None
    
    def process_event(self, event: SimulationEvent, context: Any) -> None:
        """Process an event with the appropriate handler"""
        try:
            # Use event-specific handler if available, otherwise use default
            handler = event.handler or self._default_handlers.get(event.event_type)
            
            if handler:
                handler(event, context)
            else:
                self.logger.warning(
                    f"No handler found for event type {event.event_type}"
                )
                
            # Mark as processed and clean up
            event.processed = True
            self._event_map.pop(event.id, None)
            self._cleanup_dependencies(event.id)
            self.processed_count += 1
            
        except Exception as e:
            self.logger.error(f"Error processing event {event.id}: {str(e)}")
            raise
    
    def _check_dependencies(self, event: SimulationEvent) -> bool:
        """Check if all event dependencies have been processed"""
        return all(
            dep_id not in self._event_map or 
            self._event_map[dep_id].processed
            for dep_id in event.dependencies
        )
    
    def _cleanup_dependencies(self, event_id: str) -> None:
        """Clean up dependency tracking for an event"""
        # Remove from dependency map
        self._dependency_map.pop(event_id, None)
        
        # Update dependent events
        dependent_events = self._dependency_map.get(event_id, [])
        for dep_event_id in dependent_events:
            if dep_event_id in self._event_map:
                dep_event = self._event_map[dep_event_id]
                dep_event.dependencies.remove(event_id)
    
    def get_events_by_type(self, event_type: EventType) -> List[SimulationEvent]:
        """Get all pending events of a specific type"""
        return [
            event for event in self._event_map.values()
            if event.event_type == event_type and not event.processed
        ]
    
    def clear(self) -> None:
        """Clear all pending events"""
        self._event_queue.clear()
        self._event_map.clear()
        self._dependency_map.clear()
        self.logger.debug("Cleared all pending events")