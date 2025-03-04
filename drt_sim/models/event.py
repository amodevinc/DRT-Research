from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, Any, Optional
import uuid
from types import MappingProxyType
import functools
from copy import deepcopy

class EventPriority(Enum):
    """Priority levels for events"""
    CRITICAL = 0  # System-critical events (crashes, errors)
    HIGH = 1      # Time-sensitive operations (pickups, dropoffs)
    NORMAL = 2    # Standard operations (route updates, status changes)
    LOW = 3       # Background operations (metrics collection, logging)
    
    def __lt__(self, other):
        if not isinstance(other, EventPriority):
            return NotImplemented
        return self.value < other.value
    
    def __le__(self, other):
        if not isinstance(other, EventPriority):
            return NotImplemented
        return self.value <= other.value
    
    def __gt__(self, other):
        if not isinstance(other, EventPriority):
            return NotImplemented
        return self.value > other.value
    
    def __ge__(self, other):
        if not isinstance(other, EventPriority):
            return NotImplemented
        return self.value >= other.value
    
class EventStatus(Enum):
    """Status of an event"""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

class EventType(Enum):
    """
    Comprehensive event types for DRT simulation covering the complete lifecycle
    from request creation to journey completion.
    """
    
    # System Lifecycle Events
    SIMULATION_START = "simulation.start"
    SIMULATION_END = "simulation.end"
    SIMULATION_ERROR = "simulation.error"
    WARMUP_COMPLETED = "simulation.warmup.completed"
    TIMESTEP_STARTED = "simulation.timestep.started"
    TIMESTEP_COMPLETED = "simulation.timestep.completed"
    
    # Request Creation & Validation
    REQUEST_RECEIVED = "request.received"
    REQUEST_VALIDATED = "request.validated"
    REQUEST_VALIDATION_FAILED = "request.validation.failed"
    REQUEST_CANCELLED = "request.cancelled"
    REQUEST_REJECTED = "request.rejected"
    REQUEST_EXPIRED = "request.expired"
    REQUEST_ASSIGNED = "request.assigned"

    # Matching Process
    MATCH_REQUEST_TO_VEHICLE = "match.request.to.vehicle"
    MATCH_REQUEST_TO_VEHICLE_FAILED = "match.request.to.vehicle.failed"

    # Passenger Journey Events
    START_PASSENGER_JOURNEY = "passenger.journey.started"
    PASSENGER_WALKING_TO_PICKUP = "passenger.walking.pickup.started"
    PASSENGER_ARRIVED_PICKUP = "passenger.walking.pickup.completed"
    PASSENGER_WAITING = "passenger.waiting.started"
    PASSENGER_NO_SHOW = "passenger.no_show"
    PASSENGER_BOARDING_COMPLETED = "passenger.boarding.completed"
    PASSENGER_IN_VEHICLE = "passenger.in_vehicle"
    PASSENGER_ALIGHTING_COMPLETED = "passenger.alighting.completed"
    PASSENGER_WALKING_TO_DESTINATION = "passenger.walking.destination.started"
    PASSENGER_ARRIVED_DESTINATION = "passenger.walking.destination.completed"
    PASSENGER_JOURNEY_COMPLETED = "passenger.journey.completed"
    CHECK_BOARDING_STATUS = "passenger.boarding.check_status"
    
    # Service Level Events
    EXCESS_WAIT_TIME = "passenger.wait_time.exceeded"
    EXCESS_RIDE_TIME = "passenger.ride_time.exceeded"
    SERVICE_LEVEL_VIOLATION = "service.level.violation"
    
    # Vehicle Operations
    VEHICLE_CREATED = "vehicle.created"
    VEHICLE_ACTIVATED = "vehicle.activated"
    VEHICLE_ASSIGNMENT = "vehicle.assignment"
    VEHICLE_ARRIVED_STOP = "vehicle.arrived.stop"
    VEHICLE_BOARDING_STARTED = "vehicle.boarding.started"
    VEHICLE_BOARDING_COMPLETED = "vehicle.boarding.completed"
    VEHICLE_ALIGHTING_STARTED = "vehicle.alighting.started"
    VEHICLE_ALIGHTING_COMPLETED = "vehicle.alighting.completed"
    VEHICLE_IDLE = "vehicle.idle"
    VEHICLE_BUSY = "vehicle.busy"
    VEHICLE_AT_CAPACITY = "vehicle.at_capacity"
    VEHICLE_DWELL_TIME_START = "vehicle.dwell.started"
    VEHICLE_DWELL_TIME_VIOLATION = "vehicle.dwell.violation"
    VEHICLE_DISPATCH_REQUEST = "vehicle.dispatch.request"
    VEHICLE_REROUTE_REQUEST = "vehicle.reroute.request"
    VEHICLE_EN_ROUTE = "vehicle.en_route"
    VEHICLE_AT_STOP = "vehicle.at_stop"
    VEHICLE_WAIT_TIMEOUT = "vehicle.wait.timeout"
    VEHICLE_REBALANCING_REQUIRED = "vehicle.rebalancing.required"
    VEHICLE_SERVICE_KPI_VIOLATION = "vehicle.service.kpi.violation"
    VEHICLE_POSITION_UPDATE = "vehicle.position.update"
    VEHICLE_STOP_OPERATIONS_COMPLETED = "vehicle.stop.operations.completed"
    
    # Route Management
    ROUTE_CREATED = "route.created"
    ROUTE_ACTIVATION = "route.activation"
    ROUTE_SEGMENT_STARTED = "route.segment.started"
    ROUTE_SEGMENT_COMPLETED = "route.segment.completed"
    ROUTE_UPDATE_REQUEST = "route.update.request"
    ROUTE_UPDATED = "route.updated"
    ROUTE_COMPLETED = "route.completed"
    ROUTE_SEGMENT_READY_FOR_COMPLETION = "route.segment.ready.for.completion"

    # Optimization Events
    SCHEDULED_GLOBAL_OPTIMIZATION = "optimization.global.scheduled"
    GLOBAL_OPTIMIZATION_STARTED = "optimization.global.started"
    GLOBAL_OPTIMIZATION_COMPLETED = "optimization.global.completed"
    
    # Stop Management
    STOP_ACTIVATED = "stop.activated"
    STOP_DEACTIVATED = "stop.deactivated"
    STOP_CONGESTED = "stop.congested"
    STOP_CAPACITY_EXCEEDED = "stop.capacity.exceeded"
    DETERMINE_VIRTUAL_STOPS = "determine.virtual.stops"
    STOP_SELECTION_TICK = "stop.selection.tick"
    STOP_SELECTION_COMPLETED = "stop.selection.completed" 
    STOPS_UPDATED = "stops.updated"
    
    # Fleet Management
    FLEET_REBALANCING_NEEDED = "fleet.rebalancing.needed"
    FLEET_REBALANCING_STARTED = "fleet.rebalancing.started"
    FLEET_REBALANCING_COMPLETED = "fleet.rebalancing.completed"
    FLEET_CAPACITY_WARNING = "fleet.capacity.warning"
    
    # Metrics & Analysis
    METRICS_COLLECTED = "metrics.collected"
    KPI_THRESHOLD_EXCEEDED = "metrics.kpi.threshold"
    PERFORMANCE_SNAPSHOT = "metrics.performance.snapshot"

@functools.total_ordering
@dataclass(frozen=True)
class Event:
    """
    Represents a simulation event with additional fields specific to DRT operations.
    Includes support for recurring events.
    
    The comparison operators are implemented to support priority queue ordering,
    with events sorted by priority first, then timestamp.
    """
    # Required fields (no defaults)
    event_type: EventType = field(compare=False)
    priority: EventPriority = field(compare=False)
    timestamp: datetime = field(compare=False)
    
    # Optional fields (with defaults)
    id: str = field(default_factory=lambda: str(uuid.uuid4()), compare=False)
    status: EventStatus = field(default=EventStatus.PENDING, compare=False)
    
    # Entity IDs
    vehicle_id: Optional[str] = field(default=None, compare=False)
    request_id: Optional[str] = field(default=None, compare=False)
    passenger_id: Optional[str] = field(default=None, compare=False)
    route_id: Optional[str] = field(default=None, compare=False)
    stop_id: Optional[str] = field(default=None, compare=False)
    
    # Timing fields
    scheduled_time: Optional[datetime] = field(default=None, compare=False)
    actual_time: Optional[datetime] = field(default=None, compare=False)
    created_at: datetime = field(default_factory=datetime.now, compare=False)
    processed_at: Optional[datetime] = field(default=None, compare=False)
    completed_at: Optional[datetime] = field(default=None, compare=False)
    
    # Recurring event fields
    is_recurring: bool = field(default=False, compare=False)
    recurrence_interval: Optional[float] = field(default=None, compare=False)  # in seconds
    recurrence_end: Optional[datetime] = field(default=None, compare=False)
    
    # Service quality metrics
    waiting_time: Optional[float] = field(default=None, compare=False)
    ride_time: Optional[float] = field(default=None, compare=False)
    walking_distance: Optional[float] = field(default=None, compare=False)
    deviation_minutes: Optional[float] = field(default=None, compare=False)
    service_metrics: Dict[str, float] = field(default_factory=dict, compare=False)
    
    # Additional data
    location: Optional[Dict[str, float]] = field(default=None, compare=False)  # lat/lon or x/y coordinates
    data: Dict[str, Any] = field(default_factory=dict, compare=False)
    metadata: Dict[str, Any] = field(default_factory=dict, compare=False)
    
    def __post_init__(self):
        """Convert mutable dictionaries to immutable MappingProxyTypes"""
        # We need to use object.__setattr__ because the dataclass is frozen
        object.__setattr__(self, 'data', MappingProxyType(self.data))
        object.__setattr__(self, 'metadata', MappingProxyType(self.metadata))
        object.__setattr__(self, 'service_metrics', MappingProxyType(self.service_metrics))
        object.__setattr__(self, 'location', MappingProxyType(self.location) if self.location else None)

    def create_next_recurrence(self) -> Optional['Event']:
        """Create the next event in the recurrence sequence."""
        if not self.is_recurring or not self.recurrence_interval:
            return None
            
        next_time = self.timestamp + timedelta(seconds=self.recurrence_interval)
        if self.recurrence_end and next_time > self.recurrence_end:
            return None
            
        # Create deep copies of the immutable mappings
        data_copy = deepcopy(dict(self.data)) if self.data else {}
        metadata_copy = deepcopy(dict(self.metadata)) if self.metadata else {}
        service_metrics_copy = deepcopy(dict(self.service_metrics)) if self.service_metrics else {}
        location_copy = deepcopy(dict(self.location)) if self.location else None
            
        return Event(
            event_type=self.event_type,
            priority=self.priority,
            timestamp=next_time,
            vehicle_id=self.vehicle_id,
            request_id=self.request_id,
            passenger_id=self.passenger_id,
            route_id=self.route_id,
            stop_id=self.stop_id,
            scheduled_time=self.scheduled_time,
            actual_time=self.actual_time,
            waiting_time=self.waiting_time,
            ride_time=self.ride_time,
            walking_distance=self.walking_distance,
            deviation_minutes=self.deviation_minutes,
            service_metrics=service_metrics_copy,
            location=location_copy,
            is_recurring=True,
            recurrence_interval=self.recurrence_interval,
            recurrence_end=self.recurrence_end,
            data=data_copy,
            metadata=metadata_copy
        )
    
    def __lt__(self, other):
        # First by timestamp
        if self.timestamp != other.timestamp:
            return self.timestamp < other.timestamp
        
        # Then by priority
        if self.priority != other.priority:
            return self.priority < other.priority
        
        # Then by ID or other criteria to ensure deterministic ordering
        return self.id < other.id
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert event to dictionary format with proper handling of MappingProxyType"""
        return {
            'id': self.id,
            'event_type': self.event_type.value,
            'priority': self.priority.value,
            'status': self.status.value,
            'timestamp': self.timestamp.isoformat(),
            'created_at': self.created_at.isoformat(),
            'processed_at': self.processed_at.isoformat() if self.processed_at else None,
            'completed_at': self.completed_at.isoformat() if self.completed_at else None,
            'vehicle_id': self.vehicle_id,
            'request_id': self.request_id,
            'passenger_id': self.passenger_id,
            'route_id': self.route_id,
            'stop_id': self.stop_id,
            'scheduled_time': self.scheduled_time.isoformat() if self.scheduled_time else None,
            'actual_time': self.actual_time.isoformat() if self.actual_time else None,
            'waiting_time': self.waiting_time,
            'ride_time': self.ride_time,
            'walking_distance': self.walking_distance,
            'deviation_minutes': self.deviation_minutes,
            'service_metrics': dict(self.service_metrics) if self.service_metrics else {},
            'location': dict(self.location) if self.location else None,
            'data': dict(self.data) if self.data else {},
            'metadata': dict(self.metadata) if self.metadata else {},
            'is_recurring': self.is_recurring,
            'recurrence_interval': self.recurrence_interval,
            'recurrence_end': self.recurrence_end.isoformat() if self.recurrence_end else None
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Event':
        """Create event from dictionary format"""
        # Convert ISO format strings back to datetime
        timestamp = datetime.fromisoformat(data['timestamp'])
        created_at = datetime.fromisoformat(data['created_at'])
        processed_at = (datetime.fromisoformat(data['processed_at']) 
                       if data.get('processed_at') else None)
        completed_at = (datetime.fromisoformat(data['completed_at'])
                       if data.get('completed_at') else None)
        scheduled_time = (datetime.fromisoformat(data['scheduled_time'])
                         if data.get('scheduled_time') else None)
        actual_time = (datetime.fromisoformat(data['actual_time'])
                      if data.get('actual_time') else None)
        recurrence_end = (datetime.fromisoformat(data['recurrence_end'])
                         if data.get('recurrence_end') else None)
        
        # Convert string values to enum types
        event_type = EventType(data['event_type']) if isinstance(data['event_type'], str) else data['event_type']
        priority = EventPriority(data['priority']) if isinstance(data['priority'], (int, str)) else data['priority']
        status = EventStatus(data['status']) if isinstance(data['status'], str) else data['status']
        
        return cls(
            id=data['id'],
            event_type=event_type,
            priority=priority,
            status=status,
            timestamp=timestamp,
            created_at=created_at,
            processed_at=processed_at,
            completed_at=completed_at,
            vehicle_id=data.get('vehicle_id'),
            request_id=data.get('request_id'),
            passenger_id=data.get('passenger_id'),
            route_id=data.get('route_id'),
            stop_id=data.get('stop_id'),
            scheduled_time=scheduled_time,
            actual_time=actual_time,
            waiting_time=data.get('waiting_time'),
            ride_time=data.get('ride_time'),
            walking_distance=data.get('walking_distance'),
            deviation_minutes=data.get('deviation_minutes'),
            service_metrics=data.get('service_metrics', {}),
            location=data.get('location'),
            data=data.get('data', {}),
            metadata=data.get('metadata', {}),
            is_recurring=data.get('is_recurring', False),
            recurrence_interval=data.get('recurrence_interval'),
            recurrence_end=recurrence_end
        )