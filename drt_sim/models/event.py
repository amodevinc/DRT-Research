from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Dict, Any, Optional
import uuid

class EventPriority(Enum):
    """Priority levels for events"""
    CRITICAL = 0  # System-critical events (crashes, errors)
    HIGH = 1      # Time-sensitive operations (pickups, dropoffs)
    NORMAL = 2    # Standard operations (route updates, status changes)
    LOW = 3       # Background operations (metrics collection, logging)
    
class EventStatus(Enum):
    """Status of an event"""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

class EventType(Enum):
    """
    Comprehensive event types for DRT simulation covering passenger journeys,
    vehicle operations, dispatch, routing, and scheduling.
    """
    
    # System Events
    SIMULATION_START = "simulation.start"
    SIMULATION_END = "simulation.end"
    SIMULATION_ERROR = "simulation.error"
    WARMUP_COMPLETED = "warmup.completed"
    
    # Request Lifecycle
    REQUEST_CREATED = "request.created"
    REQUEST_VALIDATED = "request.validated"
    REQUEST_REJECTED = "request.rejected"
    REQUEST_CANCELLED = "request.cancelled"
    REQUEST_EXPIRED = "request.expired"
    REQUEST_ASSIGNED = "request.assigned"
    REQUEST_REASSIGNED = "request.reassigned"
    REQUEST_NO_VEHICLE = "request.no_vehicle"
    
    # Passenger Journey - Access
    PASSENGER_WALKING_TO_PICKUP = "passenger.walking.to_pickup"
    PASSENGER_ARRIVED_AT_PICKUP_STOP = "passenger.arrived.pickup_stop"
    PASSENGER_WAITING_FOR_VEHICLE = "passenger.waiting.pickup"
    
    # Passenger Journey - Service
    REQUEST_PICKUP_STARTED = "request.pickup.started"
    REQUEST_PICKUP_COMPLETED = "request.pickup.completed"
    PASSENGER_IN_VEHICLE = "passenger.in_vehicle"
    PASSENGER_DETOUR_STARTED = "passenger.detour.started"
    PASSENGER_DETOUR_ENDED = "passenger.detour.ended"
    REQUEST_DROPOFF_STARTED = "request.dropoff.started"
    REQUEST_DROPOFF_COMPLETED = "request.dropoff.completed"
    
    # Passenger Journey - Egress
    PASSENGER_ARRIVED_AT_DESTINATION_STOP = "passenger.arrived.destination_stop"
    PASSENGER_WALKING_TO_DESTINATION = "passenger.walking.to_dest"
    PASSENGER_ARRIVED_AT_DESTINATION = "passenger.arrived.dest"
    
    # Vehicle States
    VEHICLE_CREATED = "vehicle.created"
    VEHICLE_ACTIVATED = "vehicle.activated"
    VEHICLE_ASSIGNED = "vehicle.assigned"
    VEHICLE_IDLE = "vehicle.idle"
    VEHICLE_BUSY = "vehicle.busy"
    VEHICLE_AT_CAPACITY = "vehicle.at_capacity"
    VEHICLE_AVAILABLE = "vehicle.available"
    VEHICLE_BREAKDOWN = "vehicle.breakdown"
    
    # Vehicle Movement
    VEHICLE_DEPARTED = "vehicle.departed"
    VEHICLE_ARRIVED = "vehicle.arrived"
    VEHICLE_REROUTED = "vehicle.rerouted"
    VEHICLE_STOP_REACHED = "vehicle.stop.reached"
    VEHICLE_ROUTE_STARTED = "vehicle.route.started"
    VEHICLE_ROUTE_COMPLETED = "vehicle.route.completed"
    
    # Dispatch Events
    DISPATCH_REQUESTED = "dispatch.requested"
    DISPATCH_SUCCEEDED = "dispatch.succeeded"
    DISPATCH_FAILED = "dispatch.failed"
    DISPATCH_OPTIMIZATION_STARTED = "dispatch.optimization.started"
    DISPATCH_OPTIMIZATION_COMPLETED = "dispatch.optimization.completed"
    DISPATCH_BATCH_STARTED = "dispatch.batch.started"
    DISPATCH_BATCH_COMPLETED = "dispatch.batch.completed"
    
    # Route Events
    ROUTE_CREATED = "route.created"
    ROUTE_UPDATED = "route.updated"
    ROUTE_OPTIMIZED = "route.optimized"
    ROUTE_CANCELLED = "route.cancelled"
    ROUTE_COMPLETED = "route.completed"
    ROUTE_DELAYED = "route.delayed"
    ROUTE_DETOUR_ADDED = "route.detour.added"
    ROUTE_STOPS_RESEQUENCED = "route.stops.resequenced"
    
    # Schedule Events
    SCHEDULE_CREATED = "schedule.created"
    SCHEDULE_UPDATED = "schedule.updated"
    SCHEDULE_OPTIMIZED = "schedule.optimized"
    SCHEDULE_VIOLATED = "schedule.violated"
    SCHEDULE_BUFFER_ADJUSTED = "schedule.buffer.adjusted"
    
    # Time Window Events
    PICKUP_WINDOW_STARTED = "time.pickup_window.start"
    PICKUP_WINDOW_ENDED = "time.pickup_window.end"
    DELIVERY_WINDOW_STARTED = "time.delivery_window.start"
    DELIVERY_WINDOW_ENDED = "time.delivery_window.end"
    
    # Service Quality Events
    EXCESS_WAIT_TIME = "service.excess_wait_time"
    EXCESS_RIDE_TIME = "service.excess_ride_time"
    MISSED_PICKUP_WINDOW = "service.missed_pickup_window"
    MISSED_DELIVERY_WINDOW = "service.missed_delivery_window"
    
    # Fleet Management
    FLEET_REBALANCING_NEEDED = "fleet.rebalancing.needed"
    FLEET_REBALANCING_STARTED = "fleet.rebalancing.started"
    FLEET_REBALANCING_COMPLETED = "fleet.rebalancing.completed"
    FLEET_CAPACITY_WARNING = "fleet.capacity.warning"
    
    # Stop Events
    STOP_ACTIVATED = "stop.activated"
    STOP_DEACTIVATED = "stop.deactivated"
    STOP_CONGESTED = "stop.congested"
    STOP_CAPACITY_EXCEEDED = "stop.capacity.exceeded"
    
    # Demand Events
    DEMAND_ZONE_SATURATION = "demand.zone.saturation"
    DEMAND_SURGE_DETECTED = "demand.surge.detected"
    
    # Metrics Collection
    METRICS_COLLECTED = "metrics.collected"
    METRICS_THRESHOLD_EXCEEDED = "metrics.threshold.exceeded"
    SERVICE_LEVEL_VIOLATED = "metrics.service_level.violated"

@dataclass(order=True)
class Event:
    """
    Represents a simulation event.
    
    The comparison operators are implemented to support priority queue ordering,
    with events sorted by priority first, then timestamp.
    """
    # Required fields (no defaults)
    event_type: EventType = field(compare=False)
    priority: EventPriority = field(compare=True)
    timestamp: datetime = field(compare=True)
    
    # Optional fields (with defaults)
    id: str = field(default_factory=lambda: str(uuid.uuid4()), compare=False)
    status: EventStatus = field(default=EventStatus.PENDING, compare=False)
    data: Dict[str, Any] = field(default_factory=dict, compare=False)
    metadata: Dict[str, Any] = field(default_factory=dict, compare=False)
    created_at: datetime = field(default_factory=datetime.now, compare=False)
    processed_at: Optional[datetime] = field(default=None, compare=False)
    completed_at: Optional[datetime] = field(default=None, compare=False)
    vehicle_id: Optional[str] = field(default=None, compare=False)
    request_id: Optional[str] = field(default=None, compare=False)
    passenger_id: Optional[str] = field(default=None, compare=False)
    route_id: Optional[str] = field(default=None, compare=False)
    
    def __post_init__(self):
        """Validate event after initialization"""
        if isinstance(self.event_type, str):
            self.event_type = EventType(self.event_type)
        if isinstance(self.priority, int):
            self.priority = EventPriority(self.priority)
        if isinstance(self.status, str):
            self.status = EventStatus(self.status)
    
    def mark_processing(self) -> None:
        """Mark event as being processed"""
        self.status = EventStatus.PROCESSING
        self.processed_at = datetime.now()
    
    def mark_completed(self) -> None:
        """Mark event as completed"""
        self.status = EventStatus.COMPLETED
        self.completed_at = datetime.now()
    
    def mark_failed(self, error: Optional[str] = None) -> None:
        """Mark event as failed"""
        self.status = EventStatus.FAILED
        self.completed_at = datetime.now()
        if error:
            self.metadata['error'] = error
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert event to dictionary format"""
        return {
            'id': self.id,
            'event_type': self.event_type.value,
            'priority': self.priority.value,
            'status': self.status.value,
            'timestamp': self.timestamp.isoformat(),
            'created_at': self.created_at.isoformat(),
            'processed_at': self.processed_at.isoformat() if self.processed_at else None,
            'completed_at': self.completed_at.isoformat() if self.completed_at else None,
            'data': self.data,
            'metadata': self.metadata,
            'vehicle_id': self.vehicle_id,
            'request_id': self.request_id,
            'route_id': self.route_id
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
        
        return cls(
            id=data['id'],
            event_type=data['event_type'],
            priority=data['priority'],
            status=data['status'],
            timestamp=timestamp,
            created_at=created_at,
            processed_at=processed_at,
            completed_at=completed_at,
            data=data.get('data', {}),
            metadata=data.get('metadata', {}),
            vehicle_id=data.get('vehicle_id'),
            request_id=data.get('request_id'),
            passenger_id=data.get('passenger_id'),
            route_id=data.get('route_id')
        )