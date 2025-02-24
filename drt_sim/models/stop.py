from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional
from enum import Enum
from drt_sim.models.location import Location
from drt_sim.models.base import ModelBase
import uuid
from datetime import datetime

class StopStatus(Enum):
    ACTIVE = "active"
    INACTIVE = "inactive"
    TEMPORARY = "temporary"
    MAINTENANCE = "maintenance"
    CONGESTED = "congested"

class StopType(Enum):
    PHYSICAL = "physical"
    VIRTUAL = "virtual"
    DEPOT = "depot"

class StopPurpose(Enum):
    PICKUP = "pickup"
    DROPOFF = "dropoff"
    DEPOT = "depot"
    
class PassengerStopOperation(Enum):
    BOARDING = "boarding"
    ALIGHTING = "alighting"

@dataclass
class PassengerOperation:
    """Tracks an individual passenger's operation at a stop"""
    passenger_id: str
    request_id: str
    operation_type: PassengerStopOperation
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    status: str = "pending"  # pending, in_progress, completed

    def __str__(self) -> str:
        """Provides a concise string representation of the passenger operation"""
        duration = f"|dur={(self.end_time - self.start_time).total_seconds():.1f}s" if (self.start_time and self.end_time) else ""
        return f"Op[{self.passenger_id[:8]}|{self.operation_type.value}|{self.status}{duration}]"

@dataclass
class StopOperation:
    """Represents all operations at a stop for a vehicle"""
    stop_id: str
    vehicle_id: str
    start_time: datetime
    passenger_operations: Dict[str, PassengerOperation] = field(default_factory=dict)  # passenger_id -> operation
    status: str = "pending"  # pending, in_progress, completed
    dwell_time: int = 0

    def __str__(self) -> str:
        """Provides a concise string representation of the stop operation"""
        ops_count = len(self.passenger_operations)
        return f"StopOp[{self.stop_id[:8]}|veh={self.vehicle_id[:8]}|ops={ops_count}|{self.status}|dwell={self.dwell_time}s]"

@dataclass
class Stop(ModelBase):
    location: Location
    type: StopType = StopType.VIRTUAL
    status: StopStatus = StopStatus.INACTIVE
    congestion_level: str = 'low'  # Options: 'low', 'moderate', 'high'
    current_load: int = 0
    capacity: int = 10  # Example capacity
    capacity_exceeded: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    
    def __post_init__(self):
        """Initialize ModelBase attributes after dataclass initialization"""
        self.created_at: datetime = datetime.now()
        self.updated_at: datetime = datetime.now()

    def to_dict(self) -> Dict[str, Any]:
        return {
            'id': self.id,
            'location': self.location.to_dict(),
            'type': self.type.value,
            'status': self.status.value,
            'congestion_level': self.congestion_level,
            'current_load': self.current_load,
            'capacity': self.capacity,
            'capacity_exceeded': self.capacity_exceeded,
            'metadata': self.metadata,
            'created_at': self.created_at.isoformat(),
            'updated_at': self.updated_at.isoformat()
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Stop':
        return cls(
            id=data['id'],
            location=Location.from_dict(data['location']),
            type=StopType(data['type']),
            status=StopStatus(data['status']),
            congestion_level=data['congestion_level'],
            current_load=data['current_load'],
            capacity=data['capacity'],
            capacity_exceeded=data['capacity_exceeded'],
            metadata=data['metadata']
        )

    def __str__(self) -> str:
        """Provides a concise string representation of the stop"""
        load_status = "!" if self.capacity_exceeded else ""
        return f"Stop[{self.id[:8]}|{self.type.value}|{self.status.value}|load={self.current_load}/{self.capacity}{load_status}]"

@dataclass
class StopAssignment(ModelBase):
    """Result of stop assignment process."""
    request_id: str
    origin_stop: Stop
    destination_stop: Stop
    walking_distance_origin: float
    walking_distance_destination: float
    walking_time_origin: float
    walking_time_destination: float
    expected_passenger_origin_stop_arrival_time: datetime
    total_score: float
    alternative_origins: List[Stop]
    alternative_destinations: List[Stop]
    assignment_time: datetime
    metadata: Dict[str, Any] = field(default_factory=dict)
    id: str = field(default_factory=lambda: str(uuid.uuid4()))

    def __post_init__(self):
        """Initialize ModelBase attributes after dataclass initialization"""
        super().__init__()
        # Override id if it was provided in initialization
        if not isinstance(self.id, str):
            self.id = str(uuid.uuid4())

    def to_dict(self) -> Dict[str, Any]:
        """Convert StopAssignment to dictionary representation."""
        return {
            'id': self.id,
            'request_id': self.request_id,
            'origin_stop': self.origin_stop.to_dict(),
            'destination_stop': self.destination_stop.to_dict(),
            'walking_distance_origin': self.walking_distance_origin,
            'walking_distance_destination': self.walking_distance_destination,
            'walking_time_origin': self.walking_time_origin,
            'walking_time_destination': self.walking_time_destination,
            'expected_passenger_origin_stop_arrival_time': self.expected_passenger_origin_stop_arrival_time.isoformat() if self.expected_passenger_origin_stop_arrival_time else None,
            'total_score': self.total_score,
            'alternative_origins': [stop.to_dict() for stop in self.alternative_origins],
            'alternative_destinations': [stop.to_dict() for stop in self.alternative_destinations],
            'assignment_time': self.assignment_time.isoformat(),
            'metadata': self.metadata,
            'created_at': self.created_at.isoformat(),
            'updated_at': self.updated_at.isoformat()
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'StopAssignment':
        """Create a StopAssignment instance from a dictionary."""
        return cls(
            id=data.get('id'),
            request_id=data['request_id'],
            origin_stop=Stop.from_dict(data['origin_stop']),
            destination_stop=Stop.from_dict(data['destination_stop']),
            walking_distance_origin=data['walking_distance_origin'],
            walking_distance_destination=data['walking_distance_destination'],
            walking_time_origin=data['walking_time_origin'],
            walking_time_destination=data['walking_time_destination'],
            total_score=data['total_score'],
            alternative_origins=[Stop.from_dict(stop) for stop in data['alternative_origins']],
            alternative_destinations=[Stop.from_dict(stop) for stop in data['alternative_destinations']],
            assignment_time=datetime.fromisoformat(data['assignment_time']),
            expected_passenger_origin_stop_arrival_time=datetime.fromisoformat(data['expected_passenger_origin_stop_arrival_time']) if data['expected_passenger_origin_stop_arrival_time'] else None,
            metadata=data.get('metadata', {})
        )

    def __str__(self) -> str:
        """Provides a concise string representation of the stop assignment"""
        walk_o = f"{self.walking_distance_origin:.0f}m/{self.walking_time_origin:.0f}s"
        walk_d = f"{self.walking_distance_destination:.0f}m/{self.walking_time_destination:.0f}s"
        return f"Assign[{self.id[:8]}|req={self.request_id[:8]}|walk_o={walk_o}|walk_d={walk_d}|score={self.total_score:.2f}]"