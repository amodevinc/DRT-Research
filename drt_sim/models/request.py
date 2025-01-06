# drt_sim/models/request.py
from dataclasses import dataclass
from typing import Optional, List, Dict, Any, ClassVar
from enum import Enum
from .base import ModelBase
from datetime import datetime
from .location import Location
from .vehicle import VehicleType

class RequestType(Enum):
    """Type of passenger request"""
    IMMEDIATE = "immediate"
    SCHEDULED = "scheduled"
    RECURRING = "recurring"

class RequestStatus(Enum):
    """Status of a passenger request"""
    PENDING = "pending"
    ACCEPTED = "accepted"
    ASSIGNED = "assigned"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    CANCELLED = "cancelled"
    REJECTED = "rejected"
    PICKED_UP = "picked_up"
    DROPPED_OFF = "dropped_off"

@dataclass
class RequestConstraints(ModelBase):
    """Constraints for a passenger request"""
    latest_pickup_time: Optional[datetime]
    latest_dropoff_time: Optional[datetime]
    required_vehicle_type: Optional[VehicleType]
    minimum_capacity: int
    maximum_price: Optional[float]

    def to_dict(self) -> Dict[str, Any]:
        return {
            'latest_pickup_time': self.latest_pickup_time.isoformat() if self.latest_pickup_time else None,
            'latest_dropoff_time': self.latest_dropoff_time.isoformat() if self.latest_dropoff_time else None,
            'required_vehicle_type': self.required_vehicle_type.value if self.required_vehicle_type else None,
            'minimum_capacity': self.minimum_capacity,
            'maximum_price': self.maximum_price
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'RequestConstraints':
        return cls(
            latest_pickup_time=datetime.fromisoformat(data['latest_pickup_time']) if data.get('latest_pickup_time') else None,
            latest_dropoff_time=datetime.fromisoformat(data['latest_dropoff_time']) if data.get('latest_dropoff_time') else None,
            required_vehicle_type=VehicleType(data['required_vehicle_type']) if data.get('required_vehicle_type') else None,
            minimum_capacity=data['minimum_capacity'],
            maximum_price=data.get('maximum_price')
        )

@dataclass
class Request(ModelBase):
    """Representation of a passenger request for a DRT service"""
    type: RequestType
    id: str
    passenger_id: str
    pickup_location: Location
    dropoff_location: Location
    request_time: datetime
    status: RequestStatus
    constraints: Optional[RequestConstraints] = None
    assigned_vehicle_id: Optional[str] = None
    estimated_price: Optional[float] = None
    pickup_time: Optional[datetime] = None
    dropoff_time: Optional[datetime] = None
    _version: ClassVar[str] = "1.0"

    def to_dict(self) -> Dict[str, Any]:
        return {
            'type': self.type.value,
            'id': self.id,
            'passenger_id': self.passenger_id,
            'pickup_location': self.pickup_location.to_dict(),
            'dropoff_location': self.dropoff_location.to_dict(),
            'request_time': self.request_time.isoformat(),
            'status': self.status.value,
            'constraints': self.constraints.to_dict() if self.constraints else None,
            'assigned_vehicle_id': self.assigned_vehicle_id,
            'estimated_price': self.estimated_price,
            'pickup_time': self.pickup_time.isoformat() if self.pickup_time else None,
            'dropoff_time': self.dropoff_time.isoformat() if self.dropoff_time else None,
            '_version': self._version
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Request':
        return cls(
            type=RequestType(data['type']),
            id=data['id'],
            passenger_id=data['passenger_id'],
            pickup_location=Location.from_dict(data['pickup_location']),
            dropoff_location=Location.from_dict(data['dropoff_location']),
            request_time=datetime.fromisoformat(data['request_time']),
            status=RequestStatus(data['status']),
            constraints=RequestConstraints.from_dict(data['constraints']) if data.get('constraints') else None,
            assigned_vehicle_id=data.get('assigned_vehicle_id'),
            estimated_price=data.get('estimated_price'),
            pickup_time=datetime.fromisoformat(data['pickup_time']) if data.get('pickup_time') else None,
            dropoff_time=datetime.fromisoformat(data['dropoff_time']) if data.get('dropoff_time') else None
        )

