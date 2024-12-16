# drt_sim/models/request.py
from dataclasses import dataclass
from typing import Optional, List
from enum import Enum
from .base import ModelBase
from datetime import datetime
from .location import Location
from .vehicle import VehicleType

class RequestType(Enum):
    IMMEDIATE = "immediate"
    SCHEDULED = "scheduled"
    RECURRING = "recurring"

class RequestStatus(Enum):
    PENDING = "pending"
    ACCEPTED = "accepted"
    ASSIGNED = "assigned"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    CANCELLED = "cancelled"
    REJECTED = "rejected"

@dataclass
class RequestConstraints:
    latest_pickup_time: Optional[datetime]
    latest_dropoff_time: Optional[datetime]
    required_vehicle_type: Optional[VehicleType]
    minimum_capacity: int
    maximum_price: Optional[float]

@dataclass
class Request(ModelBase):
    type: RequestType
    passenger_id: str
    pickup_location: Location
    dropoff_location: Location
    requested_time: datetime
    constraints: RequestConstraints
    status: RequestStatus
    assigned_vehicle_id: Optional[str] = None
    estimated_price: Optional[float] = None