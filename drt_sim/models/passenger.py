# drt_sim/models/passenger.py
from dataclasses import dataclass
from typing import Optional, List
from enum import Enum
from datetime import datetime
from .base import ModelBase
from .location import Location
from .vehicle import VehicleType

class PassengerType(Enum):
    REGULAR = "regular"
    ELDERLY = "elderly"
    STUDENT = "student"
    WHEELCHAIR = "wheelchair"
    CHILD = "child"

class PassengerStatus(Enum):
    WAITING = "waiting"
    PICKED_UP = "picked_up"
    IN_TRANSIT = "in_transit"
    COMPLETED = "completed"
    CANCELLED = "cancelled"

@dataclass
class PassengerPreferences:
    max_wait_time: int  # minutes
    max_detour_time: int  # minutes
    preferred_vehicle_type: Optional[VehicleType]
    accessibility_needs: List[str]
    baggage: float  # in cubic meters
    sharing_preference: bool  # whether willing to share rides

@dataclass
class Passenger(ModelBase):
    type: PassengerType
    pickup_location: Location
    dropoff_location: Location
    requested_pickup_time: datetime
    preferences: PassengerPreferences
    status: PassengerStatus
    assigned_vehicle_id: Optional[str] = None