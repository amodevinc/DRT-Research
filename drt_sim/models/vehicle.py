# drt_sim/models/vehicle.py
from dataclasses import dataclass
from typing import Optional, List
from enum import Enum
from .base import ModelBase
from .location import Location

class VehicleType(Enum):
    STANDARD = "standard"
    ELECTRIC = "electric"
    WHEELCHAIR_ACCESSIBLE = "wheelchair_accessible"
    MINIBUS = "minibus"

class VehicleStatus(Enum):
    IDLE = "idle"
    IN_SERVICE = "in_service"
    CHARGING = "charging"
    MAINTENANCE = "maintenance"
    OFF_DUTY = "off_duty"

@dataclass
class VehicleCapacity:
    seated: int
    standing: int
    wheelchair: int
    baggage: float  # in cubic meters

@dataclass
class VehicleState(ModelBase):
    vehicle_id: str
    current_location: Location
    status: VehicleStatus
    battery_level: Optional[float] = None  # For electric vehicles
    current_occupancy: int = 0
    current_route: Optional[List[Location]] = None

@dataclass
class Vehicle(ModelBase):
    type: VehicleType
    capacity: VehicleCapacity
    depot_location: Location
    registration: str
    manufacturer: str
    model: str
    year: int
    fuel_efficiency: float
    maintenance_schedule: Dict[str, datetime]
    features: List[str]
    accessibility_options: List[str]
    max_range: float  # in kilometers
    current_state: VehicleState