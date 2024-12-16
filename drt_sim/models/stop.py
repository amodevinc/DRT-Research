# drt_sim/models/stop.py
from dataclasses import dataclass
from typing import List, Dict
from enum import Enum
from .base import ModelBase
from .location import Location

class StopType(Enum):
    VIRTUAL = "virtual"
    PHYSICAL = "physical"
    HUB = "hub"
    TRANSFER = "transfer"

class StopStatus(Enum):
    ACTIVE = "active"
    INACTIVE = "inactive"
    TEMPORARY = "temporary"
    MAINTENANCE = "maintenance"

@dataclass
class StopFacilities:
    shelter: bool
    seating: bool
    lighting: bool
    accessibility_features: List[str]
    bike_rack: bool
    monitoring: bool

@dataclass
class Stop(ModelBase):
    type: StopType
    location: Location
    status: StopStatus
    capacity: int
    facilities: Optional[StopFacilities]
    service_radius: float  # meters
    operating_hours: Dict[str, List[str]]
    safety_score: float
    accessibility_score: float
    average_waiting_time: float
    historical_usage: Dict[str, int]