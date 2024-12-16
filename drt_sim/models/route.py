# drt_sim/models/route.py
from dataclasses import dataclass
from typing import List, Dict
from enum import Enum
from .base import ModelBase
from datetime import datetime
from .location import Location

class RouteStatus(Enum):
    PLANNED = "planned"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    CANCELLED = "cancelled"

@dataclass
class RouteStop:
    location: Location
    arrival_time: datetime
    departure_time: datetime
    stop_type: str  # pickup/dropoff
    passenger_id: str
    sequence: int

@dataclass
class Route(ModelBase):
    vehicle_id: str
    stops: List[RouteStop]
    status: RouteStatus
    total_distance: float
    total_duration: int  # minutes
    passenger_manifest: List[str]
    metrics: Dict[str, float]  # various route metrics