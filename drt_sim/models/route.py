# drt_sim/models/route.py
from dataclasses import dataclass, field
from typing import List, Dict, Any, ClassVar
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
class RouteStop(ModelBase):
    location: Location
    arrival_time: datetime
    departure_time: datetime
    stop_type: str  # pickup/dropoff
    passenger_id: str
    sequence: int

    def to_dict(self) -> Dict[str, Any]:
        return {
            'location': self.location.to_dict(),
            'arrival_time': self.arrival_time.isoformat(),
            'departure_time': self.departure_time.isoformat(),
            'stop_type': self.stop_type,
            'passenger_id': self.passenger_id,
            'sequence': self.sequence
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'RouteStop':
        return cls(
            location=Location.from_dict(data['location']),
            arrival_time=datetime.fromisoformat(data['arrival_time']),
            departure_time=datetime.fromisoformat(data['departure_time']),
            stop_type=data['stop_type'],
            passenger_id=data['passenger_id'],
            sequence=data['sequence']
        )

@dataclass
class Route(ModelBase):
    vehicle_id: str
    stops: List[RouteStop]
    status: RouteStatus
    total_distance: float
    total_duration: int  # minutes
    passenger_manifest: List[str]
    metrics: Dict[str, float]  # various route metrics
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    _version: ClassVar[str] = "1.0"

    def to_dict(self) -> Dict[str, Any]:
        return {
            'vehicle_id': self.vehicle_id,
            'stops': [stop.to_dict() for stop in self.stops],
            'status': self.status.value,
            'total_distance': self.total_distance,
            'total_duration': self.total_duration,
            'passenger_manifest': self.passenger_manifest,
            'metrics': self.metrics,
            'created_at': self.created_at.isoformat(),
            'updated_at': self.updated_at.isoformat(),
            '_version': self._version
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Route':
        return cls(
            vehicle_id=data['vehicle_id'],
            stops=[RouteStop.from_dict(stop) for stop in data['stops']],
            status=RouteStatus(data['status']),
            total_distance=data['total_distance'],
            total_duration=data['total_duration'],
            passenger_manifest=data['passenger_manifest'],
            metrics=data['metrics'],
            created_at=datetime.fromisoformat(data['created_at']),
            updated_at=datetime.fromisoformat(data['updated_at'])
        )

    def update_metrics(self, new_metrics: Dict[str, float]) -> None:
        """Update route metrics"""
        self.metrics.update(new_metrics)
        self.updated_at = datetime.now()