# drt_sim/models/vehicle.py
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, ClassVar
from datetime import datetime
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
    REBALANCING = "rebalancing"

@dataclass
class VehicleState(ModelBase):
    current_location: Location
    status: VehicleStatus
    battery_level: Optional[float] = None  # For electric vehicles
    current_occupancy: int = 0
    current_route: Optional[List[Location]] = None
    distance_traveled: float = 0.0
    energy_consumption: float = 0.0
    passengers: List[str] = field(default_factory=list)
    last_updated: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict[str, Any]:
        return {
            'current_location': self.current_location.to_dict(),
            'status': self.status.value,
            'battery_level': self.battery_level,
            'current_occupancy': self.current_occupancy,
            'current_route': [loc.to_dict() for loc in self.current_route] if self.current_route else None,
            'distance_traveled': self.distance_traveled,
            'energy_consumption': self.energy_consumption,
            'passengers': self.passengers,
            'last_updated': self.last_updated.isoformat()
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'VehicleState':
        return cls(
            current_location=Location.from_dict(data['current_location']),
            status=VehicleStatus(data['status']),
            battery_level=data.get('battery_level'),
            current_occupancy=data['current_occupancy'],
            current_route=[Location.from_dict(loc) for loc in data['current_route']] if data.get('current_route') else None,
            distance_traveled=data['distance_traveled'],
            energy_consumption=data['energy_consumption'],
            passengers=data['passengers'],
            last_updated=datetime.fromisoformat(data['last_updated'])
        )

@dataclass
class Vehicle(ModelBase):
    id: str
    type: VehicleType
    capacity: int
    depot_location: Location
    registration: str
    manufacturer: str
    model: str
    year: int
    fuel_efficiency: float
    maintenance_schedule: Dict[str, datetime]
    features: List[str]
    accessibility_options: List[str]
    max_range_km: float  # in kilometers
    current_state: VehicleState
    _version: ClassVar[str] = "1.0"

    def to_dict(self) -> Dict[str, Any]:
        return {
            'id': self.id,
            'type': self.type.value,
            'capacity': self.capacity,
            'depot_location': self.depot_location.to_dict(),
            'registration': self.registration,
            'manufacturer': self.manufacturer,
            'model': self.model,
            'year': self.year,
            'fuel_efficiency': self.fuel_efficiency,
            'maintenance_schedule': {k: v.isoformat() for k, v in self.maintenance_schedule.items()},
            'features': self.features,
            'accessibility_options': self.accessibility_options,
            'max_range_km': self.max_range_km,
            'current_state': self.current_state.to_dict(),
            '_version': self._version
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Vehicle':
        return cls(
            id=data['id'],
            type=VehicleType(data['type']),
            capacity=data['capacity'],
            depot_location=Location.from_dict(data['depot_location']),
            registration=data['registration'],
            manufacturer=data['manufacturer'],
            model=data['model'],
            year=data['year'],
            fuel_efficiency=data['fuel_efficiency'],
            maintenance_schedule={k: datetime.fromisoformat(v) for k, v in data['maintenance_schedule'].items()},
            features=data['features'],
            accessibility_options=data['accessibility_options'],
            max_range_km=data['max_range_km'],
            current_state=VehicleState.from_dict(data['current_state'])
        )

    def update_state(self, updates: Dict[str, Any]) -> None:
        """Update vehicle state with partial updates"""
        current_state_dict = self.current_state.to_dict()
        current_state_dict.update(updates)
        self.current_state = VehicleState.from_dict(current_state_dict)