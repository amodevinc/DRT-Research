from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, ClassVar
from datetime import datetime
from enum import Enum
from .base import ModelBase
from .location import Location
from .route import Route, RouteStatus
from ..config.config import VehicleConfig
class VehicleType(Enum):
    STANDARD = "standard"
    ELECTRIC = "electric"
    WHEELCHAIR_ACCESSIBLE = "wheelchair_accessible"
    MINIBUS = "minibus"

class VehicleStatus(Enum):
    IDLE = "idle"
    IN_SERVICE = "in_service"
    AT_STOP = "at_stop"
    CHARGING = "charging"
    OFF_DUTY = "off_duty"
    REBALANCING = "rebalancing"
    INACTIVE = "inactive"

@dataclass
class VehicleState(ModelBase):
    current_location: Location
    status: VehicleStatus
    battery_level: Optional[float] = None  # For electric vehicles
    current_occupancy: int = 0
    active_route_id: Optional[str] = None
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
            'active_route_id': self.active_route_id,
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
            active_route_id=data.get('active_route_id'),
            distance_traveled=data['distance_traveled'],
            energy_consumption=data['energy_consumption'],
            passengers=data['passengers'],
            last_updated=datetime.fromisoformat(data['last_updated'])
        )

    def update_active_route_id(self, new_route_id: str) -> None:
        """Update the current route and related state information"""
        self.active_route_id = new_route_id
        self.last_updated = datetime.now()

    def update_location(self, new_location: Location) -> None:
        """Update vehicle location and calculate distance traveled"""
        if self.current_location:
            # Calculate distance between old and new location
            # This should use a proper distance calculation method
            distance_delta = 0.0  # Placeholder for actual distance calculation
            self.distance_traveled += distance_delta
            
        self.current_location = new_location
        self.last_updated = datetime.now()

    def update_occupancy(self, delta: int) -> None:
        """Update vehicle occupancy when passengers board or alight"""
        self.current_occupancy = max(0, self.current_occupancy + delta)
        self.last_updated = datetime.now()

@dataclass
class Vehicle(ModelBase):
    id: str
    type: VehicleType
    capacity: int
    config: VehicleConfig
    registration: str
    manufacturer: str
    model: str
    year: int
    fuel_efficiency: float
    maintenance_schedule: Dict[str, datetime]
    features: List[str]
    accessibility_options: List[str]
    max_range_km: float
    current_state: VehicleState
    route_history: List[Route] = field(default_factory=list)  # Added route history
    _version: ClassVar[str] = "1.0"

    def to_dict(self) -> Dict[str, Any]:
        return {
            'id': self.id,
            'type': self.type.value,
            'capacity': self.capacity,
            'config': self.config.to_dict(),
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
            'route_history': [route.to_dict() for route in self.route_history],
            '_version': self._version
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Vehicle':
        return cls(
            id=data['id'],
            type=VehicleType(data['type']),
            capacity=data['capacity'],
            config=VehicleConfig.from_dict(data['config']),
            registration=data['registration'],
            manufacturer=data['manufacturer'],
            model=data['model'],
            year=data['year'],
            fuel_efficiency=data['fuel_efficiency'],
            maintenance_schedule={k: datetime.fromisoformat(v) for k, v in data['maintenance_schedule'].items()},
            features=data['features'],
            accessibility_options=data['accessibility_options'],
            max_range_km=data['max_range_km'],
            current_state=VehicleState.from_dict(data['current_state']),
            route_history=[Route.from_dict(route) for route in data.get('route_history', [])]
        )

    def update_state(self, updates: Dict[str, Any]) -> None:
        """Update vehicle state with partial updates"""
        current_state_dict = self.current_state.to_dict()
        current_state_dict.update(updates)
        self.current_state = VehicleState.from_dict(current_state_dict)

    def get_active_route_id(self) -> Optional[str]:
        """Get the ID of the active route"""
        return self.current_state.active_route_id
