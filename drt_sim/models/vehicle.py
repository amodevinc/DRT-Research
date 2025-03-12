from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, ClassVar
from datetime import datetime
from enum import Enum
from .base import ModelBase
from .location import Location
from .route import Route
from .stop import Stop
from ..config.config import VehicleConfig
import logging
logger = logging.getLogger(__name__)

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
    distance_traveled: float = 0.0
    energy_consumption: float = 0.0
    passengers: List[str] = field(default_factory=list)
    last_updated: datetime = field(default_factory=datetime.now)
    cumulative_occupied_time: float = 0.0
    in_service_start_time: Optional[datetime] = None
    waiting_for_passengers: bool = False  # Flag to track if vehicle is waiting for passengers at a stop
    current_stop_id: Optional[str] = None  # Track which stop the vehicle is currently at
    current_route_id: Optional[str] = None  # Track the active route ID

    def __str__(self) -> str:
        """Provides a concise string representation of the vehicle state"""
        battery = f"|bat={self.battery_level:.1f}%" if self.battery_level is not None else ""
        waiting = "|waiting" if self.waiting_for_passengers else ""
        stop = f"|at_stop={self.current_stop_id[:8]}" if self.current_stop_id else ""
        return f"State[{self.status.value}|occ={self.current_occupancy}|dist={self.distance_traveled:.1f}km{battery}{waiting}{stop}]"

    def to_dict(self) -> Dict[str, Any]:
        return {
            'current_location': self.current_location.to_dict() if type(self.current_location) == Location else self.current_location,
            'status': self.status.value,
            'battery_level': self.battery_level,
            'current_occupancy': self.current_occupancy,
            'distance_traveled': self.distance_traveled,
            'energy_consumption': self.energy_consumption,
            'passengers': self.passengers,
            'last_updated': self.last_updated.isoformat(),
            'cumulative_occupied_time': self.cumulative_occupied_time,
            'in_service_start_time': self.in_service_start_time.isoformat() if self.in_service_start_time else None,
            'waiting_for_passengers': self.waiting_for_passengers,
            'current_stop_id': self.current_stop_id,
            'current_route_id': self.current_route_id
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'VehicleState':
        return cls(
            current_location=Location.from_dict(data['current_location']),
            status=VehicleStatus(data['status']),
            battery_level=data.get('battery_level'),
            current_occupancy=data['current_occupancy'],
            distance_traveled=data['distance_traveled'],
            energy_consumption=data['energy_consumption'],
            passengers=data['passengers'],
            last_updated=datetime.fromisoformat(data['last_updated']),
            cumulative_occupied_time=data.get('cumulative_occupied_time', 0.0),
            in_service_start_time=datetime.fromisoformat(data['in_service_start_time']) if data['in_service_start_time'] else None,
            waiting_for_passengers=data.get('waiting_for_passengers', False),
            current_stop_id=data.get('current_stop_id'),
            current_route_id=data.get('current_route_id')
        )

    def update_location(self, new_location: Location, distance_covered: Optional[float] = None) -> None:
        """Update vehicle location and calculate distance traveled"""
        if self.current_location:
            distance_delta = distance_covered if distance_covered is not None else 0
            self.distance_traveled += distance_delta
            
        self.current_location = new_location
        self.last_updated = datetime.now()

    def update_occupancy(self, delta: int) -> None:
        """Update vehicle occupancy when passengers board or alight"""
        self.current_occupancy = max(0, self.current_occupancy + delta)
        self.last_updated = datetime.now()

    def clone(self) -> 'VehicleState':
        """Create a deep copy of the vehicle state"""
        return VehicleState(
            current_location=self.current_location,
            status=self.status,
            battery_level=self.battery_level,
            current_occupancy=self.current_occupancy,
            distance_traveled=self.distance_traveled,
            energy_consumption=self.energy_consumption,
            passengers=self.passengers.copy(),
            last_updated=self.last_updated,
            cumulative_occupied_time=self.cumulative_occupied_time,
            in_service_start_time=self.in_service_start_time,
            waiting_for_passengers=self.waiting_for_passengers,
            current_stop_id=self.current_stop_id,
            current_route_id=self.current_route_id
        )

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
    depot_stop: Optional[Stop] = None  # The vehicle's home depot
    route_history: List[Route] = field(default_factory=list)  # Added route history
    _version: ClassVar[str] = "1.0"

    def __str__(self) -> str:
        """Provides a concise string representation of the vehicle"""
        active_route = f"|route={len(self.route_history)}" if self.route_history else ""
        depot = f"|depot={self.depot_stop.id[:8]}" if self.depot_stop else ""
        return f"Vehicle[{self.id[:8]}|{self.type.value}|cap={self.capacity}|{self.current_state}{active_route}{depot}]"

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
            'depot_stop': self.depot_stop.to_dict() if self.depot_stop else None,
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
            depot_stop=Stop.from_dict(data['depot_stop']) if data.get('depot_stop') else None,
            route_history=[Route.from_dict(route) for route in data.get('route_history', [])]
        )

    def update_state(self, updates: Dict[str, Any]) -> None:
        """Update vehicle state with partial updates"""
        current_state_dict = self.current_state.to_dict()
        current_state_dict.update(updates)
        self.current_state = VehicleState.from_dict(current_state_dict)
