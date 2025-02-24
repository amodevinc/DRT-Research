from .status import SimulationStatus
from .vehicle_system_state import VehicleSystemState
from .request_system_state import RequestSystemState
from .passenger_system_state import PassengerSystemState
from .route_system_state import RouteSystemState
from .stop_system_state import StopSystemState, StopAssignmentSystemState
from .assignment_system_state import AssignmentSystemState
from ..base import ModelBase
from dataclasses import dataclass, field
from typing import Dict, List, Any
from datetime import datetime
from ..base import ModelBase


@dataclass
class SimulationState(ModelBase):
    """Comprehensive state of the DRT simulation"""
    current_time: datetime
    status: SimulationStatus
    vehicles: Dict[str, Any] = field(default_factory=dict)
    requests: Dict[str, Any] = field(default_factory=dict)
    routes: Dict[str, Any] = field(default_factory=dict)
    passengers: Dict[str, Any] = field(default_factory=dict)
    stops: Dict[str, Any] = field(default_factory=dict)
    stop_assignments: Dict[str, Any] = field(default_factory=dict)
    assignments: Dict[str, Any] = field(default_factory=dict)
    stop_coordination_states: Dict[str, Any] = field(default_factory=dict)
    metrics: Dict[str, float] = field(default_factory=dict)
    events: List[Dict[str, Any]] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert state to dictionary format"""
        return {
            "current_time": self.current_time.isoformat(),
            "status": self.status.value,
            "vehicles": self.vehicles,
            "requests": self.requests,
            "passengers": self.passengers,
            "routes": self.routes,
            "stops": self.stops,
            "stop_assignments": self.stop_assignments,
            "assignments": self.assignments,
            "stop_coordination_states": self.stop_coordination_states,
            "metrics": self.metrics,
            "events": self.events
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'SimulationState':
        """Create state from dictionary format"""
        return cls(
            current_time=datetime.fromisoformat(data["current_time"]),
            status=SimulationStatus(data["status"]),
            vehicles=data.get("vehicles", {}),
            requests=data.get("requests", {}),
            routes=data.get("routes", {}),
            passengers=data.get("passengers", {}),
            stops=data.get("stops", {}),
            stop_assignments=data.get("stop_assignments", {}),
            assignments=data.get("assignments", {}),
            stop_coordination_states=data.get("stop_coordination_states", {}),
            metrics=data.get("metrics", {}),
            events=data.get("events", [])
        )