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
    vehicles: VehicleSystemState
    requests: RequestSystemState
    passengers: PassengerSystemState
    routes: RouteSystemState
    stops: StopSystemState
    stop_assignments: StopAssignmentSystemState
    assignments: AssignmentSystemState
    events: List[Dict[str, Any]] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert state to dictionary format"""
        return {
            "current_time": self.current_time.isoformat(),
            "status": self.status.value,
            "vehicles": self.vehicles.to_dict(),
            "requests": self.requests.to_dict(),
            "passengers": self.passengers.to_dict(),
            "routes": self.routes.to_dict(),
            "stops": self.stops.to_dict(),
            "stop_assignments": self.stop_assignments.to_dict(),
            "assignments": self.assignments.to_dict(),
            "events": self.events
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'SimulationState':
        """Create state from dictionary format"""
        return cls(
            current_time=datetime.fromisoformat(data["current_time"]),
            status=SimulationStatus(data["status"]),
            vehicles=VehicleSystemState.from_dict(data["vehicles"]),
            requests=RequestSystemState.from_dict(data["requests"]),
            passengers=PassengerSystemState.from_dict(data["passengers"]),
            routes=RouteSystemState.from_dict(data["routes"]),
            stops=StopSystemState.from_dict(data["stops"]),
            stop_assignments=StopAssignmentSystemState.from_dict(data["stop_assignments"]),
            assignments=AssignmentSystemState.from_dict(data["assignments"]),
            events=data.get("events", [])
        )