from dataclasses import dataclass, field
from typing import Dict, List, Any
from datetime import datetime
from enum import Enum
from .base import ModelBase
from .vehicle import VehicleState, VehicleStatus
from .request import Request, RequestStatus
from .passenger import PassengerState, PassengerStatus
from .route import Route, RouteStatus
from .stop import Stop, StopStatus, StopAssignment
from .matching import Assignment
class SimulationStatus(Enum):
    """Possible states of the simulation"""
    INITIALIZED = "initialized"
    WARMING_UP = "warming_up"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"
    STOPPED = "stopped"

@dataclass
class VehicleSystemState(ModelBase):
    """Represents the state of the vehicle system"""
    vehicles: Dict[str, VehicleState]
    vehicles_by_status: Dict[VehicleStatus, List[str]]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "vehicles": {
                vid: vstate.to_dict() for vid, vstate in self.vehicles.items()
            },
            "vehicles_by_status": {
                status.value: vids for status, vids in self.vehicles_by_status.items()
            }
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'VehicleSystemState':
        return cls(
            vehicles={
                vid: VehicleState.from_dict(vstate) 
                for vid, vstate in data["vehicles"].items()
            },
            vehicles_by_status={
                VehicleStatus(status): vids 
                for status, vids in data["vehicles_by_status"].items()
            }
        )

@dataclass
class RequestSystemState(ModelBase):
    """Represents the state of the request system"""
    active_requests: Dict[str, Request]
    requests_by_status: Dict[RequestStatus, List[str]]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "active_requests": {
                rid: req.to_dict() for rid, req in self.active_requests.items()
            },
            "requests_by_status": {
                status.value: rids for status, rids in self.requests_by_status.items()
            },
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'RequestSystemState':
        return cls(
            active_requests={
                rid: Request.from_dict(req) 
                for rid, req in data["active_requests"].items()
            },
            requests_by_status={
                RequestStatus(status): rids 
                for status, rids in data["requests_by_status"].items()
            },
        )


@dataclass
class PassengerSystemState(ModelBase):
    """Represents the state of the passenger system"""
    active_passengers: Dict[str, PassengerState]
    passengers_by_status: Dict[PassengerStatus, List[str]]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "active_passengers": {
                pid: pstate.to_dict() for pid, pstate in self.active_passengers.items()
            },
            "passengers_by_status": {
                status.value: pids for status, pids in self.passengers_by_status.items()
            },
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'PassengerSystemState':
        return cls(
            active_passengers={
                pid: PassengerState.from_dict(pstate) 
                for pid, pstate in data["active_passengers"].items()
            },
            passengers_by_status={
                PassengerStatus(status): pids 
                for status, pids in data["passengers_by_status"].items()
            },
        )

@dataclass
class RouteSystemState(ModelBase):
    """Represents the state of the routing system"""
    active_routes: Dict[str, Route]
    routes_by_status: Dict[RouteStatus, List[str]]
    routes_by_vehicle: Dict[str, str]  # vehicle_id -> route_id
    passenger_route_mapping: Dict[str, str]  # passenger_id -> route_id

    def to_dict(self) -> Dict[str, Any]:
        return {
            "active_routes": {
                rid: route.to_dict() for rid, route in self.active_routes.items()
            },
            "routes_by_status": {
                status.value: rids for status, rids in self.routes_by_status.items()
            },
            "routes_by_vehicle": self.routes_by_vehicle,
            "passenger_route_mapping": self.passenger_route_mapping,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'RouteSystemState':
        return cls(
            active_routes={
                rid: Route.from_dict(route) 
                for rid, route in data["active_routes"].items()
            },
            routes_by_status={
                RouteStatus(status): rids 
                for status, rids in data["routes_by_status"].items()
            },
            routes_by_vehicle=data["routes_by_vehicle"],
            passenger_route_mapping=data["passenger_route_mapping"],
        )

@dataclass
class StopSystemState(ModelBase):
    """Represents the state of the stop system"""
    stops: Dict[str, Stop]
    stops_by_status: Dict[StopStatus, List[str]]
    active_stops: List[str]
    def to_dict(self) -> Dict[str, Any]:
        return {
            "stops": {
                sid: stop.to_dict() for sid, stop in self.stops.items()
            },
            "stops_by_status": {
                status.value: sids for status, sids in self.stops_by_status.items()
            },
            "active_stops": self.active_stops,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'StopSystemState':
        return cls(
            stops={
                sid: Stop.from_dict(stop) 
                for sid, stop in data["stops"].items()
            },
            stops_by_status={
                StopStatus(status): sids 
                for status, sids in data["stops_by_status"].items()
            },
            active_stops=data["active_stops"],
        )
    
    
@dataclass
class StopAssignmentSystemState(ModelBase):
    """Represents the state of the stop assignment system"""
    assignments: Dict[str, StopAssignment]
    assignments_by_request: Dict[str, str]  # request_id -> stop_assignment_id
    assignments_by_stop: Dict[str, List[str]]  # stop_id -> List[stop_assignment_ids]
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "assignments": {
                aid: assignment.to_dict() for aid, assignment in self.assignments.items()
            },
            "assignments_by_request": self.assignments_by_request,
            "assignments_by_stop": self.assignments_by_stop,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'StopAssignmentSystemState':
        return cls(
            assignments={
                aid: StopAssignment.from_dict(assignment)
                for aid, assignment in data["assignments"].items()
            },
            assignments_by_request=data["assignments_by_request"],
            assignments_by_stop=data["assignments_by_stop"],
        )
    
@dataclass
class AssignmentSystemState(ModelBase):
    """Represents the state of the request-vehicle assignment system"""
    assignments: Dict[str, Assignment]
    assignments_by_request: Dict[str, str]  # request_id -> assignment_id
    assignments_by_vehicle: Dict[str, List[str]]  # vehicle_id -> List[assignment_ids]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "assignments": {
                aid: assignment.to_dict() for aid, assignment in self.assignments.items()
            },
            "assignments_by_request": self.assignments_by_request,
            "assignments_by_vehicle": self.assignments_by_vehicle,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'AssignmentSystemState':
        return cls(
            assignments={
                aid: Assignment.from_dict(assignment)
                for aid, assignment in data["assignments"].items()
            },
            assignments_by_request=data["assignments_by_request"],
            assignments_by_vehicle=data["assignments_by_vehicle"],
        )

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
            "assignments": self.assignments.to_dict(),  # Add this line
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