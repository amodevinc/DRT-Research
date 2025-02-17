from dataclasses import dataclass
from typing import Dict, List, Any
from ..base import ModelBase
from ..stop import Stop, StopStatus, StopAssignment

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