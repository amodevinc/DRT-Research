from dataclasses import dataclass
from typing import Dict, List, Any
from ..base import ModelBase
from ..matching import Assignment

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