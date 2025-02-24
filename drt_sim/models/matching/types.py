# drt_sim/models/matching/types.py
from dataclasses import dataclass, field
from typing import Dict, Any
from datetime import datetime
import uuid
from drt_sim.models.base import ModelBase
from drt_sim.models.route import Route
@dataclass
class Assignment(ModelBase):
    """Result of matching process."""
    request_id: str
    vehicle_id: str
    stop_assignment_id: str
    assignment_time: datetime
    route: Route
    estimated_pickup_time: datetime
    estimated_dropoff_time: datetime
    waiting_time_mins: float
    in_vehicle_time_mins: float
    assignment_score: float
    computation_time: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    id: str = field(default_factory=lambda: str(uuid.uuid4()))

    def __post_init__(self):
        """Initialize ModelBase attributes after dataclass initialization"""
        super().__init__()
        # Override id if it was provided in initialization
        if not isinstance(self.id, str):
            self.id = str(uuid.uuid4())
        
    def to_dict(self) -> Dict[str, Any]:
        """Convert assignment to dictionary format"""
        base_dict = {
            "id": self.id,
            "request_id": self.request_id,
            "vehicle_id": self.vehicle_id,
            "stop_assignment_id": self.stop_assignment_id,
            "assignment_time": self.assignment_time.isoformat(),
            "estimated_pickup_time": self.estimated_pickup_time.isoformat(),
            "estimated_dropoff_time": self.estimated_dropoff_time.isoformat(),
            "waiting_time_mins": self.waiting_time_mins,
            "in_vehicle_time_mins": self.in_vehicle_time_mins,
            "assignment_score": self.assignment_score,
            "computation_time": self.computation_time,
            "metadata": self.metadata,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat()
        }
        return base_dict

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Assignment':
        """Create assignment from dictionary format"""
        return cls(
            id=data["id"],
            request_id=data["request_id"],
            vehicle_id=data["vehicle_id"],
            stop_assignment_id=data["stop_assignment_id"],
            assignment_time=datetime.fromisoformat(data["assignment_time"]),
            estimated_pickup_time=datetime.fromisoformat(data["estimated_pickup_time"]),
            estimated_dropoff_time=datetime.fromisoformat(data["estimated_dropoff_time"]),
            waiting_time_mins=data["waiting_time_mins"],
            in_vehicle_time_mins=data["in_vehicle_time_mins"],
            assignment_score=data["assignment_score"],
            computation_time=data.get("computation_time", 0.0),
            metadata=data.get("metadata", {})
        )
