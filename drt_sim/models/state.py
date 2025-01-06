# drt_sim/models/state.py

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, Any
from enum import Enum

class StateType(Enum):
    """Types of state that can be tracked"""
    VEHICLE = "vehicle"
    REQUEST = "request"
    ROUTE = "route"
    PASSENGER = "passenger"
    SYSTEM = "system"

@dataclass
class StateSnapshot:
    """Represents a point-in-time snapshot of an entity's state"""
    entity_id: str
    state_type: StateType
    timestamp: datetime
    data: Dict[str, Any]
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class SystemState:
    """Complete system state at a point in time"""
    timestamp: datetime
    vehicles: Dict[str, Dict[str, Any]]
    requests: Dict[str, Dict[str, Any]]
    routes: Dict[str, Dict[str, Any]]
    passengers: Dict[str, Dict[str, Any]]
    metrics: Dict[str, float]
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert state to dictionary"""
        return {
            'timestamp': self.timestamp.isoformat(),
            'vehicles': self.vehicles,
            'requests': self.requests,
            'routes': self.routes,
            'passengers': self.passengers,
            'metrics': self.metrics,
            'metadata': self.metadata
        }