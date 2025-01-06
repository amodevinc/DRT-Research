# models/simulation.py
from enum import Enum
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Optional, Dict, Any, List

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
class SimulationState:
    """Represents the current state of the simulation"""
    current_time: datetime
    status: SimulationStatus
    metrics: Dict[str, Any] = field(default_factory=dict)
    events: List[Dict[str, Any]] = field(default_factory=list)
    vehicles: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    requests: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    routes: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert state to dictionary format"""
        return {
            "current_time": self.current_time.isoformat(),
            "status": self.status.value,
            "metrics": self.metrics or {},
            "events": self.events or [],
            "vehicles": self.vehicles or {},
            "requests": self.requests or {},
            "routes": self.routes or {}
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'SimulationState':
        """Create state from dictionary format"""
        return cls(
            current_time=datetime.fromisoformat(data["current_time"]),
            status=SimulationStatus(data["status"]),
            metrics=data.get("metrics", {}),
            events=data.get("events", []),
            vehicles=data.get("vehicles", {}),
            requests=data.get("requests", {}),
            routes=data.get("routes", {})
        )
