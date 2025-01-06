# drt_sim/models/stop.py
from dataclasses import dataclass, field
from typing import Dict, Any
from enum import Enum

class StopStatus(Enum):
    ACTIVE = "active"
    INACTIVE = "inactive"
    TEMPORARY = "temporary"
    MAINTENANCE = "maintenance"

@dataclass
class Stop:
    id: str
    name: str
    location: Dict[str, float]  # e.g., {'latitude': 37.7749, 'longitude': -122.4194}
    status: StopStatus = StopStatus.INACTIVE
    congestion_level: str = 'low'  # Options: 'low', 'moderate', 'high'
    current_load: int = 0
    capacity: int = 50  # Example capacity
    capacity_exceeded: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            'id': self.id,
            'name': self.name,
            'location': self.location,
            'status': self.status.value,
            'congestion_level': self.congestion_level,
            'current_load': self.current_load,
            'capacity': self.capacity,
            'capacity_exceeded': self.capacity_exceeded,
            'metadata': self.metadata
        }