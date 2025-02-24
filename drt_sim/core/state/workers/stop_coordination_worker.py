from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, Set, Optional, Any
from drt_sim.core.state.base import StateWorker, StateContainer
from drt_sim.models.location import Location
from drt_sim.config.config import ParameterSet
import logging

logger = logging.getLogger(__name__)

@dataclass
class StopCoordinationState:
    route_stop_id: str
    expected_passenger_request_ids: Set[str] = field(default_factory=set)
    arrived_passenger_request_ids: Set[str] = field(default_factory=set)
    segment_id: Optional[str] = None
    vehicle_id: Optional[str] = None
    vehicle_location: Optional[Location] = None
    vehicle_is_at_stop: bool = False
    wait_start_time: Optional[datetime] = None
    wait_timeout_event_id: Optional[str] = None
    movement_start_time: Optional[datetime] = None
    actual_distance: Optional[float] = None

    def __str__(self):
        return (f"StopCoordinationState(route_stop_id={self.route_stop_id}, "
                f"segment_id={self.segment_id}, "
                f"expected={len(self.expected_passenger_request_ids)}, "
                f"arrived={len(self.arrived_passenger_request_ids)}, "
                f"vehicle={self.vehicle_id}, "
                f"vehicle_at_stop={self.vehicle_is_at_stop})")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "route_stop_id": self.route_stop_id,
            "segment_id": self.segment_id,
            "expected_passenger_request_ids": list(self.expected_passenger_request_ids),
            "arrived_passenger_request_ids": list(self.arrived_passenger_request_ids),
            "vehicle_id": self.vehicle_id,
            "vehicle_location": self.vehicle_location.to_dict() if self.vehicle_location else None,
            "vehicle_is_at_stop": self.vehicle_is_at_stop,
            "wait_start_time": self.wait_start_time.isoformat() if self.wait_start_time else None,
            "wait_timeout_event_id": self.wait_timeout_event_id,
            "movement_start_time": self.movement_start_time.isoformat() if self.movement_start_time else None,
            "actual_distance": self.actual_distance
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'StopCoordinationState':
        return cls(
            route_stop_id=data["route_stop_id"],
            segment_id=data["segment_id"],
            expected_passenger_request_ids=set(data["expected_passenger_request_ids"]),
            arrived_passenger_request_ids=set(data["arrived_passenger_request_ids"]),
            vehicle_id=data["vehicle_id"],
            vehicle_location=Location.from_dict(data["vehicle_location"]) if data["vehicle_location"] else None,
            vehicle_is_at_stop=data["vehicle_is_at_stop"],
            wait_start_time=datetime.fromisoformat(data["wait_start_time"]) if data["wait_start_time"] else None,
            wait_timeout_event_id=data["wait_timeout_event_id"],
            movement_start_time=datetime.fromisoformat(data["movement_start_time"]) if data["movement_start_time"] else None,
            actual_distance=data["actual_distance"]
        )

class StopCoordinationStateWorker(StateWorker):
    """Worker for managing stop coordination states"""
    
    def __init__(self):
        self._states = StateContainer[StopCoordinationState]()
        
    def initialize(self, config: Optional[ParameterSet] = None) -> None:
        logger.info("StopCoordinationStateWorker initialized")
        """Initialize worker"""
        pass
        
    def take_snapshot(self, timestamp: datetime) -> None:
        """Take snapshot of current states"""
        self._states.take_snapshot(timestamp)
        
    def get_state(self) -> Dict[str, StopCoordinationState]:
        """Get current states"""
        return self._states.items
        
    def update_state(self, state: Dict[str, StopCoordinationState]) -> None:
        """Update worker state"""
        for route_stop_id, stop_state in state.items():
            if route_stop_id in self._states.items:
                self._states.update(route_stop_id, stop_state)
            else:
                self._states.add(route_stop_id, stop_state)
                
    def restore_state(self, saved_state: Dict[str, Any]) -> None:
        """Restore worker state from saved state"""
        self.update_state(saved_state)
        
    def begin_transaction(self) -> None:
        """Begin state transaction"""
        self._states.begin_transaction()
        
    def commit_transaction(self) -> None:
        """Commit current transaction"""
        self._states.commit_transaction()
        
    def rollback_transaction(self) -> None:
        """Rollback current transaction"""
        self._states.rollback_transaction()
        
    def cleanup(self) -> None:
        """Clean up worker resources"""
        self._states = StateContainer[StopCoordinationState]()
        
    def get_stop_state(self, route_stop_id: str) -> Optional[StopCoordinationState]:
        """Get coordination state for a route stop"""
        return self._states.get(route_stop_id)
        
    def update_stop_state(self, state: StopCoordinationState) -> None:
        """Update coordination state for a route stop"""
        if state.route_stop_id in self._states.items:
            self._states.update(state.route_stop_id, state)
        else:
            self._states.add(state.route_stop_id, state)
    
    def remove_stop_state(self, route_stop_id: str) -> None:
        """Remove coordination state for a route stop"""
        self._states.remove(route_stop_id)