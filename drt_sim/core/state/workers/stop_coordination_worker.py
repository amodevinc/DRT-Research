from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, Set, Optional, Any
from drt_sim.core.state.base import StateWorker, StateContainer
from drt_sim.models.location import Location
from drt_sim.models.base import ModelBase
from drt_sim.config.config import ParameterSet
import logging

logger = logging.getLogger(__name__)

@dataclass
class StopCoordinationState(ModelBase):
    """State tracking for stop coordination between passengers and vehicles"""
    route_stop_id: str
    expected_pickup_request_ids: Set[str]
    expected_dropoff_request_ids: Set[str]
    arrived_pickup_request_ids: Set[str]
    boarded_request_ids: Set[str]
    completed_dropoff_request_ids: Set[str]
    vehicle_id: Optional[str] = None
    vehicle_location: Optional[Location] = None
    vehicle_is_at_stop: bool = False
    wait_start_time: Optional[datetime] = None
    wait_timeout_event_id: Optional[str] = None
    last_vehicle_arrival_time: Optional[datetime] = None
    movement_start_time: Optional[datetime] = None
    actual_distance: Optional[float] = None
    segment_id: Optional[str] = None
    # Fields to track route changes while at stop
    pending_route_change: bool = False
    new_route_id: Optional[str] = None
    route_change_time: Optional[datetime] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation"""
        result = {
            'id': self.id,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'updated_at': self.updated_at.isoformat() if self.updated_at else None,
            'route_stop_id': self.route_stop_id,
            'expected_pickup_request_ids': list(self.expected_pickup_request_ids),
            'expected_dropoff_request_ids': list(self.expected_dropoff_request_ids),
            'arrived_pickup_request_ids': list(self.arrived_pickup_request_ids),
            'boarded_request_ids': list(self.boarded_request_ids),
            'completed_dropoff_request_ids': list(self.completed_dropoff_request_ids),
            'vehicle_id': self.vehicle_id,
            'vehicle_location': self.vehicle_location.to_dict() if self.vehicle_location else None,
            'vehicle_is_at_stop': self.vehicle_is_at_stop,
            'wait_start_time': self.wait_start_time.isoformat() if self.wait_start_time else None,
            'wait_timeout_event_id': self.wait_timeout_event_id,
            'last_vehicle_arrival_time': self.last_vehicle_arrival_time.isoformat() if self.last_vehicle_arrival_time else None,
            'movement_start_time': self.movement_start_time.isoformat() if self.movement_start_time else None,
            'actual_distance': self.actual_distance,
            'segment_id': self.segment_id,
            'pending_route_change': self.pending_route_change,
            'new_route_id': self.new_route_id,
            'route_change_time': self.route_change_time.isoformat() if self.route_change_time else None
        }
        return result
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'StopCoordinationState':
        """Create from dictionary representation"""
        state = cls(
            route_stop_id=data['route_stop_id'],
            expected_pickup_request_ids=set(data.get('expected_pickup_request_ids', [])),
            expected_dropoff_request_ids=set(data.get('expected_dropoff_request_ids', [])),
            arrived_pickup_request_ids=set(data.get('arrived_pickup_request_ids', [])),
            boarded_request_ids=set(data.get('boarded_request_ids', [])),
            completed_dropoff_request_ids=set(data.get('completed_dropoff_request_ids', [])),
            vehicle_id=data.get('vehicle_id'),
            vehicle_location=Location.from_dict(data['vehicle_location']) if data.get('vehicle_location') else None,
            vehicle_is_at_stop=data.get('vehicle_is_at_stop', False),
            wait_start_time=datetime.fromisoformat(data['wait_start_time']) if data.get('wait_start_time') else None,
            wait_timeout_event_id=data.get('wait_timeout_event_id'),
            last_vehicle_arrival_time=datetime.fromisoformat(data['last_vehicle_arrival_time']) if data.get('last_vehicle_arrival_time') else None,
            movement_start_time=datetime.fromisoformat(data['movement_start_time']) if data.get('movement_start_time') else None,
            actual_distance=data.get('actual_distance'),
            segment_id=data.get('segment_id'),
            pending_route_change=data.get('pending_route_change', False),
            new_route_id=data.get('new_route_id'),
            route_change_time=datetime.fromisoformat(data['route_change_time']) if data.get('route_change_time') else None
        )
        
        # Set ID and timestamps if present
        if 'id' in data:
            state.id = data['id']
        if 'created_at' in data and data['created_at']:
            state.created_at = datetime.fromisoformat(data['created_at'])
        if 'updated_at' in data and data['updated_at']:
            state.updated_at = datetime.fromisoformat(data['updated_at'])
            
        return state

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