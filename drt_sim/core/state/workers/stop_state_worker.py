from typing import Dict, Any, Optional, List
from datetime import datetime, timedelta
import uuid

from drt_sim.core.state.base import StateWorker, StateContainer
from drt_sim.models.stop import Stop, StopStatus, StopType
from drt_sim.core.logging_config import setup_logger
from collections import defaultdict
from drt_sim.models.simulation import StopSystemState
from drt_sim.models.location import Location
class StopStateWorker(StateWorker):
    """Manages state for system stops"""
    
    def __init__(self):
        self.stops = StateContainer[Stop]()
        self.initialized = False
        self.logger = setup_logger(__name__)
        self.stop_usage_history: Dict[str, List[datetime]] = {}
        
    def initialize(self, config: Optional[Any] = None) -> None:
        """Initialize stop state worker"""
        self.initialized = True
        depot_locations = config.get("depot_locations", [])
        for depot_location in depot_locations:
            stop = Stop(
                id=str(uuid.uuid4()),
                location=Location(lat=depot_location[1], lon=depot_location[0]),
                status=StopStatus.ACTIVE,
                type=StopType.DEPOT
            )
            self.stops.add(stop.id, stop)

        self.logger.info("Stop state worker initialized")

    def update_stops_bulk(self, 
                         updated_stops: List[Stop], 
                         metadata: Optional[Dict[str, Any]] = None) -> None:
        """
        Update multiple stops at once, handling additions, removals, and modifications.
        
        Args:
            updated_stops: List of stops with their new states
            metadata: Optional metadata about the update operation
        """
        if not self.initialized:
            raise RuntimeError("Worker not initialized")
            
        try:
            current_stop_ids = set(self.stops.items.keys())
            updated_stop_ids = {stop.id for stop in updated_stops}
            
            # Handle removals (stops that exist but aren't in updated list)
            for stop_id in current_stop_ids - updated_stop_ids:
                self._deactivate_stop(stop_id)
            
            # Handle updates and additions
            for stop in updated_stops:
                if stop.id in self.stops.items:
                    self._update_stop(stop, metadata)
                else:
                    self._add_stop(stop, metadata)
                    
            self.logger.info(f"Bulk updated {len(updated_stops)} stops")
            
        except Exception as e:
            self.logger.error(f"Failed to perform bulk stop update: {str(e)}")
            raise

    def get_recent_requests(self, time_window: timedelta) -> Dict[str, int]:
        """
        Get recent usage counts for stops within the specified time window.
        
        Args:
            time_window: Time window to consider
            
        Returns:
            Dictionary mapping stop IDs to their usage counts
        """
        current_time = datetime.now()
        cutoff_time = current_time - time_window
        
        usage_counts = {}
        for stop_id, timestamps in self.stop_usage_history.items():
            # Filter timestamps within window
            recent_usage = sum(1 for t in timestamps if t >= cutoff_time)
            if recent_usage > 0:
                usage_counts[stop_id] = recent_usage
                
        return usage_counts
    
    def update_stop_status(self, stop_id: str, status: StopStatus, metadata: Optional[Dict[str, Any]] = None) -> None:
        """Update the status of a stop"""
        try:
            stop = self.get_stop(stop_id)
            if not stop:
                raise ValueError(f"Stop {stop_id} not found")
            stop.status = status
            if metadata:
                stop.metadata.update(metadata)
            self._update_stop(stop_id, stop)
        except Exception as e:
            self.logger.error(f"Failed to update stop status: {str(e)}")
            raise
    
    def create_new_stop(self, stop: Stop) -> None:
        """Create a new stop"""
        self._add_stop(stop)

    def get_active_stops(self) -> List[Stop]:
        """Get all active stops"""
        return [
            stop for stop in self.stops.items.values()
            if stop.status == StopStatus.ACTIVE and stop.type != StopType.DEPOT
        ]

    def get_all_stops(self) -> List[Stop]:
        """Get all stops regardless of status"""
        return list(self.stops.items.values())

    def _add_stop(self, stop: Stop, metadata: Optional[Dict[str, Any]] = None) -> None:
        """Add a new stop to the system"""
        self.stops.add(stop.id, stop)
            
        if metadata:
            stop.metadata.update(metadata)
            
        self.stop_usage_history[stop.id] = []

    def _update_stop(self, stop: Stop, metadata: Optional[Dict[str, Any]] = None) -> None:
        """Update an existing stop"""
        old_stop = self.stops.get(stop.id)
        if not old_stop:
            raise ValueError(f"Stop {stop.id} not found")
            
        # Update stop data
        self.stops.update(stop.id, stop)
        if metadata:
            stop.metadata.update(metadata)

    def _deactivate_stop(self, stop_id: str) -> None:
        """Deactivate a stop"""
        stop = self.stops.get(stop_id)
        if not stop:
            raise ValueError(f"Stop {stop_id} not found")
            
        if stop.status == StopStatus.ACTIVE:
            stop.status = StopStatus.INACTIVE

    def record_stop_usage(self, stop_id: str, timestamp: datetime) -> None:
        """Record usage of a stop"""
        if stop_id in self.stop_usage_history:
            self.stop_usage_history[stop_id].append(timestamp)

    def get_stop(self, stop_id: str) -> Optional[Stop]:
        """
        Get a stop by its ID.
        
        Args:
            stop_id: ID of the stop to retrieve
            
        Returns:
            Stop object if found, None otherwise
        """
        return self.stops.get(stop_id)
    
    def get_depot_stops(self) -> List[Stop]:
        """Get all depot stops"""
        return [
            stop for stop in self.stops.items.values()
            if stop.type == StopType.DEPOT
        ]
    
    def get_physical_stops(self) -> List[Stop]:
        """Get all physical stops"""
        return [
            stop for stop in self.stops.items.values()
            if stop.type == StopType.PHYSICAL
        ]

    def take_snapshot(self, timestamp: datetime) -> None:
        """Take snapshot of stop states"""
        if not self.initialized:
            raise RuntimeError("Worker not initialized")
        self.stops.take_snapshot(timestamp)

    def get_state(self) -> StopSystemState:
        """Get current state of the stop system"""
        if not self.initialized:
            raise RuntimeError("Worker not initialized")
            
        try:
            # Get stops by status
            stops_by_status = defaultdict(list)
            for sid, stop in self.stops.items.items():
                stops_by_status[stop.status].append(sid)
            
            # Get active and congested stops
            active_stops = [
                sid for sid, stop in self.stops.items.items()
                if stop.status == StopStatus.ACTIVE
            ]
            
            return StopSystemState(
                stops=self.stops.items,
                stops_by_status=dict(stops_by_status),
                active_stops=active_stops,
            )
            
        except Exception as e:
            self.logger.error(f"Failed to get stop system state: {str(e)}")
            raise

    def update_state(self, state: StopSystemState) -> None:
        """Update stop system state"""
        if not self.initialized:
            raise RuntimeError("Worker not initialized")
            
        try:
            # Update stops
            for stop_id, stop in state.stops.items():
                if stop_id in self.stops.items:
                    self.stops.update(stop_id, stop)
                else:
                    self.stops.add(stop_id, stop)
            
            self.logger.info("Stop system state updated successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to update stop system state: {str(e)}")
            raise

    def restore_state(self, saved_state: Dict[str, Any]) -> None:
        """Restore stop states from a saved StopSystemState"""
        try:
            # Reset current state
            self.stops = StateContainer[Stop]()
            self.stop_usage_history = {}
            
            # Restore using update_state
            self.update_state(saved_state)
            
            self.logger.debug("Restored stop system state")

        except Exception as e:
            self.logger.error(f"Error restoring stop system state: {str(e)}")
            raise
            
    def begin_transaction(self) -> None:
        """Begin state transaction"""
        self.stops.begin_transaction()

    def commit_transaction(self) -> None:
        """Commit current transaction"""
        self.stops.commit_transaction()

    def rollback_transaction(self) -> None:
        """Rollback current transaction"""
        self.stops.rollback_transaction()

    def cleanup(self) -> None:
        """Clean up resources"""
        self.stops.clear_history()
        self.stop_usage_history.clear()
        self.initialized = False
        self.logger.info("Stop state worker cleaned up")