# drt_sim/core/state/workers/stop_worker.py

from typing import Optional, Dict, Any
from datetime import datetime
import logging

from drt_sim.core.state.base import StateContainer, StateWorker
from drt_sim.models.stop import Stop, StopStatus


class StopStateWorker(StateWorker):
    """
    Concrete implementation of StateWorker for managing stops.
    """

    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.stops_container: StateContainer[Stop] = StateContainer()

    def initialize(self, config: Optional[Any] = None) -> None:
        """
        Initialize worker with initial stop data.
        """
        try:
            # Load initial stops from config or external source
            initial_stops = config.get('initial_stops', [])
            for stop_data in initial_stops:
                stop = Stop(**stop_data)
                self.stops_container.add(stop.id, stop)
                self.logger.info(f"Initialized Stop ID: {stop.id}")

        except Exception as e:
            self.logger.error(f"Failed to initialize StopWorker: {str(e)}")
            raise

    def take_snapshot(self, timestamp: datetime) -> None:
        """
        Take a snapshot of the current stop states.
        """
        try:
            self.stops_container.take_snapshot(timestamp)
            self.logger.info(f"Snapshot taken at {timestamp}")
        except Exception as e:
            self.logger.error(f"Failed to take snapshot: {str(e)}")
            raise

    def get_metrics(self) -> Dict[str, Any]:
        """
        Get metrics related to stops.
        """
        try:
            active_stops = [stop for stop in self.stops_container.items.values() if stop.status == StopStatus.ACTIVE]
            inactive_stops = [stop for stop in self.stops_container.items.values() if stop.status == StopStatus.INACTIVE]
            congested_stops = [stop for stop in self.stops_container.items.values() if stop.congestion_level in ['high', 'moderate', 'low']]

            metrics = {
                'total_stops': len(self.stops_container.items),
                'active_stops': len(active_stops),
                'inactive_stops': len(inactive_stops),
                'congested_stops': len(congested_stops),
                'high_congestion': len([s for s in congested_stops if s.congestion_level == 'high']),
                'moderate_congestion': len([s for s in congested_stops if s.congestion_level == 'moderate']),
                'low_congestion': len([s for s in congested_stops if s.congestion_level == 'low']),
                'capacity_exceeded_stops': len([s for s in self.stops_container.items.values() if s.capacity_exceeded])
            }

            return metrics

        except Exception as e:
            self.logger.error(f"Failed to get metrics: {str(e)}")
            raise

    def get_state(self) -> Dict[str, Any]:
        """
        Get the current state of all stops.
        """
        try:
            state = {stop_id: stop.to_dict() for stop_id, stop in self.stops_container.items.items()}
            return state
        except Exception as e:
            self.logger.error(f"Failed to get state: {str(e)}")
            raise

    def update_state(self, state: Dict[str, Any]) -> None:
        """
        Update the state of stops.
        """
        try:
            for stop_id, updates in state.items():
                self.stops_container.update(stop_id, updates)
                self.logger.info(f"Updated Stop ID: {stop_id} with {updates}")
        except Exception as e:
            self.logger.error(f"Failed to update state: {str(e)}")
            raise

    def restore_state(self, saved_state: Dict[str, Any]) -> None:
        """
        Restore the state of stops from a saved snapshot.
        """
        try:
            for stop_id, stop_data in saved_state.items():
                if stop_id in self.stops_container.items:
                    self.stops_container.update(stop_id, stop_data)
                else:
                    # Add new stops if they don't exist
                    stop = Stop(**stop_data)
                    self.stops_container.add(stop.id, stop)
            self.logger.info("State restored successfully.")
        except Exception as e:
            self.logger.error(f"Failed to restore state: {str(e)}")
            raise

    def begin_transaction(self) -> None:
        """
        Begin a state transaction.
        """
        try:
            self.stops_container.begin_transaction()
            self.logger.info("State transaction started.")
        except Exception as e:
            self.logger.error(f"Failed to begin transaction: {str(e)}")
            raise

    def commit_transaction(self) -> None:
        """
        Commit the current state transaction.
        """
        try:
            self.stops_container.commit_transaction()
            self.logger.info("State transaction committed.")
        except Exception as e:
            self.logger.error(f"Failed to commit transaction: {str(e)}")
            raise

    def rollback_transaction(self) -> None:
        """
        Rollback the current state transaction.
        """
        try:
            self.stops_container.rollback_transaction()
            self.logger.info("State transaction rolled back.")
        except Exception as e:
            self.logger.error(f"Failed to rollback transaction: {str(e)}")
            raise

    def cleanup(self) -> None:
        """
        Clean up worker resources.
        """
        try:
            self.stops_container.clear_history()
            self.logger.info("StopWorker cleanup completed.")
        except Exception as e:
            self.logger.error(f"Failed to cleanup StopWorker: {str(e)}")
            raise

    # Additional helper methods if needed
    def get_stop(self, stop_id: str) -> Optional[Stop]:
        """Retrieve a stop by its ID."""
        return self.stops_container.get(stop_id)

    def update_stop_status(self, stop_id: str, status: StopStatus, metadata: Dict[str, Any]) -> None:
        """Update the status of a stop."""
        self.stops_container.update(stop_id, {'status': status, 'metadata': metadata})

    def update_stop_congestion(self, stop_id: str, congestion_level: str, metadata: Dict[str, Any]) -> None:
        """Update the congestion level of a stop."""
        self.stops_container.update(stop_id, {'congestion_level': congestion_level, 'metadata': metadata})

    def update_stop_capacity(self, stop_id: str, current_load: int, capacity: int, metadata: Dict[str, Any]) -> None:
        """Update the capacity of a stop."""
        self.stops_container.update(stop_id, {
            'current_load': current_load,
            'capacity': capacity,
            'capacity_exceeded': current_load > capacity,
            'metadata': metadata
        })
