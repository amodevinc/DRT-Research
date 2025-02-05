from typing import Optional, Dict, Any, List
from datetime import datetime
from collections import defaultdict

from drt_sim.core.state.base import StateContainer, StateWorker
from drt_sim.models.passenger import PassengerStatus, PassengerState
from drt_sim.models.simulation import PassengerSystemState
from drt_sim.models.stop import StopPurpose
from drt_sim.core.logging_config import setup_logger

logger = setup_logger(__name__)

class PassengerStateWorker(StateWorker):
    """
    Manages passenger states throughout their journey in the DRT system.
    Tracks all passenger-related states, and transitions.
    """
    
    def __init__(self):
        self.logger = setup_logger(__name__)
        self.passengers_container: StateContainer[PassengerState] = StateContainer()
        self.initialized = False

    def initialize(self, config: Optional[Any] = None) -> None:
        """Initialize worker with configuration"""
        self.initialized = True
        self.logger.info("PassengerStateWorker initialized successfully")

    def create_passenger_state(self, passenger_state: PassengerState) -> None:
        """Create initial passenger state"""
        if not self.initialized:
            raise RuntimeError("Worker not initialized")
            
        try:
            self.passengers_container.add(passenger_state.id, passenger_state)
            self.logger.info(f"Created passenger state for passenger {passenger_state.id}")
            
        except Exception as e:
            self.logger.error(f"Failed to create passenger state: {str(e)}")
            raise

    def get_passenger(self, passenger_id: str) -> Optional[PassengerState]:
        """Get passenger state by ID"""
        if not self.initialized:
            raise RuntimeError("Worker not initialized")
            
        return self.passengers_container.get(passenger_id)
    
    def get_all_passenger_ids_for_request_ids(self, request_ids: List[str]) -> List[str]:
        """Get all passenger IDs for a list of request IDs"""
        return [
            p.id for p in self.passengers_container.items.values()
            if p.request_id in request_ids
        ]
    
    def get_passengers_at_stop(
        self,
        status: PassengerStatus,
        stop_id: str,
        stop_purpose: StopPurpose = StopPurpose.PICKUP
    ) -> List[PassengerState]:
        """
        Get all passengers with specified status at a specific stop.
        
        Args:
            status: The PassengerStatus to filter by
            stop_id: The ID of the stop to check
            stop_purpose: StopPurpose enum indicating pickup or dropoff stop
            
        Returns:
            List of PassengerState objects matching the criteria
        """
        if not self.initialized:
            raise RuntimeError("Worker not initialized")
            
        def matches_stop(passenger: PassengerState) -> bool:
            if stop_purpose == StopPurpose.PICKUP:
                return (passenger.assigned_origin_stop and 
                    passenger.assigned_origin_stop.id == stop_id)
            elif stop_purpose == StopPurpose.DROPOFF:
                return (passenger.assigned_destination_stop and 
                    passenger.assigned_destination_stop.id == stop_id)            
        return [
            passenger for passenger in self.passengers_container.items.values()
            if passenger.status == status and matches_stop(passenger)
        ]

    def update_passenger_status(
        self,
        passenger_id: str,
        new_status: PassengerStatus,
        current_time: datetime,
        additional_data: Optional[Dict[str, Any]] = None
    ) -> PassengerState:
        """Update passenger status"""
        if not self.initialized:
            raise RuntimeError("Worker not initialized")
            
        try:
            passenger_state = self.passengers_container.get(passenger_id)
            if not passenger_state:
                raise ValueError(f"No state found for passenger {passenger_id}")
            
            old_status = passenger_state.status
            passenger_state.status = new_status

            self._update_passenger_state_times(passenger_state, new_status, current_time)
            
            # Update additional data if provided
            if additional_data:
                for key, value in additional_data.items():
                    if hasattr(passenger_state, key):
                        setattr(passenger_state, key, value)
                        
            # Update location if provided
            if additional_data and 'current_location' in additional_data:
                passenger_state.current_location = additional_data['current_location']
            
            self.logger.info(
                f"Updated passenger {passenger_id} status from {old_status} to {new_status}"
            )
            return passenger_state
            
        except Exception as e:
            self.logger.error(f"Failed to update passenger status: {str(e)}")
            raise

    def record_service_violation(
        self,
        passenger_id: str,
        violation_type: str,
        violation_value: float,
        violation_time: datetime
    ) -> None:
        """Record a service level violation"""
        if not self.initialized:
            raise RuntimeError("Worker not initialized")
            
        try:
            passenger_state = self.get_passenger(passenger_id)
            if passenger_state:
                violation_data = {
                    'type': violation_type,
                    'value': violation_value,
                    'time': violation_time
                }
                if not hasattr(passenger_state, 'service_violations'):
                    passenger_state.service_violations = []
                passenger_state.service_violations.append(violation_data)
                
            self.logger.info(f"Recorded service violation for passenger {passenger_id}: {violation_data}")
                
        except Exception as e:
            self.logger.error(f"Failed to record service violation: {str(e)}")
            raise

    def _update_passenger_state_times(
        self,
        passenger_state: PassengerState,
        new_status: PassengerStatus,
        current_time: datetime
    ) -> None:
        """Update timings based on status transitions"""
        if new_status == PassengerStatus.ARRIVED_AT_PICKUP:
            passenger_state.walking_to_pickup_end_time = current_time
            passenger_state.waiting_start_time = current_time
            passenger_state.walk_time_to_origin_stop = passenger_state.walking_to_pickup_end_time - passenger_state.walking_to_pickup_start_time
            
        elif new_status == PassengerStatus.IN_VEHICLE:
            passenger_state.waiting_end_time = current_time
            passenger_state.in_vehicle_start_time = current_time
            passenger_state.wait_time = passenger_state.waiting_end_time - passenger_state.waiting_start_time
        elif new_status == PassengerStatus.WALKING_TO_DESTINATION:
            passenger_state.in_vehicle_end_time = current_time
            passenger_state.walking_to_destination_start_time = current_time
            passenger_state.in_vehicle_time = passenger_state.in_vehicle_end_time - passenger_state.in_vehicle_start_time
        elif new_status == PassengerStatus.ARRIVED_AT_DESTINATION:
            passenger_state.walking_to_destination_end_time = current_time
            passenger_state.walk_time_from_destination_stop = passenger_state.walking_to_destination_end_time - passenger_state.walking_to_destination_start_time
            passenger_state.total_journey_time = passenger_state.in_vehicle_time + passenger_state.wait_time + passenger_state.walk_time_to_origin_stop + passenger_state.walk_time_from_destination_stop


    def get_active_passengers(self) -> List[PassengerState]:
        """Get list of all active passengers"""
        return [
            p for p in self.passengers_container.items.values()
            if p.status not in [
                PassengerStatus.ARRIVED_AT_DESTINATION,
                PassengerStatus.CANCELLED
            ]
        ]

    def get_all_passengers(self) -> List[PassengerState]:
        """Get list of all passengers"""
        return list(self.passengers_container.items.values())

    def get_passengers_by_status(self, status: PassengerStatus) -> List[PassengerState]:
        """Get list of passengers with specific status"""
        return [
            p for p in self.passengers_container.items.values()
            if p.status == status
        ]
    
    def take_snapshot(self, timestamp: datetime) -> None:
        """Take snapshot of passenger states"""
        if not self.initialized:
            raise RuntimeError("Worker not initialized")
        self.passengers_container.take_snapshot(timestamp)

    def get_metrics(self) -> Dict[str, Any]:
        """Get current passenger system metrics"""
        if not self.initialized:
            raise RuntimeError("Worker not initialized")
        
        all_passengers = self.get_all_passengers()

    def get_state(self) -> PassengerSystemState:
        """Get current state of the passenger system"""
        if not self.initialized:
            raise RuntimeError("Worker not initialized")
            
        try:
            # Get passengers by status
            passengers_by_status = defaultdict(list)
            for pid, passenger in self.passengers_container.items.items():
                passengers_by_status[passenger.status].append(pid)
            
            return PassengerSystemState(
                active_passengers={
                    pid: pstate for pid, pstate in self.passengers_container.items.items()
                },
                passengers_by_status=dict(passengers_by_status),
            )
            
        except Exception as e:
            self.logger.error(f"Failed to get passenger system state: {str(e)}")
            raise

    def update_state(self, state: PassengerSystemState) -> None:
        """Update passenger system state"""
        if not self.initialized:
            raise RuntimeError("Worker not initialized")
            
        try:
            # Update passenger states
            self.passengers_container.items = state.active_passengers
            
            self.logger.info("Passenger system state updated successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to update passenger system state: {str(e)}")
            raise

    def restore_state(self, saved_state: Dict[str, Any]) -> None:
        """Restore state from saved PassengerSystemState"""
        if not self.initialized:
            raise RuntimeError("Worker not initialized")
            
        try:
            # Reset current state
            self.passengers_container = StateContainer[PassengerState]()
            # Restore using update_state
            self.update_state(saved_state)
            
            self.logger.debug("Restored passenger system state")
            
        except Exception as e:
            self.logger.error(f"Failed to restore state: {str(e)}")
            raise

    def begin_transaction(self) -> None:
        """Begin state transaction"""
        if not self.initialized:
            raise RuntimeError("Worker not initialized")
        self.passengers_container.begin_transaction()

    def commit_transaction(self) -> None:
        """Commit current transaction"""
        if not self.initialized:
            raise RuntimeError("Worker not initialized")
        self.passengers_container.commit_transaction()

    def rollback_transaction(self) -> None:
        """Rollback current transaction"""
        if not self.initialized:
            raise RuntimeError("Worker not initialized")
        self.passengers_container.rollback_transaction()

    def cleanup(self) -> None:
        """Clean up worker resources"""
        if not self.initialized:
            return
            
        self.passengers_container.clear_history()
        self.initialized = False
        self.logger.info("Cleaned up passenger state worker")