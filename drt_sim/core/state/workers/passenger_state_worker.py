# drt_sim/core/state/workers/passenger_worker.py

from typing import Optional, Dict, Any, List
from datetime import datetime
import logging

from drt_sim.core.state.base import StateContainer, StateWorker
from drt_sim.models.passenger import Passenger, PassengerStatus, PassengerState


class PassengerStateWorker(StateWorker):
    """
    Concrete implementation of StateWorker for managing passengers.
    """
    
    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.passengers_container: StateContainer[PassengerState] = StateContainer()
        self.metrics = {
            'total_passengers': 0,
            'active_passengers': 0,
            'completed_passengers': 0,
            'cancelled_passengers': 0,
            'average_wait_time': 0.0,        # in seconds
            'average_in_vehicle_time': 0.0,  # in seconds
            'average_journey_time': 0.0,     # in seconds
            'service_level_violations': 0
            # Additional metrics can be added here
        }

    def initialize(self, config: Optional[Any] = None) -> None:
        """
        Initialize worker with initial passenger data.
        """
        try:
            self.logger.info("PassengerStateWorker initialized successfully")
        except Exception as e:
            self.logger.error(f"Failed to initialize PassengerStateWorker: {str(e)}")
            raise

    def take_snapshot(self, timestamp: datetime) -> None:
        """
        Take a snapshot of the current passenger states.
        """
        try:
            self.passengers_container.take_snapshot(timestamp)
            self.logger.info(f"Snapshot taken at {timestamp}")
        except Exception as e:
            self.logger.error(f"Failed to take snapshot: {str(e)}")
            raise

    def get_metrics(self) -> Dict[str, Any]:
        """
        Get metrics related to passengers.
        """
        try:
            # Reset metrics before recalculating
            self.metrics = {
                'total_passengers': len(self.passengers_container.items),
                'active_passengers': len([
                    p for p in self.passengers_container.items.values()
                    if p.status not in [PassengerStatus.COMPLETED, PassengerStatus.CANCELLED]
                ]),
                'completed_passengers': len([
                    p for p in self.passengers_container.items.values()
                    if p.status == PassengerStatus.COMPLETED
                ]),
                'cancelled_passengers': len([
                    p for p in self.passengers_container.items.values()
                    if p.status == PassengerStatus.CANCELLED
                ]),
                'average_wait_time': self._calculate_average_wait_time(),
                'average_in_vehicle_time': self._calculate_average_in_vehicle_time(),
                'average_journey_time': self._calculate_average_journey_time(),
                'service_level_violations': self._calculate_service_level_violations()
                # Additional metrics can be calculated here
            }
            return self.metrics
        except Exception as e:
            self.logger.error(f"Failed to get metrics: {str(e)}")
            raise

    def get_state(self) -> Dict[str, Any]:
        """
        Get the current state of all passengers.
        """
        try:
            state = {passenger_id: passenger_state.to_dict()
                     for passenger_id, passenger_state in self.passengers_container.items.items()}
            return state
        except Exception as e:
            self.logger.error(f"Failed to get state: {str(e)}")
            raise

    def update_state(self, state: Dict[str, Any]) -> None:
        """
        Update the state of passengers.
        """
        try:
            for passenger_id, updates in state.items():
                passenger_state: PassengerState = self.passengers_container.get(passenger_id)
                if not passenger_state:
                    self.logger.warning(f"Passenger ID: {passenger_id} does not exist. Skipping update.")
                    continue

                previous_status = passenger_state.status
                passenger_state.status = updates.get('status', passenger_state.status)
                passenger_state.current_location = updates.get('current_location', passenger_state.current_location)
                passenger_state.assigned_vehicle = updates.get('assigned_vehicle', passenger_state.assigned_vehicle)

                # Update timing metrics
                self._update_timing_metrics(passenger_state, updates)

                # Update journey metrics if available
                self._update_journey_metrics(passenger_state, updates)

                # Update service level violations
                self._update_service_level_violations(passenger_state, updates)

                self.passengers_container.update(passenger_id, updates)
                self.logger.info(f"Updated PassengerState for Passenger ID: {passenger_id} with {updates}")

                # Update overall metrics based on status changes
                self._update_overall_metrics(previous_status, passenger_state.status)
        except Exception as e:
            self.logger.error(f"Failed to update state: {str(e)}")
            raise

    def restore_state(self, saved_state: Dict[str, Any]) -> None:
        """
        Restore the state of passengers from a saved snapshot.
        """
        try:
            for passenger_id, passenger_data in saved_state.items():
                if passenger_id in self.passengers_container.items:
                    self.passengers_container.update(passenger_id, passenger_data)
                    self.logger.info(f"Restored PassengerState for Passenger ID: {passenger_id}")
                else:
                    # Create new PassengerState if not exists
                    passenger_state = PassengerState(**passenger_data)
                    self.passengers_container.add(passenger_id, passenger_state)
                    self.logger.info(f"Added new PassengerState for Passenger ID: {passenger_id} during state restoration")
            self.logger.info("State restored successfully.")
        except Exception as e:
            self.logger.error(f"Failed to restore state: {str(e)}")
            raise

    def begin_transaction(self) -> None:
        """
        Begin a state transaction.
        """
        try:
            self.passengers_container.begin_transaction()
            self.logger.info("State transaction started.")
        except Exception as e:
            self.logger.error(f"Failed to begin transaction: {str(e)}")
            raise

    def commit_transaction(self) -> None:
        """
        Commit the current state transaction.
        """
        try:
            self.passengers_container.commit_transaction()
            self.logger.info("State transaction committed.")
        except Exception as e:
            self.logger.error(f"Failed to commit transaction: {str(e)}")
            raise

    def rollback_transaction(self) -> None:
        """
        Rollback the current state transaction.
        """
        try:
            self.passengers_container.rollback_transaction()
            self.logger.info("State transaction rolled back.")
        except Exception as e:
            self.logger.error(f"Failed to rollback transaction: {str(e)}")
            raise

    def cleanup(self) -> None:
        """
        Clean up worker resources.
        """
        try:
            self.passengers_container.clear_history()
            self.logger.info("PassengerWorker cleanup completed.")
        except Exception as e:
            self.logger.error(f"Failed to cleanup PassengerWorker: {str(e)}")
            raise

    def get_passenger_state(self, passenger_id: str) -> Optional[PassengerState]:
        """Retrieve the PassengerState by passenger ID."""
        return self.passengers_container.get(passenger_id)

    def create_passenger_state(self, passenger_state: PassengerState) -> None:
        """Create a new PassengerState."""
        try:
            self.passengers_container.add(passenger_state.id, passenger_state)
            self.logger.info(f"Created PassengerState for Passenger ID: {passenger_state.id}")
            self.metrics['total_passengers'] += 1
            if passenger_state.status not in [PassengerStatus.COMPLETED, PassengerStatus.CANCELLED]:
                self.metrics['active_passengers'] += 1
            elif passenger_state.status == PassengerStatus.COMPLETED:
                self.metrics['completed_passengers'] += 1
            elif passenger_state.status == PassengerStatus.CANCELLED:
                self.metrics['cancelled_passengers'] += 1
        except Exception as e:
            self.logger.error(f"Failed to create PassengerState for Passenger ID: {passenger_state.id}: {str(e)}")
            raise

    def delete_passenger_state(self, passenger_id: str) -> None:
        """Delete a PassengerState."""
        try:
            passenger_state = self.passengers_container.get(passenger_id)
            if not passenger_state:
                self.logger.warning(f"Passenger ID: {passenger_id} does not exist. Cannot delete.")
                return

            self.passengers_container.remove(passenger_id)
            self.logger.info(f"Deleted PassengerState for Passenger ID: {passenger_id}")
            self.metrics['total_passengers'] -= 1
            if passenger_state.status not in [PassengerStatus.COMPLETED, PassengerStatus.CANCELLED]:
                self.metrics['active_passengers'] -= 1
            elif passenger_state.status == PassengerStatus.COMPLETED:
                self.metrics['completed_passengers'] -= 1
            elif passenger_state.status == PassengerStatus.CANCELLED:
                self.metrics['cancelled_passengers'] -= 1
        except Exception as e:
            self.logger.error(f"Failed to delete PassengerState for Passenger ID: {passenger_id}: {str(e)}")
            raise

    def get_passengers_by_status(self, status: PassengerStatus) -> List[PassengerState]:
        """Retrieve all passengers with a specific status."""
        return [p for p in self.passengers_container.items.values() if p.status == status]

    def get_passengers_in_vehicle(self, vehicle_id: str) -> List[PassengerState]:
        """Retrieve all passengers currently in a specific vehicle."""
        return [p for p in self.passengers_container.items.values()
                if p.status == PassengerStatus.IN_VEHICLE and p.assigned_vehicle == vehicle_id]

    # Metrics Calculation Methods

    def _calculate_average_wait_time(self) -> float:
        """
        Calculate the average wait time for passengers.
        Wait time is defined as the duration between ARRIVED_AT_PICKUP and PICKUP_COMPLETED.
        """
        wait_times = []
        for passenger in self.passengers_container.items.values():
            if passenger.walking_to_pickup_end and passenger.waiting_end:
                wait_duration = (passenger.waiting_end - passenger.waiting_start).total_seconds()
                wait_times.append(wait_duration)
        if wait_times:
            average_wait = sum(wait_times) / len(wait_times)
            self.logger.debug(f"Calculated average wait time: {average_wait} seconds")
            return average_wait
        return 0.0

    def _calculate_average_in_vehicle_time(self) -> float:
        """
        Calculate the average in-vehicle time for passengers.
        """
        in_vehicle_times = []
        for passenger in self.passengers_container.items.values():
            if passenger.in_vehicle_end and passenger.in_vehicle_start:
                in_vehicle_duration = (passenger.in_vehicle_end - passenger.in_vehicle_start).total_seconds()
                in_vehicle_times.append(in_vehicle_duration)
        if in_vehicle_times:
            average_in_vehicle = sum(in_vehicle_times) / len(in_vehicle_times)
            self.logger.debug(f"Calculated average in-vehicle time: {average_in_vehicle} seconds")
            return average_in_vehicle
        return 0.0

    def _calculate_average_journey_time(self) -> float:
        """
        Calculate the average journey time for passengers.
        Journey time is defined as the duration between WALKING_TO_PICKUP and ARRIVED_AT_DESTINATION.
        """
        journey_times = []
        for passenger in self.passengers_container.items.values():
            if passenger.walking_to_pickup_start and passenger.walking_to_destination_end:
                journey_duration = (passenger.walking_to_destination_end - passenger.walking_to_pickup_start).total_seconds()
                journey_times.append(journey_duration)
        if journey_times:
            average_journey = sum(journey_times) / len(journey_times)
            self.logger.debug(f"Calculated average journey time: {average_journey} seconds")
            return average_journey
        return 0.0

    def _calculate_service_level_violations(self) -> int:
        """
        Calculate the total number of service level violations.
        """
        violations = 0
        for passenger in self.passengers_container.items.values():
            if passenger.service_level_violations:
                violations += len(passenger.service_level_violations)
        self.logger.debug(f"Total service level violations: {violations}")
        return violations

    # Internal Methods for State Updates

    def _update_timing_metrics(self, passenger_state: PassengerState, updates: Dict[str, Any]) -> None:
        """
        Update timing metrics based on the provided updates.
        """
        if 'walking_to_pickup_start' in updates:
            passenger_state.walking_to_pickup_start = updates['walking_to_pickup_start']
        if 'walking_to_pickup_end' in updates:
            passenger_state.walking_to_pickup_end = updates['walking_to_pickup_end']
        if 'waiting_start' in updates:
            passenger_state.waiting_start = updates['waiting_start']
        if 'waiting_end' in updates:
            passenger_state.waiting_end = updates['waiting_end']
        if 'boarding_start' in updates:
            passenger_state.boarding_start = updates['boarding_start']
        if 'boarding_end' in updates:
            passenger_state.boarding_end = updates['boarding_end']
        if 'in_vehicle_start' in updates:
            passenger_state.in_vehicle_start = updates['in_vehicle_start']
        if 'in_vehicle_end' in updates:
            passenger_state.in_vehicle_end = updates['in_vehicle_end']
        if 'walking_to_destination_start' in updates:
            passenger_state.walking_to_destination_start = updates['walking_to_destination_start']
        if 'walking_to_destination_end' in updates:
            passenger_state.walking_to_destination_end = updates['walking_to_destination_end']
        if 'completion_time' in updates:
            passenger_state.completion_time = updates['completion_time']

    def _update_journey_metrics(self, passenger_state: PassengerState, updates: Dict[str, Any]) -> None:
        """
        Update journey metrics based on the provided updates.
        """
        if 'access_walking_distance' in updates:
            passenger_state.access_walking_distance = updates['access_walking_distance']
        if 'egress_walking_distance' in updates:
            passenger_state.egress_walking_distance = updates['egress_walking_distance']
        if 'total_wait_time' in updates:
            passenger_state.total_wait_time = updates['total_wait_time']
        if 'total_in_vehicle_time' in updates:
            passenger_state.total_in_vehicle_time = updates['total_in_vehicle_time']
        if 'total_journey_time' in updates:
            passenger_state.total_journey_time = updates['total_journey_time']
        if 'route_deviation_ratio' in updates:
            passenger_state.route_deviation_ratio = updates['route_deviation_ratio']

    def _update_service_level_violations(self, passenger_state: PassengerState, updates: Dict[str, Any]) -> None:
        """
        Update service level violations based on the provided updates.
        """
        if 'service_level_violation' in updates:
            violation = updates['service_level_violation']
            passenger_state.service_level_violations.append(violation)
            self.logger.debug(f"Passenger ID: {passenger_state.id} has a new service level violation: {violation}")

    def _update_overall_metrics(self, previous_status: PassengerStatus, new_status: PassengerStatus) -> None:
        """
        Update overall metrics based on status transitions.
        """
        if previous_status not in [PassengerStatus.COMPLETED, PassengerStatus.CANCELLED] and \
           new_status in [PassengerStatus.COMPLETED, PassengerStatus.CANCELLED]:
            self.metrics['active_passengers'] -= 1
            if new_status == PassengerStatus.COMPLETED:
                self.metrics['completed_passengers'] += 1
            elif new_status == PassengerStatus.CANCELLED:
                self.metrics['cancelled_passengers'] += 1
