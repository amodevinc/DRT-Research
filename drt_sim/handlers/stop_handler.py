# drt_sim/handlers/stop_handler.py

from typing import Optional, List
from datetime import datetime, timedelta
import logging

from .base import BaseHandler
from drt_sim.models.event import (
    Event, EventType, EventPriority
)
from drt_sim.models.stop import StopStatus
from drt_sim.config.config import ScenarioConfig
from drt_sim.core.simulation.context import SimulationContext
from drt_sim.core.state.manager import StateManager


class StopHandler(BaseHandler):
    """
    Handles all stop-related events in the DRT simulation, managing the
    activation, deactivation, congestion, and capacity of stops.
    """

    def handle_stop_activated(self, event: Event) -> Optional[List[Event]]:
        """
        Handle the activation of a stop.
        """
        stop_id = event.data.get('stop_id')
        try:
            self.logger.info(f"Activating Stop ID: {stop_id}")

            # Retrieve the stop from the state manager
            stop = self.state_manager.stop_worker.get_stop(stop_id)
            if not stop:
                raise ValueError(f"Stop ID: {stop_id} does not exist.")

            # Update stop status to ACTIVE
            self.state_manager.stop_worker.update_stop_status(
                stop_id,
                StopStatus.ACTIVE,
                {
                    'activated_time': self.context.current_time
                }
            )

            self.logger.info(f"Stop {stop_id} activated successfully.")

            # Optionally, notify vehicles or other components about the activation
            # For example, re-evaluate vehicle assignments to this stop

            return None

        except Exception as e:
            self.logger.error(f"Failed to activate Stop ID: {stop_id}: {str(e)}")
            # Create SIMULATION_ERROR event
            error_event = Event(
                event_type=EventType.SIMULATION_ERROR,
                timestamp=self.context.current_time,
                priority=EventPriority.CRITICAL,
                data={
                    'original_event_id': event.id,
                    'original_event_type': event.event_type.value,
                    'error_message': str(e),
                    'error_type': e.__class__.__name__
                }
            )
            return [error_event]

    def handle_stop_deactivated(self, event: Event) -> Optional[List[Event]]:
        """
        Handle the deactivation of a stop.
        """
        stop_id = event.data.get('stop_id')
        try:
            self.logger.info(f"Deactivating Stop ID: {stop_id}")

            # Retrieve the stop from the state manager
            stop = self.state_manager.stop_worker.get_stop(stop_id)
            if not stop:
                raise ValueError(f"Stop ID: {stop_id} does not exist.")

            # Update stop status to INACTIVE
            self.state_manager.stop_worker.update_stop_status(
                stop_id,
                StopStatus.INACTIVE,
                {
                    'deactivated_time': self.context.current_time
                }
            )

            self.logger.info(f"Stop {stop_id} deactivated successfully.")

            # Optionally, handle vehicles currently assigned to this stop
            # For example, reroute vehicles or notify passengers

            return None

        except Exception as e:
            self.logger.error(f"Failed to deactivate Stop ID: {stop_id}: {str(e)}")
            # Create SIMULATION_ERROR event
            error_event = Event(
                event_type=EventType.SIMULATION_ERROR,
                timestamp=self.context.current_time,
                priority=EventPriority.CRITICAL,
                data={
                    'original_event_id': event.id,
                    'original_event_type': event.event_type.value,
                    'error_message': str(e),
                    'error_type': e.__class__.__name__
                }
            )
            return [error_event]

    def handle_stop_congested(self, event: Event) -> Optional[List[Event]]:
        """
        Handle the congestion of a stop.
        """
        stop_id = event.data.get('stop_id')
        congestion_level = event.data.get('congestion_level', 'moderate')  # e.g., low, moderate, high
        try:
            self.logger.info(f"Handling congestion for Stop ID: {stop_id} with level: {congestion_level}")

            # Retrieve the stop from the state manager
            stop = self.state_manager.stop_worker.get_stop(stop_id)
            if not stop:
                raise ValueError(f"Stop ID: {stop_id} does not exist.")

            # Update stop's congestion status
            self.state_manager.stop_worker.update_stop_congestion(
                stop_id,
                congestion_level,
                {
                    'congested_time': self.context.current_time
                }
            )

            self.logger.info(f"Stop {stop_id} marked as {congestion_level}ly congested.")

            # Optionally, create events to handle the impact of congestion
            # For example, delay vehicle arrivals or reroute vehicles
            follow_up_events = []

            if congestion_level == 'high':
                # Example: Notify vehicles to wait or reroute
                affected_vehicles = self.state_manager.vehicle_worker.get_vehicles_at_stop(stop_id)
                for vehicle in affected_vehicles:
                    # Create an event to delay departure
                    delay_event = Event(
                        event_type=EventType.VEHICLE_DELAYED,
                        timestamp=self.context.current_time + timedelta(seconds=self.config.stop.delay_duration),
                        priority=EventPriority.NORMAL,
                        vehicle_id=vehicle.id,
                        data={'reason': 'Stop congestion'}
                    )
                    follow_up_events.append(delay_event)

            return follow_up_events if follow_up_events else None

        except Exception as e:
            self.logger.error(f"Failed to handle congestion for Stop ID: {stop_id}: {str(e)}")
            # Create SIMULATION_ERROR event
            error_event = Event(
                event_type=EventType.SIMULATION_ERROR,
                timestamp=self.context.current_time,
                priority=EventPriority.CRITICAL,
                data={
                    'original_event_id': event.id,
                    'original_event_type': event.event_type.value,
                    'error_message': str(e),
                    'error_type': e.__class__.__name__
                }
            )
            return [error_event]

    def handle_stop_capacity_exceeded(self, event: Event) -> Optional[List[Event]]:
        """
        Handle scenarios where a stop's capacity is exceeded.
        """
        stop_id = event.data.get('stop_id')
        current_load = event.data.get('current_load', 0)
        capacity = event.data.get('capacity', 0)
        try:
            self.logger.info(f"Handling capacity exceeded for Stop ID: {stop_id}. Current Load: {current_load}, Capacity: {capacity}")

            # Retrieve the stop from the state manager
            stop = self.state_manager.stop_worker.get_stop(stop_id)
            if not stop:
                raise ValueError(f"Stop ID: {stop_id} does not exist.")

            # Update stop's capacity status
            self.state_manager.stop_worker.update_stop_capacity(
                stop_id,
                current_load,
                capacity,
                {
                    'capacity_exceeded_time': self.context.current_time
                }
            )

            self.logger.info(f"Stop {stop_id} capacity exceeded: {current_load} > {capacity}.")

            # Optionally, create events to handle the impact of capacity exceeding
            # For example, prevent further assignments or reroute vehicles
            follow_up_events = []

            # Example: Reject new requests at this stop
            reject_event = Event(
                event_type=EventType.REQUEST_REJECTED,
                timestamp=self.context.current_time,
                priority=EventPriority.HIGH,
                data={'reason': 'Stop capacity exceeded'}
            )
            follow_up_events.append(reject_event)

            # Example: Notify fleet manager to rebalance
            rebalance_event = Event(
                event_type=EventType.FLEET_REBALANCING_NEEDED,
                timestamp=self.context.current_time,
                priority=EventPriority.LOW,
                data={'stop_id': stop_id}
            )
            follow_up_events.append(rebalance_event)

            return follow_up_events

        except Exception as e:
            self.logger.error(f"Failed to handle capacity exceeded for Stop ID: {stop_id}: {str(e)}")
            # Create SIMULATION_ERROR event
            error_event = Event(
                event_type=EventType.SIMULATION_ERROR,
                timestamp=self.context.current_time,
                priority=EventPriority.CRITICAL,
                data={
                    'original_event_id': event.id,
                    'original_event_type': event.event_type.value,
                    'error_message': str(e),
                    'error_type': e.__class__.__name__
                }
            )
            return [error_event]

    # Optionally, implement any additional helper methods as needed
