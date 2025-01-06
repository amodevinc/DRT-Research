# drt_sim/handlers/dispatch_handler.py

from typing import Optional, List
from datetime import datetime, timedelta
import logging

from .base import BaseHandler
from drt_sim.models.event import (
    Event, EventType, EventPriority
)
from drt_sim.models.vehicle import VehicleStatus
from drt_sim.models.request import RequestStatus
from drt_sim.algorithms.dispatch.fcfs_dispatch import FCFSDispatch
from drt_sim.algorithms.base_interfaces.dispatch_base import DispatchStrategy
from drt_sim.config.config import ScenarioConfig
from drt_sim.core.simulation.context import SimulationContext
from drt_sim.core.state.manager import StateManager


class DispatchHandler(BaseHandler):
    """
    Handles all dispatch-related events in the DRT simulation, managing the
    assignment of vehicles to passenger requests and optimizing dispatch operations.
    """

    def __init__(
        self,
        config: ScenarioConfig,
        context: SimulationContext,
        state_manager: StateManager
    ):
        super().__init__(config, context, state_manager)
        self.dispatch_algorithm: Optional[DispatchStrategy] = self._initialize_dispatch_algorithm()

    def _initialize_dispatch_algorithm(self) -> Optional[DispatchStrategy]:
        """
        Initialize the dispatch algorithm based on configuration.
        """
        algo_name = self.config.algorithm.dispatch_strategy.lower()
        self.logger.info(f"Initializing dispatch algorithm: {algo_name}")

        if algo_name == 'fcfs':
            return FCFSDispatch(vehicle_speed_kmh=self.config.vehicle.speed)
        # elif algo_name == 'ga':
        #     return GADispatchAlgorithm(self.config, self.context, self.state_manager)
        # elif algo_name == 'rl':
        #     return RLDispatchAlgorithm(self.config, self.context, self.state_manager)
        else:
            self.logger.error(f"Unsupported dispatch algorithm: {algo_name}")
            return None

    def handle_dispatch_requested(self, event: Event) -> Optional[List[Event]]:
        """
        Handle dispatch request event by assigning a vehicle to the request.
        """
        request_id = event.request_id
        try:
            request = self.state_manager.request_worker.get_request(request_id)
            self.logger.info(f"Handling dispatch request for Request ID: {request_id}")

            if not self.dispatch_algorithm:
                raise ValueError("Dispatch algorithm not initialized.")

            # Select vehicle using the dispatch algorithm
            vehicle = self.dispatch_algorithm.assign_vehicle(request)

            if vehicle:
                self.logger.info(f"Assigned Vehicle ID: {vehicle.id} to Request ID: {request_id}")

                # Update vehicle and request statuses
                self.state_manager.vehicle_worker.update_vehicle_state(
                    vehicle.id,
                    VehicleStatus.ASSIGNED,
                    {
                        'assigned_request': request_id,
                        'assignment_time': self.context.current_time
                    }
                )

                self.state_manager.request_worker.update_request_status(
                    request_id,
                    RequestStatus.ASSIGNED,
                    {
                        'assigned_vehicle': vehicle.id,
                        'assignment_time': self.context.current_time
                    }
                )

                # Create DISPATCH_SUCCEEDED event
                dispatch_succeeded_event = Event(
                    event_type=EventType.DISPATCH_SUCCEEDED,
                    timestamp=self.context.current_time,
                    priority=EventPriority.HIGH,
                    request_id=request_id,
                    vehicle_id=vehicle.id,
                    data={'request': request.to_dict(), 'vehicle': vehicle.to_dict()}
                )

                return [dispatch_succeeded_event]
            else:
                self.logger.warning(f"No available vehicles to assign for Request ID: {request_id}")

                # Create DISPATCH_FAILED event
                dispatch_failed_event = Event(
                    event_type=EventType.DISPATCH_FAILED,
                    timestamp=self.context.current_time,
                    priority=EventPriority.HIGH,
                    request_id=request_id,
                    data={'reason': 'No available vehicles'}
                )

                return [dispatch_failed_event]

        except Exception as e:
            self.logger.error(f"Failed to handle dispatch request for Request ID: {request_id}: {str(e)}")
            # Create DISPATCH_FAILED event with error details
            dispatch_failed_event = Event(
                event_type=EventType.DISPATCH_FAILED,
                timestamp=self.context.current_time,
                priority=EventPriority.HIGH,
                request_id=request_id,
                data={'reason': str(e)}
            )
            return [dispatch_failed_event]

    def handle_dispatch_succeeded(self, event: Event) -> Optional[List[Event]]:
        """
        Handle successful dispatch by scheduling vehicle departure.
        """
        request_id = event.request_id
        vehicle_id = event.vehicle_id

        try:
            self.logger.info(f"Dispatch succeeded for Request ID: {request_id} with Vehicle ID: {vehicle_id}")

            # Retrieve request and vehicle
            request = self.state_manager.request_worker.get_request(request_id)
            vehicle = self.state_manager.vehicle_worker.get_vehicle(vehicle_id)

            if not request or not vehicle:
                raise ValueError("Invalid request or vehicle ID.")

            # Schedule vehicle departure event
            departure_event = Event(
                event_type=EventType.VEHICLE_DEPARTED,
                timestamp=self.context.current_time + timedelta(seconds=self.config.dispatch.departure_delay),
                priority=EventPriority.HIGH,
                vehicle_id=vehicle_id,
                data={
                    'destination': request.pickup_location,
                    'purpose': 'pickup'
                }
            )

            self.logger.info(f"Scheduled VEHICLE_DEPARTED event for Vehicle ID: {vehicle_id} at {departure_event.timestamp}")

            return [departure_event]

        except Exception as e:
            self.logger.error(f"Failed to handle dispatch succeeded for Request ID: {request_id}: {str(e)}")
            # Create DISPATCH_FAILED event with error details
            dispatch_failed_event = Event(
                event_type=EventType.DISPATCH_FAILED,
                timestamp=self.context.current_time,
                priority=EventPriority.HIGH,
                request_id=request_id,
                data={'reason': f"Dispatch succeeded handler error: {str(e)}"}
            )
            return [dispatch_failed_event]

    def handle_dispatch_failed(self, event: Event) -> Optional[List[Event]]:
        """
        Handle failed dispatch attempts by retrying or rejecting the request.
        """
        request_id = event.request_id
        reason = event.data.get('reason', 'Unspecified')

        try:
            self.logger.warning(f"Dispatch failed for Request ID: {request_id}. Reason: {reason}")

            request = self.state_manager.request_worker.get_request(request_id)
            if not request:
                raise ValueError(f"Request ID: {request_id} does not exist.")

            # Check if retry attempts are available
            if request.dispatch_attempts < self.config.dispatch.max_retries:
                self.logger.info(f"Retrying dispatch for Request ID: {request_id}. Attempt {request.dispatch_attempts + 1}")

                # Increment dispatch attempts
                self.state_manager.request_worker.increment_dispatch_attempts(request_id)

                # Create another DISPATCH_REQUESTED event
                retry_event = Event(
                    event_type=EventType.DISPATCH_REQUESTED,
                    timestamp=self.context.current_time + timedelta(seconds=self.config.dispatch.retry_delay),
                    priority=EventPriority.HIGH,
                    request_id=request_id,
                    data={'request': request.to_dict()}
                )

                return [retry_event]
            else:
                self.logger.info(f"Max dispatch attempts reached for Request ID: {request_id}. Marking as unassignable.")

                # Update request status to unassignable or rejected
                self.state_manager.request_worker.update_request_status(
                    request_id,
                    RequestStatus.REJECTED,
                    {
                        'rejection_time': self.context.current_time,
                        'rejection_reason': 'Dispatch failed after maximum retries'
                    }
                )

                # Optionally, notify system or take alternative actions
                notify_event = Event(
                    event_type=EventType.REQUEST_REJECTED,
                    timestamp=self.context.current_time,
                    priority=EventPriority.HIGH,
                    request_id=request_id,
                    data={'reason': 'Dispatch failed after maximum retries'}
                )

                return [notify_event]

        except Exception as e:
            self.logger.error(f"Failed to handle dispatch failure for Request ID: {request_id}: {str(e)}")
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

    def handle_dispatch_batch_started(self, event: Event) -> Optional[List[Event]]:
        """
        Handle the start of a dispatch batch optimization process.
        """
        try:
            self.logger.info("Starting dispatch batch optimization.")

            # Invoke the dispatch algorithm's batch optimization
            batch_results = self.dispatch_algorithm.optimize_dispatch_batch()

            # Process batch results and create appropriate events
            follow_up_events = []
            for result in batch_results:
                if result['status'] == 'success':
                    follow_up_event = Event(
                        event_type=EventType.DISPATCH_SUCCEEDED,
                        timestamp=self.context.current_time,
                        priority=EventPriority.HIGH,
                        request_id=result['request_id'],
                        vehicle_id=result['vehicle_id'],
                        data={'request': result['request'].to_dict(), 'vehicle': result['vehicle'].to_dict()}
                    )
                else:
                    follow_up_event = Event(
                        event_type=EventType.DISPATCH_FAILED,
                        timestamp=self.context.current_time,
                        priority=EventPriority.HIGH,
                        request_id=result['request_id'],
                        data={'reason': result.get('reason', 'Batch dispatch failed')}
                    )
                follow_up_events.append(follow_up_event)

            # Schedule DISPATCH_BATCH_COMPLETED event
            batch_completed_event = Event(
                event_type=EventType.DISPATCH_BATCH_COMPLETED,
                timestamp=self.context.current_time + timedelta(seconds=self.config.dispatch.batch_processing_time),
                priority=EventPriority.LOW,
                data={'batch_size': len(batch_results)}
            )

            follow_up_events.append(batch_completed_event)

            self.logger.info("Completed dispatch batch optimization.")

            return follow_up_events

        except Exception as e:
            self.logger.error(f"Failed to handle dispatch batch started: {str(e)}")
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

    def handle_dispatch_batch_completed(self, event: Event) -> Optional[List[Event]]:
        """
        Handle the completion of a dispatch batch optimization process.
        """
        try:
            batch_size = event.data.get('batch_size', 0)
            self.logger.info(f"Dispatch batch completed with {batch_size} dispatch operations.")

            # Optionally, trigger further actions such as logging or monitoring
            # For example, log batch completion metrics
            # This can be expanded based on specific requirements

            return None

        except Exception as e:
            self.logger.error(f"Failed to handle dispatch batch completed: {str(e)}")
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

    def handle_dispatch_optimization_started(self, event: Event) -> Optional[List[Event]]:
        """
        Handle the start of dispatch optimization.
        """
        try:
            self.logger.info("Dispatch optimization started.")

            # Trigger optimization process (implementation depends on algorithm)
            optimization_results = self.dispatch_algorithm.optimize()

            # Schedule DISPATCH_OPTIMIZATION_COMPLETED event
            optimization_completed_event = Event(
                event_type=EventType.DISPATCH_OPTIMIZATION_COMPLETED,
                timestamp=self.context.current_time + timedelta(seconds=self.config.dispatch.optimization_duration),
                priority=EventPriority.LOW,
                data={'results': optimization_results}
            )

            self.logger.info("Dispatch optimization completed.")

            return [optimization_completed_event]

        except Exception as e:
            self.logger.error(f"Failed to handle dispatch optimization started: {str(e)}")
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

    def handle_dispatch_optimization_completed(self, event: Event) -> Optional[List[Event]]:
        """
        Handle the completion of dispatch optimization.
        """
        try:
            optimization_results = event.data.get('results', {})
            self.logger.info("Dispatch optimization process completed.")
            self.logger.debug(f"Optimization results: {optimization_results}")

            # Optionally, handle optimization results, such as adjusting vehicle assignments
            # This can be expanded based on specific requirements

            return None

        except Exception as e:
            self.logger.error(f"Failed to handle dispatch optimization completed: {str(e)}")
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
