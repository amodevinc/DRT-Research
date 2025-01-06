# drt_sim/handlers/passenger_handler.py

from typing import Optional, List
from datetime import datetime, timedelta
import logging

from .base import BaseHandler
from drt_sim.models.event import (
    Event, EventType, EventPriority
)
from drt_sim.models.passenger import PassengerStatus
from drt_sim.models.vehicle import VehicleStatus
from drt_sim.models.passenger import PassengerState
from drt_sim.models.location import Location


class PassengerHandler(BaseHandler):
    """
    Handles all passenger-related events in the DRT simulation, managing the
    lifecycle of passengers from arrival to departure.
    """

    def handle_passenger_walking_to_pickup(self, event: Event) -> Optional[List[Event]]:
        """
        Handle a passenger walking towards the pickup location.
        """
        passenger_id = event.passenger_id
        request_id = event.request_id
        try:
            self.logger.info(f"Passenger {passenger_id} is walking to pickup for Request ID: {request_id}")

            # Retrieve PassengerState
            passenger_state = self.state_manager.passenger_worker.get_passenger_state(passenger_id)
            if not passenger_state:
                raise ValueError(f"PassengerState for Passenger ID: {passenger_id} does not exist.")

            # Update status to WALKING_TO_PICKUP
            updates = {
                'status': PassengerStatus.WALKING_TO_PICKUP.value,
                'walking_to_pickup_start': self.context.current_time,
                'current_location': passenger_state.origin.to_dict()
            }
            self.state_manager.passenger_worker.update_state({passenger_id: updates})

            # Calculate walking duration
            walking_duration = self._calculate_walking_duration(passenger_state.origin, passenger_state.pickup_point)

            # Schedule arrival at pickup
            arrival_event = Event(
                event_type=EventType.PASSENGER_ARRIVED_AT_PICKUP,
                timestamp=self.context.current_time + timedelta(seconds=walking_duration),
                priority=EventPriority.NORMAL,
                passenger_id=passenger_id,
                request_id=request_id,
                data={'pickup_location': passenger_state.pickup_point.to_dict()}
            )

            self.logger.info(f"Scheduled PASSENGER_ARRIVED_AT_PICKUP for Passenger {passenger_id} at {arrival_event.timestamp}")

            return [arrival_event]

        except Exception as e:
            self.logger.error(f"Failed to handle PASSENGER_WALKING_TO_PICKUP for Passenger {passenger_id}: {str(e)}")
            return self._create_error_event(event, e)

    def handle_passenger_arrived_at_pickup_stop(self, event: Event) -> Optional[List[Event]]:
        """
        Handle a passenger arriving at the pickup location.
        """
        passenger_id = event.passenger_id
        request_id = event.request_id
        try:
            self.logger.info(f"Passenger {passenger_id} arrived at pickup for Request ID: {request_id}")

            # Retrieve PassengerState
            passenger_state = self.state_manager.passenger_worker.get_passenger_state(passenger_id)
            if not passenger_state:
                raise ValueError(f"PassengerState for Passenger ID: {passenger_id} does not exist.")

            # Update status to WAITING_FOR_VEHICLE
            updates = {
                'status': PassengerStatus.WAITING_FOR_VEHICLE.value,
                'walking_to_pickup_end': self.context.current_time,
                'current_location': passenger_state.pickup_point.to_dict(),
                'waiting_start': self.context.current_time
            }
            self.state_manager.passenger_worker.update_state({passenger_id: updates})

            # Optionally, trigger vehicle dispatch if not already handled elsewhere

            return None

        except Exception as e:
            self.logger.error(f"Failed to handle PASSENGER_ARRIVED_AT_PICKUP for Passenger {passenger_id}: {str(e)}")
            return self._create_error_event(event, e)

    def handle_request_pickup_started(self, event: Event) -> Optional[List[Event]]:
        """
        Handle the start of the pickup process for a request.
        """
        passenger_id = event.passenger_id
        request_id = event.request_id
        vehicle_id = event.vehicle_id
        try:
            self.logger.info(f"Starting pickup for Passenger {passenger_id} with Vehicle {vehicle_id} for Request ID: {request_id}")

            # Retrieve PassengerState
            passenger_state = self.state_manager.passenger_worker.get_passenger_state(passenger_id)
            if not passenger_state:
                raise ValueError(f"PassengerState for Passenger ID: {passenger_id} does not exist.")

            # Update passenger status to PICKUP_STARTED
            updates = {
                'status': PassengerStatus.PICKUP_STARTED.value,
                'boarding_start': self.context.current_time,
                'assigned_vehicle': vehicle_id
            }
            self.state_manager.passenger_worker.update_state({passenger_id: updates})

            # Update vehicle status to LOADING
            self.state_manager.vehicle_worker.update_vehicle_state(
                vehicle_id,
                VehicleStatus.LOADING,
                {'current_passengers': [passenger_id]}
            )

            # Schedule pickup completion based on boarding time
            pickup_duration = self.config.passenger.boarding_time
            pickup_completed_event = Event(
                event_type=EventType.PICKUP_COMPLETED,
                timestamp=self.context.current_time + timedelta(seconds=pickup_duration),
                priority=EventPriority.HIGH,
                passenger_id=passenger_id,
                request_id=request_id,
                vehicle_id=vehicle_id,
                data={'pickup_location': passenger_state.pickup_point.to_dict()}
            )

            self.logger.info(f"Scheduled PICKUP_COMPLETED for Passenger {passenger_id} at {pickup_completed_event.timestamp}")

            return [pickup_completed_event]

        except Exception as e:
            self.logger.error(f"Failed to handle REQUEST_PICKUP_STARTED for Passenger {passenger_id}: {str(e)}")
            return self._create_error_event(event, e)

    def handle_pickup_completed(self, event: Event) -> Optional[List[Event]]:
        """
        Handle the completion of the pickup process for a request.
        """
        passenger_id = event.passenger_id
        request_id = event.request_id
        vehicle_id = event.vehicle_id
        try:
            self.logger.info(f"Completed pickup for Passenger {passenger_id} with Vehicle {vehicle_id} for Request ID: {request_id}")

            # Retrieve PassengerState
            passenger_state = self.state_manager.passenger_worker.get_passenger_state(passenger_id)
            if not passenger_state:
                raise ValueError(f"PassengerState for Passenger ID: {passenger_id} does not exist.")

            # Update passenger status to IN_VEHICLE
            updates = {
                'status': PassengerStatus.IN_VEHICLE.value,
                'boarding_end': self.context.current_time,
                'in_vehicle_start': self.context.current_time,
                'current_location': passenger_state.origin.to_dict(),
                'total_wait_time': (passenger_state.boarding_start - passenger_state.waiting_start).total_seconds()
            }
            self.state_manager.passenger_worker.update_state({passenger_id: updates})

            # Update request status to IN_VEHICLE
            self.state_manager.request_worker.update_request_status(
                request_id,
                status='in_vehicle',
                metadata={'vehicle_id': vehicle_id, 'pickup_time': self.context.current_time}
            )

            # Update vehicle status to OCCUPIED
            self.state_manager.vehicle_worker.update_vehicle_state(
                vehicle_id,
                VehicleStatus.OCCUPIED,
                {'current_passengers': [passenger_id], 'next_destination': passenger_state.destination.to_dict()}
            )

            # Schedule vehicle departure towards dropoff
            departure_delay = self.config.vehicle.departure_delay  # seconds
            departure_event = Event(
                event_type=EventType.VEHICLE_DEPARTED,
                timestamp=self.context.current_time + timedelta(seconds=departure_delay),
                priority=EventPriority.HIGH,
                vehicle_id=vehicle_id,
                data={
                    'destination': passenger_state.destination.to_dict(),
                    'purpose': 'dropoff'
                }
            )

            self.logger.info(f"Scheduled VEHICLE_DEPARTED for Vehicle {vehicle_id} at {departure_event.timestamp}")

            return [departure_event]

        except Exception as e:
            self.logger.error(f"Failed to handle PICKUP_COMPLETED for Passenger {passenger_id}: {str(e)}")
            return self._create_error_event(event, e)

    def handle_passenger_in_vehicle(self, event: Event) -> Optional[List[Event]]:
        """
        Handle passenger being in the vehicle.
        """
        passenger_id = event.passenger_id
        request_id = event.request_id
        vehicle_id = event.vehicle_id
        try:
            self.logger.info(f"Passenger {passenger_id} is now in Vehicle {vehicle_id} for Request ID: {request_id}")

            # Retrieve PassengerState
            passenger_state = self.state_manager.passenger_worker.get_passenger_state(passenger_id)
            if not passenger_state:
                raise ValueError(f"PassengerState for Passenger ID: {passenger_id} does not exist.")

            # Update passenger status to IN_VEHICLE
            updates = {
                'status': PassengerStatus.IN_VEHICLE.value,
                'in_vehicle_start': self.context.current_time,
                'assigned_vehicle': vehicle_id
            }
            self.state_manager.passenger_worker.update_state({passenger_id: updates})

            # Update request status if necessary
            self.state_manager.request_worker.update_request_status(
                request_id,
                status='in_vehicle',
                metadata={'in_vehicle_time': self.context.current_time, 'vehicle_id': vehicle_id}
            )

            return None

        except Exception as e:
            self.logger.error(f"Failed to handle PASSENGER_IN_VEHICLE for Passenger {passenger_id}: {str(e)}")
            return self._create_error_event(event, e)

    def handle_passenger_detour_started(self, event: Event) -> Optional[List[Event]]:
        """
        Handle the start of a detour for a passenger.
        """
        passenger_id = event.passenger_id
        request_id = event.request_id
        vehicle_id = event.vehicle_id
        detour_location = event.data.get('detour_location')
        try:
            self.logger.info(f"Passenger {passenger_id} started detour at Vehicle {vehicle_id} for Request ID: {request_id} towards {detour_location}")

            # Retrieve PassengerState
            passenger_state = self.state_manager.passenger_worker.get_passenger_state(passenger_id)
            if not passenger_state:
                raise ValueError(f"PassengerState for Passenger ID: {passenger_id} does not exist.")

            # Update passenger status to DETOUR_STARTED
            updates = {
                'status': PassengerStatus.DETOUR_STARTED.value,
                'detour_start_time': self.context.current_time,
                'current_location': detour_location.to_dict()
            }
            self.state_manager.passenger_worker.update_state({passenger_id: updates})

            # Schedule detour completion based on detour duration
            detour_duration = self.config.passenger.detour_duration  # seconds
            detour_completed_event = Event(
                event_type=EventType.PASSENGER_DETOUR_ENDED,
                timestamp=self.context.current_time + timedelta(seconds=detour_duration),
                priority=EventPriority.NORMAL,
                passenger_id=passenger_id,
                request_id=request_id,
                vehicle_id=vehicle_id,
                data={'detour_location': detour_location.to_dict()}
            )

            self.logger.info(f"Scheduled PASSENGER_DETOUR_ENDED for Passenger {passenger_id} at {detour_completed_event.timestamp}")

            return [detour_completed_event]

        except Exception as e:
            self.logger.error(f"Failed to handle PASSENGER_DETOUR_STARTED for Passenger {passenger_id}: {str(e)}")
            return self._create_error_event(event, e)

    def handle_passenger_detour_ended(self, event: Event) -> Optional[List[Event]]:
        """
        Handle the completion of a detour for a passenger.
        """
        passenger_id = event.passenger_id
        request_id = event.request_id
        vehicle_id = event.vehicle_id
        try:
            self.logger.info(f"Passenger {passenger_id} completed detour at Vehicle {vehicle_id} for Request ID: {request_id}")

            # Retrieve PassengerState
            passenger_state = self.state_manager.passenger_worker.get_passenger_state(passenger_id)
            if not passenger_state:
                raise ValueError(f"PassengerState for Passenger ID: {passenger_id} does not exist.")

            # Update passenger status to RESUMED_VEHICLE_TRIP
            updates = {
                'status': PassengerStatus.RESUMED_VEHICLE_TRIP.value,
                'detour_end_time': self.context.current_time,
                'current_location': passenger_state.destination.to_dict()
            }
            self.state_manager.passenger_worker.update_state({passenger_id: updates})

            # Update vehicle's route or status if necessary

            return None

        except Exception as e:
            self.logger.error(f"Failed to handle PASSENGER_DETOUR_ENDED for Passenger {passenger_id}: {str(e)}")
            return self._create_error_event(event, e)

    def handle_request_dropoff_started(self, event: Event) -> Optional[List[Event]]:
        """
        Handle the start of the dropoff process for a request.
        """
        passenger_id = event.passenger_id
        request_id = event.request_id
        vehicle_id = event.vehicle_id
        try:
            self.logger.info(f"Starting dropoff for Passenger {passenger_id} with Vehicle {vehicle_id} for Request ID: {request_id}")

            # Retrieve PassengerState
            passenger_state = self.state_manager.passenger_worker.get_passenger_state(passenger_id)
            if not passenger_state:
                raise ValueError(f"PassengerState for Passenger ID: {passenger_id} does not exist.")

            # Update passenger status to DROPOFF_STARTED
            updates = {
                'status': PassengerStatus.DROPOFF_STARTED.value,
                'dropoff_start_time': self.context.current_time,
                'current_location': passenger_state.dropoff_point.to_dict()
            }
            self.state_manager.passenger_worker.update_state({passenger_id: updates})

            # Update request status to DROPOFF_STARTED
            self.state_manager.request_worker.update_request_status(
                request_id,
                status='dropoff_started',
                metadata={'vehicle_id': vehicle_id, 'dropoff_start_time': self.context.current_time}
            )

            # Update vehicle status to UNLOADING
            self.state_manager.vehicle_worker.update_vehicle_state(
                vehicle_id,
                VehicleStatus.UNLOADING,
                {'current_passengers': [passenger_id]}
            )

            # Schedule dropoff completion based on alighting time
            dropoff_duration = self.config.passenger.alighting_time  # seconds
            dropoff_completed_event = Event(
                event_type=EventType.PASSENGER_DROPOFF_COMPLETED,
                timestamp=self.context.current_time + timedelta(seconds=dropoff_duration),
                priority=EventPriority.HIGH,
                passenger_id=passenger_id,
                request_id=request_id,
                vehicle_id=vehicle_id,
                data={'dropoff_location': passenger_state.dropoff_point.to_dict()}
            )

            self.logger.info(f"Scheduled PASSENGER_DROPOFF_COMPLETED for Passenger {passenger_id} at {dropoff_completed_event.timestamp}")

            return [dropoff_completed_event]

        except Exception as e:
            self.logger.error(f"Failed to handle REQUEST_DROPOFF_STARTED for Passenger {passenger_id}: {str(e)}")
            return self._create_error_event(event, e)

    def handle_pickup_completed(self, event: Event) -> Optional[List[Event]]:
        """
        Handle the completion of the pickup process for a request.
        """
        passenger_id = event.passenger_id
        request_id = event.request_id
        vehicle_id = event.vehicle_id
        try:
            self.logger.info(f"Completed pickup for Passenger {passenger_id} with Vehicle {vehicle_id} for Request ID: {request_id}")

            # Retrieve PassengerState
            passenger_state = self.state_manager.passenger_worker.get_passenger_state(passenger_id)
            if not passenger_state:
                raise ValueError(f"PassengerState for Passenger ID: {passenger_id} does not exist.")

            # Update passenger status to IN_VEHICLE
            updates = {
                'status': PassengerStatus.IN_VEHICLE.value,
                'boarding_end': self.context.current_time,
                'in_vehicle_start': self.context.current_time,
                'assigned_vehicle': vehicle_id,
                'total_wait_time': (self.context.current_time - passenger_state.waiting_start).total_seconds()
            }
            self.state_manager.passenger_worker.update_state({passenger_id: updates})

            # Update request status to IN_VEHICLE
            self.state_manager.request_worker.update_request_status(
                request_id,
                status='in_vehicle',
                metadata={'in_vehicle_time': self.context.current_time, 'vehicle_id': vehicle_id}
            )

            # Update vehicle status to OCCUPIED
            self.state_manager.vehicle_worker.update_vehicle_state(
                vehicle_id,
                VehicleStatus.OCCUPIED,
                {'current_passengers': [passenger_id], 'next_destination': passenger_state.destination.to_dict()}
            )

            # Schedule vehicle departure towards dropoff
            departure_delay = self.config.vehicle.departure_delay  # seconds
            departure_event = Event(
                event_type=EventType.VEHICLE_DEPARTED,
                timestamp=self.context.current_time + timedelta(seconds=departure_delay),
                priority=EventPriority.HIGH,
                vehicle_id=vehicle_id,
                data={
                    'destination': passenger_state.destination.to_dict(),
                    'purpose': 'dropoff'
                }
            )

            self.logger.info(f"Scheduled VEHICLE_DEPARTED for Vehicle {vehicle_id} at {departure_event.timestamp}")

            return [departure_event]

        except Exception as e:
            self.logger.error(f"Failed to handle PICKUP_COMPLETED for Passenger {passenger_id}: {str(e)}")
            return self._create_error_event(event, e)

    def handle_request_dropoff_completed(self, event: Event) -> Optional[List[Event]]:
        """
        Handle the completion of the dropoff process for a request.
        """
        passenger_id = event.passenger_id
        request_id = event.request_id
        vehicle_id = event.vehicle_id
        try:
            self.logger.info(f"Completed dropoff for Passenger {passenger_id} with Vehicle {vehicle_id} for Request ID: {request_id}")

            # Retrieve PassengerState
            passenger_state = self.state_manager.passenger_worker.get_passenger_state(passenger_id)
            if not passenger_state:
                raise ValueError(f"PassengerState for Passenger ID: {passenger_id} does not exist.")

            # Update passenger status to WALKING_TO_DESTINATION
            updates = {
                'status': PassengerStatus.WALKING_TO_DESTINATION.value,
                'dropoff_end_time': self.context.current_time,
                'walking_to_destination_start': self.context.current_time,
                'current_location': passenger_state.dropoff_point.to_dict(),
                'total_in_vehicle_time': (self.context.current_time - passenger_state.in_vehicle_start).total_seconds()
            }
            self.state_manager.passenger_worker.update_state({passenger_id: updates})

            # Update request status to COMPLETED
            self.state_manager.request_worker.update_request_status(
                request_id,
                status='completed',
                metadata={
                    'dropoff_time': self.context.current_time,
                    'dropoff_location': passenger_state.dropoff_point.to_dict(),
                    'service_level_violations': passenger_state.service_level_violations,
                    'total_wait_time': passenger_state.total_wait_time,
                    'total_in_vehicle_time': passenger_state.total_in_vehicle_time,
                    'total_journey_time': self._calculate_total_journey_time(passenger_state)
                }
            )

            # Update vehicle status to IDLE or adjust based on other assignments
            self.state_manager.vehicle_worker.update_vehicle_state(
                vehicle_id,
                VehicleStatus.IDLE,
                {
                    'current_passengers': [],
                    'last_service': request_id,
                    'next_destination': None
                }
            )

            # Schedule arrival at destination based on walking speed
            walking_duration = self._calculate_walking_duration(passenger_state.dropoff_point, passenger_state.destination)
            arrival_event = Event(
                event_type=EventType.PASSENGER_ARRIVED_AT_DESTINATION,
                timestamp=self.context.current_time + timedelta(seconds=walking_duration),
                priority=EventPriority.NORMAL,
                passenger_id=passenger_id,
                request_id=request_id,
                data={'destination_location': passenger_state.destination.to_dict()}
            )

            self.logger.info(f"Scheduled PASSENGER_ARRIVED_AT_DESTINATION for Passenger {passenger_id} at {arrival_event.timestamp}")

            return [arrival_event]

        except Exception as e:
            self.logger.error(f"Failed to handle REQUEST_DROPOFF_COMPLETED for Passenger {passenger_id}: {str(e)}")
            return self._create_error_event(event, e)

    def handle_passenger_walking_to_destination(self, event: Event) -> Optional[List[Event]]:
        """
        Handle a passenger walking towards their final destination.
        """
        passenger_id = event.passenger_id
        request_id = event.request_id
        try:
            self.logger.info(f"Passenger {passenger_id} is walking to destination for Request ID: {request_id}")

            # Retrieve PassengerState
            passenger_state = self.state_manager.passenger_worker.get_passenger_state(passenger_id)
            if not passenger_state:
                raise ValueError(f"PassengerState for Passenger ID: {passenger_id} does not exist.")

            # Update passenger status to WALKING_TO_DESTINATION
            updates = {
                'status': PassengerStatus.WALKING_TO_DESTINATION.value,
                'walking_to_destination_start': self.context.current_time,
                'current_location': passenger_state.destination.to_dict()
            }
            self.state_manager.passenger_worker.update_state({passenger_id: updates})

            # Calculate walking duration
            walking_duration = self._calculate_walking_duration(passenger_state.destination, passenger_state.destination)

            # Schedule arrival at destination
            arrival_event = Event(
                event_type=EventType.PASSENGER_ARRIVED_AT_DESTINATION,
                timestamp=self.context.current_time + timedelta(seconds=walking_duration),
                priority=EventPriority.NORMAL,
                passenger_id=passenger_id,
                request_id=request_id,
                data={'destination_location': passenger_state.destination.to_dict()}
            )

            self.logger.info(f"Scheduled PASSENGER_ARRIVED_AT_DESTINATION for Passenger {passenger_id} at {arrival_event.timestamp}")

            return [arrival_event]

        except Exception as e:
            self.logger.error(f"Failed to handle PASSENGER_WALKING_TO_DESTINATION for Passenger {passenger_id}: {str(e)}")
            return self._create_error_event(event, e)

    def handle_passenger_arrived_at_destination(self, event: Event) -> Optional[List[Event]]:
        """
        Handle a passenger arriving at their final destination.
        """
        passenger_id = event.passenger_id
        request_id = event.request_id
        try:
            self.logger.info(f"Passenger {passenger_id} arrived at destination for Request ID: {request_id}")

            # Retrieve PassengerState
            passenger_state = self.state_manager.passenger_worker.get_passenger_state(passenger_id)
            if not passenger_state:
                raise ValueError(f"PassengerState for Passenger ID: {passenger_id} does not exist.")

            # Update passenger status to COMPLETED
            updates = {
                'status': PassengerStatus.COMPLETED.value,
                'walking_to_destination_end': self.context.current_time,
                'completion_time': self.context.current_time,
                'total_journey_time': (self.context.current_time - passenger_state.walking_to_pickup_start).total_seconds()
            }
            self.state_manager.passenger_worker.update_state({passenger_id: updates})

            # Update request status to COMPLETED
            self.state_manager.request_worker.update_request_status(
                request_id,
                status='completed',
                metadata={
                    'arrival_time': self.context.current_time,
                    'destination_location': passenger_state.destination.to_dict(),
                    'service_level_violations': passenger_state.service_level_violations,
                    'total_wait_time': passenger_state.total_wait_time,
                    'total_in_vehicle_time': passenger_state.total_in_vehicle_time,
                    'total_journey_time': passenger_state.total_journey_time
                }
            )

            # Optionally, delete passenger state if no longer needed
            # self.state_manager.passenger_worker.delete_passenger_state(passenger_id)

            return None

        except Exception as e:
            self.logger.error(f"Failed to handle PASSENGER_ARRIVED_AT_DESTINATION for Passenger {passenger_id}: {str(e)}")
            return self._create_error_event(event, e)

    # Additional Event Handlers can be implemented similarly

    # Helper Methods

    def _calculate_walking_duration(self, origin: Location, destination: Location) -> int:
        """
        Calculate walking duration based on distance and walking speed.
        Placeholder implementation; replace with actual distance calculation.
        """
        walking_speed_kmh = self.config.passenger.walking_speed_kmh  # e.g., 5 km/h
        walking_speed_mps = walking_speed_kmh * 1000 / 3600  # Convert to m/s

        # Placeholder distance calculation (in meters)
        # Replace with actual geospatial distance between origin and destination
        distance = self._calculate_distance(origin, destination)  # meters

        duration = distance / walking_speed_mps
        return int(duration)

    def _calculate_distance(self, loc1: Location, loc2: Location) -> float:
        """
        Calculate the Euclidean distance between two locations.
        Replace with actual geospatial distance calculation if needed.
        """
        from math import radians, sin, cos, sqrt, atan2

        # Haversine formula for more accurate distance
        R = 6371e3  # Earth radius in meters
        phi1 = radians(loc1.latitude)
        phi2 = radians(loc2.latitude)
        delta_phi = radians(loc2.latitude - loc1.latitude)
        delta_lambda = radians(loc2.longitude - loc1.longitude)

        a = sin(delta_phi / 2) ** 2 + cos(phi1) * cos(phi2) * sin(delta_lambda / 2) ** 2
        c = 2 * atan2(sqrt(a), sqrt(1 - a))

        distance = R * c  # in meters
        return distance

    def _calculate_total_journey_time(self, passenger_state: PassengerState) -> Optional[float]:
        """
        Calculate total journey time for a passenger.
        """
        if passenger_state.walking_to_pickup_start and passenger_state.walking_to_destination_end:
            journey_time = (passenger_state.walking_to_destination_end - passenger_state.walking_to_pickup_start).total_seconds()
            self.logger.debug(f"Calculated total journey time for Passenger {passenger_state.id}: {journey_time} seconds")
            return journey_time
        return None

    def _create_error_event(self, original_event: Event, error: Exception) -> List[Event]:
        """
        Create a SIMULATION_ERROR event based on the original event and exception.
        """
        return [
            Event(
                event_type=EventType.SIMULATION_ERROR,
                timestamp=self.context.current_time,
                priority=EventPriority.CRITICAL,
                data={
                    'original_event_id': original_event.id,
                    'original_event_type': original_event.event_type.value,
                    'error_message': str(error),
                    'error_type': error.__class__.__name__
                }
            )
        ]
