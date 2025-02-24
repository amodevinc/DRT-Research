from datetime import datetime, timedelta
from copy import deepcopy
from typing import Dict, Any, Optional, List, Tuple
import traceback
from drt_sim.models.route import Route, RouteStop, RouteStatus, RouteSegment
from drt_sim.models.vehicle import Vehicle, VehicleStatus
from drt_sim.models.stop import Stop, StopAssignment
from drt_sim.models.location import Location
from drt_sim.network.manager import NetworkManager
from drt_sim.core.simulation.context import SimulationContext
from drt_sim.config.config import ParameterSet, MatchingAssignmentConfig, VehicleConfig
import logging
logger = logging.getLogger(__name__)

class RouteService:
    """Service layer for route operations and business logic"""
    
    def __init__(
        self,
        network_manager: NetworkManager,
        sim_context: SimulationContext,
        config: ParameterSet
    ):
        self.network_manager = network_manager
        self.sim_context = sim_context
        self.config = config

    async def create_new_route(
        self,
        vehicle: Vehicle,
        stop_assignment: StopAssignment
    ) -> Optional[Route]:
        """
        Creates a new route for a vehicle with initial pickup and dropoff stops
        """
        try:
            logger.info(f"Starting new route creation for vehicle {vehicle.id} with stop assignment {stop_assignment.id}")
            
            # Get path info for initial segment (vehicle to pickup)
            initial_path_info = await self._get_segment_with_waypoints(
                vehicle.current_state.current_location,
                stop_assignment.origin_stop.location
            )
            
            if not initial_path_info or initial_path_info['distance'] == float('inf'):
                logger.warning(f"Could not calculate path to pickup for vehicle {vehicle.id}")
                return None

            # Calculate arrival times
            pickup_arrival_time = self.sim_context.current_time + timedelta(
                seconds=initial_path_info['duration']
            )
            
            # Get path info for service segment (pickup to dropoff)
            service_path_info = await self._get_segment_with_waypoints(
                stop_assignment.origin_stop.location,
                stop_assignment.destination_stop.location
            )

            if not service_path_info or service_path_info['distance'] == float('inf'):
                logger.warning(f"Could not calculate path to dropoff for vehicle {vehicle.id}")
                return None
            
            dropoff_arrival_time = pickup_arrival_time + timedelta(
                seconds=vehicle.config.boarding_time + service_path_info['duration']
            )

            logger.debug(f"Creating route stops with pickup arrival at {pickup_arrival_time} and dropoff at {dropoff_arrival_time}")

            # Create route stops
            pickup_stop = RouteStop(
                stop=stop_assignment.origin_stop,
                sequence=0,
                current_load=0,
                pickup_passengers=[stop_assignment.request_id],
                dropoff_passengers=[],
                planned_arrival_time=pickup_arrival_time,
                planned_departure_time=pickup_arrival_time + timedelta(
                    seconds=vehicle.config.boarding_time
                ),
                service_time=vehicle.config.boarding_time
            )

            dropoff_stop = RouteStop(
                stop=stop_assignment.destination_stop,
                sequence=1,
                current_load=1,
                pickup_passengers=[],
                dropoff_passengers=[stop_assignment.request_id],
                planned_arrival_time=dropoff_arrival_time,
                planned_departure_time=dropoff_arrival_time + timedelta(
                    seconds=vehicle.config.alighting_time
                ),
                service_time=vehicle.config.alighting_time
            )

            logger.debug("Creating route segments")
            # Create segments with waypoints
            segments: List[RouteSegment] = []
            
            initial_segment = RouteSegment(
                origin=None,  # No RouteStop for vehicle's current position
                destination=pickup_stop,
                estimated_duration=initial_path_info['duration'],
                estimated_distance=initial_path_info['distance'],
                origin_location=vehicle.current_state.current_location,
                destination_location=stop_assignment.origin_stop.location,
                waypoints=initial_path_info['waypoints']
            )
            segments.append(initial_segment)

            service_segment = RouteSegment(
                origin=pickup_stop,
                destination=dropoff_stop,
                estimated_duration=service_path_info['duration'],
                estimated_distance=service_path_info['distance'],
                origin_location=stop_assignment.origin_stop.location,
                destination_location=stop_assignment.destination_stop.location,
                waypoints=service_path_info['waypoints']
            )
            segments.append(service_segment)

            total_distance = sum(segment.estimated_distance for segment in segments)
            total_duration = sum(segment.estimated_duration for segment in segments) + \
                           vehicle.config.boarding_time + \
                           vehicle.config.alighting_time
                           
            logger.info(f"Created new route for vehicle {vehicle.id} with total distance {total_distance}m and duration {total_duration}s")

            return Route(
                vehicle_id=vehicle.id,
                stops=[pickup_stop, dropoff_stop],
                segments=segments,
                status=RouteStatus.CREATED,
                total_distance=total_distance,
                total_duration=total_duration,
                current_segment_index=0
            )
            
        except Exception as e:
            logger.error(f"Error creating new route for vehicle {vehicle.id}: {traceback.format_exc()}")
            return None

    async def create_modified_route(
        self,
        current_route: Route,
        stop_assignment: StopAssignment,
        pickup_idx: int,
        dropoff_idx: int,
        vehicle: Vehicle
    ) -> Optional[Route]:
        """
        Creates new route with stops inserted at specified positions
        """
        try:
            logger.debug(f"Starting route modification for vehicle {vehicle.id} with stop assignment {stop_assignment.id}")
            logger.debug(f"Attempting insertion at pickup_idx={pickup_idx}, dropoff_idx={dropoff_idx}")
            
            new_route = deepcopy(current_route)
            new_stops = new_route.stops
            
            logger.debug("Finding compatible stops")
            # Find compatible stops or create new ones
            existing_pickup_stop, actual_pickup_idx = self._find_compatible_stop(
                new_stops,
                stop_assignment.origin_stop,
                pickup_idx,
                stop_assignment.request_id,
                True,
                stop_assignment.expected_passenger_origin_stop_arrival_time
            )
            
            existing_dropoff_stop, actual_dropoff_idx = self._find_compatible_stop(
                new_stops,
                stop_assignment.destination_stop,
                dropoff_idx,
                stop_assignment.request_id,
                False
            )
            
            logger.debug(f"Found pickup stop: existing={existing_pickup_stop is not None}, idx={actual_pickup_idx}")
            logger.debug(f"Found dropoff stop: existing={existing_dropoff_stop is not None}, idx={actual_dropoff_idx}")
            
            # Process stops
            logger.debug("Processing pickup stop")
            new_stops = await self._process_pickup_stop(
                new_stops,
                existing_pickup_stop,
                actual_pickup_idx,
                stop_assignment,
                vehicle
            )
            
            logger.debug("Processing dropoff stop")
            new_stops = await self._process_dropoff_stop(
                new_stops,
                existing_dropoff_stop,
                actual_dropoff_idx,
                stop_assignment,
                vehicle
            )
            
            # Update route properties
            logger.debug("Updating stop timings")
            await self._update_stop_timings(new_stops, vehicle)
            
            logger.debug("Updating occupancies")
            self._update_occupancies(new_stops)
            
            logger.debug("Creating new segments")
            # Recreate segments for the modified route
            new_segments: List[RouteSegment] = []
            
            # Handle initial segment from vehicle's current position if needed
            if not new_stops[0].completed:
                logger.debug("Adding initial positioning segment")
                
                # Get current segment if vehicle is en route
                current_segment = None
                if current_route and current_route.segments:
                    current_segment = current_route.get_current_segment()
                
                # If vehicle is in the middle of a segment, use its current position
                if (current_segment and 
                    vehicle.current_state.status == VehicleStatus.IN_SERVICE and 
                    not current_segment.completed):
                    logger.debug(f"Vehicle is currently en route in segment {current_segment.id}")
                    
                    # Get path info from current position to next stop
                    initial_path_info = await self._get_segment_with_waypoints(
                        vehicle.current_state.current_location,
                        new_stops[0].stop.location
                    )
                    
                    if not initial_path_info or initial_path_info['distance'] == float('inf'):
                        logger.warning(f"Could not calculate path from current position to next stop for vehicle {vehicle.id}")
                        return None
                    
                    initial_segment = RouteSegment(
                        origin=None,
                        destination=new_stops[0],
                        estimated_duration=initial_path_info['duration'],
                        estimated_distance=initial_path_info['distance'],
                        origin_location=vehicle.current_state.current_location,
                        destination_location=new_stops[0].stop.location,
                        waypoints=initial_path_info['waypoints']
                    )
                    new_segments.append(initial_segment)
                else:
                    # Vehicle is at a stop or hasn't started, use normal path calculation
                    initial_path_info = await self._get_segment_with_waypoints(
                        vehicle.current_state.current_location,
                        new_stops[0].stop.location
                    )
                    
                    if not initial_path_info or initial_path_info['distance'] == float('inf'):
                        logger.warning(f"Could not calculate path to first stop for vehicle {vehicle.id}")
                        return None
                    
                    initial_segment = RouteSegment(
                        origin=None,
                        destination=new_stops[0],
                        estimated_duration=initial_path_info['duration'],
                        estimated_distance=initial_path_info['distance'],
                        origin_location=vehicle.current_state.current_location,
                        destination_location=new_stops[0].stop.location,
                        waypoints=initial_path_info['waypoints']
                    )
                    new_segments.append(initial_segment)
            
            # Add segments between stops
            logger.debug("Adding inter-stop segments")
            for i in range(len(new_stops) - 1):
                origin_stop = new_stops[i]
                destination_stop = new_stops[i + 1]
                
                # Skip if both stops are completed
                if origin_stop.completed and destination_stop.completed:
                    logger.debug(f"Skipping completed segment between stops {i} and {i+1}")
                    continue
                
                # Get path info with waypoints
                path_info = await self._get_segment_with_waypoints(
                    origin_stop.stop.location,
                    destination_stop.stop.location
                )
                
                if not path_info or path_info['distance'] == float('inf'):
                    logger.warning(f"Could not calculate path between stops {i} and {i+1}")
                    return None
                
                logger.debug(f"Segment {i}: distance={path_info['distance']}m, "
                          f"duration={path_info['duration']}s, "
                          f"waypoints={len(path_info['waypoints'])}")
                
                segment = RouteSegment(
                    origin=origin_stop,
                    destination=destination_stop,
                    estimated_duration=path_info['duration'],
                    estimated_distance=path_info['distance'],
                    origin_location=origin_stop.stop.location,
                    destination_location=destination_stop.stop.location,
                    completed=origin_stop.completed and destination_stop.completed,
                    waypoints=path_info['waypoints']
                )
                new_segments.append(segment)
            
            new_route.stops = new_stops
            new_route.segments = new_segments
            new_route.total_distance = sum(segment.estimated_distance for segment in new_segments)
            new_route.total_duration = sum(segment.estimated_duration for segment in new_segments) + \
                                     sum(stop.service_time for stop in new_stops)

            new_route.current_segment_index = 0
            for i, segment in enumerate(new_segments):
                if not segment.completed:
                    new_route.current_segment_index = i
                    break

            
            # Validate route consistency
            is_valid, error_msg = new_route.validate_passenger_consistency()
            if not is_valid:
                logger.warning(f"Route is not valid, not being considered for assignment. Error: {error_msg}")
                return None
                
            logger.debug(f"Successfully modified route for vehicle {vehicle.id}: total_distance={new_route.total_distance}m, total_duration={new_route.total_duration}s")
            return new_route
            
        except Exception as e:
            logger.error(f"Error modifying route for vehicle {vehicle.id}: {traceback.format_exc()}")
            return None

    def validate_route_constraints(
        self,
        route: Route,
        vehicle: Vehicle,
        stop_assignment: StopAssignment,
        config: MatchingAssignmentConfig
    ) -> bool:
        """
        Validates if route meets all operational constraints
        
        Args:
            route: Route to validate
            vehicle: Vehicle operating route
            stop_assignment: Stop assignment being validated
            config: Configuration with constraints
            
        Returns:
            True if route is valid, False otherwise
        """
        try:
            # Check vehicle capacity
            if not self._validate_capacity(route, vehicle.capacity):
                return False
                
            # Get relevant stops
            pickup_stop = self._find_stop_for_request(
                route, 
                stop_assignment.request_id,
                is_pickup=True
            )
            dropoff_stop = self._find_stop_for_request(
                route,
                stop_assignment.request_id,
                is_pickup=False
            )
            
            if not pickup_stop or not dropoff_stop:
                return False
            
            # Validate time constraints
            return self._validate_time_constraints(
                pickup_stop,
                dropoff_stop,
                stop_assignment,
                config
            )
            
        except Exception as e:
            logger.error(f"Error validating route: {str(e)}")
            return False

    def calculate_route_metrics(
        self,
        original_route: Route,
        new_route: Route,
        stop_assignment: StopAssignment
    ) -> Dict[str, float]:
        """
        Calculates metrics for route modification
        
        Args:
            original_route: Route before modification
            new_route: Route after modification
            stop_assignment: Stop assignment being inserted
            
        Returns:
            Dictionary of metric values
        """
        try:
            pickup_stop = self._find_stop_for_request(
                new_route,
                stop_assignment.request_id,
                is_pickup=True
            )
            dropoff_stop = self._find_stop_for_request(
                new_route,
                stop_assignment.request_id,
                is_pickup=False
            )
            
            if not pickup_stop or not dropoff_stop:
                raise ValueError("Cannot find pickup or dropoff stop")

            # Calculate metrics
            return {
                "waiting_time": self._calculate_waiting_time(
                    pickup_stop,
                    stop_assignment
                ),
                "ride_time": self._calculate_ride_time(
                    pickup_stop,
                    dropoff_stop
                ),
                "delay": self._calculate_delay_impact(
                    original_route,
                    new_route
                ),
                "distance": new_route.total_distance,
                "duration": new_route.total_duration,
                "distance_added": max(0, new_route.total_distance - original_route.total_distance),
                "duration_added": max(0, new_route.total_duration - original_route.total_duration)
            }
            
        except Exception as e:
            logger.error(f"Error calculating metrics: {str(e)}")
            return {}

    def get_next_stops(
        self,
        route: Route,
        count: int = 1
    ) -> List[RouteStop]:
        """
        Gets next uncompleted stops in route
        
        Args:
            route: Route to get stops from
            count: Number of stops to get
            
        Returns:
            List of next uncompleted stops
        """
        try:
            return [
                stop for stop in route.stops
                if not stop.completed
            ][:count]
        except Exception as e:
            logger.error(f"Error getting next stops: {str(e)}")
            return []

    # === Private Helper Methods ===

    def _find_compatible_stop(
        self,
        stops: List[RouteStop],
        target_stop: Stop,
        preferred_idx: int,
        request_id: str,
        is_pickup: bool,
        expected_passenger_origin_stop_arrival_time: Optional[datetime] = None
    ) -> Tuple[Optional[RouteStop], int]:
        """
        Finds existing compatible stop or determines insertion point.
        
        A stop is compatible if:
        1. It has the same physical location (stop.id)
        2. It hasn't been completed yet
        3. For pickups: The passenger's arrival time is compatible with the stop's timing
        4. For dropoffs: The stop occurs after the pickup
        """
        # Look for existing compatible stop
        for i, stop in enumerate(stops):
            if (
                stop.stop.id == target_stop.id and
                not stop.completed and
                self._is_stop_time_compatible(stop, is_pickup, expected_passenger_origin_stop_arrival_time)
            ):
                logger.debug(f"Found compatible {'pickup' if is_pickup else 'dropoff'} stop at index {i}")
                return stop, i
                
        logger.debug(f"No compatible {'pickup' if is_pickup else 'dropoff'} stop found, will insert at index {preferred_idx}")
        return None, preferred_idx

    async def _process_pickup_stop(
        self,
        stops: List[RouteStop],
        existing_stop: Optional[RouteStop],
        index: int,
        stop_assignment: StopAssignment,
        vehicle: Vehicle
    ) -> List[RouteStop]:
        """Process pickup stop insertion or modification"""
        if existing_stop:
            # If the stop is already in the list, just update it, don't insert again
            if stop_assignment.request_id not in existing_stop.pickup_passengers:
                existing_stop.pickup_passengers.append(stop_assignment.request_id)
                existing_stop.service_time = self._calculate_stop_service_time(
                    existing_stop, 
                    vehicle.config
                )
            # Don't insert if the stop is already in the list
            if existing_stop not in stops:
                stops.insert(index, existing_stop)
        else:
            new_stop = RouteStop(
                stop=stop_assignment.origin_stop,
                sequence=index,
                pickup_passengers=[stop_assignment.request_id],
                dropoff_passengers=[],
                service_time=vehicle.config.boarding_time,
                current_load=0  # Will be updated by _update_occupancies
            )
            stops.insert(index, new_stop)
        return stops

    async def _process_dropoff_stop(
        self,
        stops: List[RouteStop],
        existing_stop: Optional[RouteStop],
        index: int,
        stop_assignment: StopAssignment,
        vehicle: Vehicle
    ) -> List[RouteStop]:
        """Process dropoff stop insertion or modification"""
        # First, find if and where this passenger's pickup occurs
        pickup_index = -1
        for i, stop in enumerate(stops):
            if stop_assignment.request_id in stop.pickup_passengers:
                pickup_index = i
                break
        
        # If we found a pickup and we're trying to insert the dropoff before it,
        # adjust the index to be after the pickup
        if pickup_index != -1 and index <= pickup_index:
            index = pickup_index + 1
            logger.debug(f"Adjusted dropoff index to {index} to maintain pickup-before-dropoff order")

        if existing_stop:
            # Only add to existing stop if it's after the pickup
            if pickup_index == -1 or index > pickup_index:
                if stop_assignment.request_id not in existing_stop.dropoff_passengers:
                    existing_stop.dropoff_passengers.append(stop_assignment.request_id)
                    existing_stop.service_time = self._calculate_stop_service_time(
                        existing_stop, 
                        vehicle.config
                    )
                # Don't insert if the stop is already in the list
                if existing_stop not in stops:
                    stops.insert(index, existing_stop)
            else:
                logger.warning(f"Prevented adding dropoff for {stop_assignment.request_id} at index {index} before pickup at {pickup_index}")
        else:
            # Only create new stop if it's after the pickup
            if pickup_index == -1 or index > pickup_index:
                new_stop = RouteStop(
                    stop=stop_assignment.destination_stop,
                    sequence=index,
                    pickup_passengers=[],
                    dropoff_passengers=[stop_assignment.request_id],
                    service_time=vehicle.config.alighting_time,
                    current_load=0  # Will be updated by _update_occupancies
                )
                stops.insert(index, new_stop)
            else:
                logger.warning(f"Prevented creating dropoff for {stop_assignment.request_id} at index {index} before pickup at {pickup_index}")
        
        return stops
    
    async def _update_stop_timings(
        self,
        stops: List[RouteStop],
        vehicle: Vehicle
    ) -> None:
        """Updates arrival and departure times for all stops in route sequence"""
        try:
            current_time = self.sim_context.current_time
            current_location = vehicle.current_state.current_location
            last_departure_time = current_time

            for i, stop in enumerate(stops):
                if stop.completed:
                    last_departure_time = stop.actual_departure_time or stop.planned_departure_time
                    current_location = stop.stop.location
                    continue

                # Calculate travel time from previous location
                travel_time = await self.network_manager.calculate_travel_time(
                    current_location,
                    stop.stop.location
                )
                
                if travel_time == float('inf'):
                    raise ValueError(f"Unable to calculate travel time to stop {stop.stop.id}")

                # Update arrival time based on previous departure
                stop.planned_arrival_time = last_departure_time + timedelta(seconds=travel_time)
                
                # Calculate service time based on passengers
                stop.service_time = self._calculate_stop_service_time(stop, vehicle.config)
                
                # Update departure time considering service time
                stop.planned_departure_time = stop.planned_arrival_time + timedelta(
                    seconds=min(stop.service_time, vehicle.config.max_dwell_time)
                )
                
                # Update tracking variables
                last_departure_time = stop.planned_departure_time
                current_location = stop.stop.location

        except Exception as e:
            logger.error(f"Error updating stop timings: {traceback.format_exc()}")
            raise

    def _update_occupancies(self, stops: List[RouteStop]) -> None:
        """Updates occupancy tracking for all stops"""
        current_occupancy = 0
        
        for stop in stops:
            if not stop.completed:
                current_occupancy += len(stop.pickup_passengers)
                current_occupancy -= len(stop.dropoff_passengers)
                stop.current_load = current_occupancy

    async def _calculate_route_distance(self, stops: List[RouteStop]) -> float:
        """Calculates total distance of route"""
        total_distance = 0.0
        
        for i in range(len(stops) - 1):
            distance = await self.network_manager.calculate_distance(
                stops[i].stop.location,
                stops[i + 1].stop.location
            )
            total_distance += distance
            
        return total_distance

    def _calculate_route_duration(self, stops: List[RouteStop]) -> float:
        """Calculates total duration in minutes"""
        if not stops:
            return 0.0
        
        first_stop = stops[0]
        last_stop = stops[-1]
        
        if first_stop.planned_arrival_time and last_stop.planned_departure_time:
            return (
                last_stop.planned_departure_time - first_stop.planned_arrival_time
            ).total_seconds() / 60
            
        return 0.0

    def _validate_capacity(self, route: Route, max_capacity: int) -> bool:
        """Validates vehicle capacity constraints"""
        for stop in route.stops:
            if stop.current_load > max_capacity:
                return False
        return True

    def _validate_time_constraints(
        self,
        pickup_stop: RouteStop,
        dropoff_stop: RouteStop,
        stop_assignment: StopAssignment,
        config: MatchingAssignmentConfig
    ) -> bool:
        """Validates timing constraints for pickup and dropoff"""
        try:
            # Check passenger arrival feasibility
            passenger_arrival = stop_assignment.expected_passenger_origin_stop_arrival_time
            
            # Validate waiting time
            waiting_time = (
                pickup_stop.planned_arrival_time - passenger_arrival
            ).total_seconds() / 60
            
            if waiting_time > config.max_waiting_time_mins:
                return False
                
            # Validate ride time
            ride_time = (
                dropoff_stop.planned_arrival_time - pickup_stop.planned_departure_time
            ).total_seconds() / 60
            
            if ride_time > config.max_in_vehicle_time_mins:
                return False
                
            return True
            
        except Exception as e:
            logger.error(f"Error validating time constraints: {str(e)}")
            return False

    def _calculate_stop_service_time(
        self,
        stop: RouteStop,
        vehicle_config: VehicleConfig
    ) -> int:
        """Calculates service time based on passengers"""
        return (
            len(stop.pickup_passengers) * vehicle_config.boarding_time +
            len(stop.dropoff_passengers) * vehicle_config.alighting_time
        )

    def _find_stop_for_request(
        self,
        route: Route,
        request_id: str,
        is_pickup: bool
    ) -> Optional[RouteStop]:
        """Finds stop containing request"""
        for stop in route.stops:
            if is_pickup and request_id in stop.pickup_passengers:
                return stop
            elif not is_pickup and request_id in stop.dropoff_passengers:
                return stop
        return None

    def _calculate_waiting_time(
        self,
        pickup_stop: RouteStop,
        stop_assignment: StopAssignment
    ) -> float:
        """Calculates passenger waiting time in minutes"""
        passenger_arrival = (
            stop_assignment.assignment_time + 
            timedelta(seconds=stop_assignment.walking_time_origin)
        )
        
        waiting_time = (
            pickup_stop.planned_arrival_time - passenger_arrival
        ).total_seconds() / 60
        
        return max(0, waiting_time)

    def _calculate_ride_time(
        self,
        pickup_stop: RouteStop,
        dropoff_stop: RouteStop
    ) -> float:
        """Calculates in-vehicle time in minutes"""
        return (
            dropoff_stop.planned_arrival_time - pickup_stop.planned_departure_time
        ).total_seconds() / 60

    def _calculate_delay_impact(
        self,
        original_route: Route,
        new_route: Route
    ) -> float:
        """Calculates total delay impact on existing passengers"""
        total_delay = 0.0
        original_times = {
            stop.stop.id: stop.planned_arrival_time 
            for stop in original_route.stops
        }
        
        for stop in new_route.stops:
            if stop.stop.id in original_times:
                original_time = original_times[stop.stop.id]
                if original_time and stop.planned_arrival_time:
                    delay = (
                        stop.planned_arrival_time - original_time
                    ).total_seconds() / 60
                    total_delay += max(0, delay)
        
        return total_delay

    def _is_stop_time_compatible(
        self,
        stop: RouteStop,
        is_pickup: bool,
        expected_passenger_origin_stop_arrival_time: Optional[datetime] = None
    ) -> bool:
        """
        Checks if stop timing is compatible for combining
        
        Args:
            stop: Stop to check
            is_pickup: Whether this is a pickup stop
            expected_passenger_origin_stop_arrival_time: Expected arrival time of the passenger
            
        Returns:
            True if stop can be combined, False otherwise
        """
        if stop.completed:
            return False
            
        if not stop.planned_arrival_time:
            return False
        
        if is_pickup and expected_passenger_origin_stop_arrival_time:
            # For pickups, ensure that:
            # 1. The passenger arrives before the vehicle's planned arrival + max dwell time
            # 2. The vehicle doesn't arrive too early before the passenger
            max_wait = timedelta(seconds=self.config.vehicle.max_dwell_time)
            too_early_threshold = timedelta(minutes=5)  # Don't arrive more than 5 mins early
            
            passenger_arrival = expected_passenger_origin_stop_arrival_time
            vehicle_arrival = stop.planned_arrival_time
            
            # Check if timing works for both passenger and vehicle
            return (
                passenger_arrival <= vehicle_arrival + max_wait and
                vehicle_arrival - too_early_threshold <= passenger_arrival
            )
        else:
            # For dropoffs, be more lenient with timing since we've already
            # enforced pickup-before-dropoff order in _process_dropoff_stop
            return True

    def get_route_summary(self, route: Route) -> Dict[str, Any]:
        """
        Gets summary metrics for route
        
        Returns:
            Dictionary containing:
            - total_distance: Total route distance
            - total_duration: Total route duration in minutes
            - total_passengers: Total passengers served
            - completed_stops: Number of completed stops
            - remaining_stops: Number of remaining stops
            - current_occupancy: Current vehicle occupancy
            - on_time_percentage: Percentage of on-time stops
        """
        try:
            completed_stops = sum(1 for stop in route.stops if stop.completed)
            total_stops = len(route.stops)
            
            # Get unique passengers
            unique_passengers = set()
            for stop in route.stops:
                unique_passengers.update(stop.pickup_passengers)
                unique_passengers.update(stop.dropoff_passengers)
            
            # Calculate on-time percentage
            on_time_stops = sum(
                1 for stop in route.stops 
                if stop.completed and 
                stop.actual_arrival_time and 
                stop.planned_arrival_time and
                stop.actual_arrival_time <= stop.planned_arrival_time + timedelta(minutes=5)
            )
            
            on_time_percentage = (
                (on_time_stops / completed_stops * 100)
                if completed_stops > 0 else 100
            )
            
            return {
                "total_distance": route.total_distance,
                "total_duration": route.total_duration,
                "total_passengers": len(unique_passengers),
                "completed_stops": completed_stops,
                "remaining_stops": total_stops - completed_stops,
                "current_occupancy": route.stops[-1].current_load if route.stops else 0,
                "on_time_percentage": on_time_percentage
            }
            
        except Exception as e:
            logger.error(f"Error getting route summary: {str(e)}")
            return {}

    def estimate_completion_time(self, route: Route) -> Optional[datetime]:
        """Estimates when route will be completed"""
        try:
            remaining_stops = [stop for stop in route.stops if not stop.completed]
            if not remaining_stops:
                return None
                
            last_stop = remaining_stops[-1]
            return last_stop.planned_departure_time
            
        except Exception as e:
            logger.error(f"Error estimating completion time: {str(e)}")
            return None

    async def _get_segment_with_waypoints(
        self,
        origin: Location,
        destination: Location
    ) -> Dict[str, Any]:
        """Get detailed path information including waypoints."""
        path_info = await self.network_manager.get_path_info(
            origin,
            destination
        )
        
        # Extract waypoint locations from path_info
        waypoints = [w['location'] for w in path_info.get('waypoints', [])]
        
        return {
            'distance': path_info['distance'],
            'duration': path_info['duration'],
            'waypoints': waypoints,
            'path': path_info.get('path', [])
        }