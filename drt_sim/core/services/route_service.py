from datetime import datetime, timedelta
from copy import deepcopy
from typing import Dict, Any, Optional, List, Tuple
import traceback
import logging

from drt_sim.models.route import Route, RouteStop, RouteStatus
from drt_sim.models.vehicle import Vehicle, VehicleStatus
from drt_sim.models.stop import Stop, StopAssignment
from drt_sim.models.location import Location
from drt_sim.network.manager import NetworkManager
from drt_sim.core.simulation.context import SimulationContext
from drt_sim.config.config import ParameterSet, MatchingAssignmentConfig, VehicleConfig

logger = logging.getLogger(__name__)

class RouteService:
    """Service layer for route operations and business logic, simplified to work without segments"""
    
    def __init__(
        self,
        network_manager: NetworkManager,
        sim_context: SimulationContext,
        config: ParameterSet
    ):
        self.network_manager = network_manager
        self.sim_context = sim_context
        self.config = config

    async def _get_path_with_waypoints(
        self,
        origin: Location,
        destination: Location
    ) -> Optional[Dict[str, Any]]:
        """
        Get detailed path information including waypoints.
        
        Args:
            origin: Origin location
            destination: Destination location
            
        Returns:
            Dictionary with path details or None if path couldn't be calculated
        """
        try:
            path_info = await self.network_manager.get_path_info(
                origin,
                destination
            )
            
            if not path_info or path_info.get('distance') == float('inf'):
                logger.warning(f"Could not find path from {origin.id} to {destination.id}")
                return None
            
            # Extract waypoint locations from path_info
            waypoints = [w['location'] for w in path_info.get('waypoints', [])]
            
            return {
                'distance': path_info['distance'],
                'duration': path_info['duration'],
                'waypoints': waypoints,
                'path': path_info.get('path', [])
            }
        except Exception as e:
            logger.error(f"Error getting path with waypoints: {traceback.format_exc()}")
            return None
        
    async def create_new_route(
        self,
        vehicle: Vehicle,
        stop_assignment: StopAssignment
    ) -> Optional[Route]:
        """
        Creates a new route for a vehicle with initial pickup and dropoff stops.
        Simplified to work without segments.
        
        Args:
            vehicle: Vehicle to create route for
            stop_assignment: Initial stop assignment for the route
            
        Returns:
            New Route object or None if creation failed
        """
        try:
            logger.info(f"Starting new route creation for vehicle {vehicle.id} with stop assignment {stop_assignment.id}")
            
            # Get path info from vehicle to pickup location
            initial_path_info = await self._get_path_with_waypoints(
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
            
            # Get path info from pickup to dropoff
            service_path_info = await self._get_path_with_waypoints(
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

            # Create pickup stop with travel information
            pickup_stop = RouteStop(
                stop=stop_assignment.origin_stop,
                sequence=0,
                current_load=0,
                pickup_passengers=[stop_assignment.request_id],
                dropoff_passengers=[],
                planned_arrival_time=pickup_arrival_time,
                planned_departure_time=pickup_arrival_time + timedelta(seconds=vehicle.config.boarding_time),
                service_time=vehicle.config.boarding_time,
                origin_location=vehicle.current_state.current_location,
                estimated_duration_to_stop=initial_path_info['duration'],
                estimated_distance_to_stop=initial_path_info['distance']
            )

            # Create dropoff stop with travel information
            dropoff_stop = RouteStop(
                stop=stop_assignment.destination_stop,
                sequence=1,
                current_load=1,
                pickup_passengers=[],
                dropoff_passengers=[stop_assignment.request_id],
                planned_arrival_time=dropoff_arrival_time,
                planned_departure_time=dropoff_arrival_time + timedelta(seconds=vehicle.config.alighting_time),
                service_time=vehicle.config.alighting_time,
                origin_location=stop_assignment.origin_stop.location,
                estimated_duration_to_stop=service_path_info['duration'],
                estimated_distance_to_stop=service_path_info['distance']
            )

            # Create the route with just stops
            route = Route(
                vehicle_id=vehicle.id,
                stops=[pickup_stop, dropoff_stop],
                status=RouteStatus.CREATED,
                current_stop_index=0
            )
            
            # Calculate totals
            route.recalculate_total_distance()
            route.recalculate_total_duration()
            
            # Validate route integrity
            is_valid, error_msg = route.validate_route_integrity()
            if not is_valid:
                logger.error(f"Route integrity validation failed: {error_msg}")
                return None
            
            logger.info(f"Created new route for vehicle {vehicle.id} with total distance {route.total_distance}m and duration {route.total_duration}s")
            return route
            
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
        Creates new route with stops inserted at specified positions.
        Simplified to work without segments.
        
        Args:
            current_route: Existing route to modify
            stop_assignment: New stop assignment to add
            pickup_idx: Index to insert pickup stop
            dropoff_idx: Index to insert dropoff stop
            vehicle: Vehicle operating the route
            
        Returns:
            Modified Route or None if modification failed
        """
        try:
            logger.debug(f"Starting route modification for vehicle {vehicle.id} with stop assignment {stop_assignment.id}")
            logger.debug(f"Attempting insertion at pickup_idx={pickup_idx}, dropoff_idx={dropoff_idx}")
            
            # Create a deep copy of the route to modify
            new_route = deepcopy(current_route)
            
            # Store current vehicle location and active stop for proper state transition
            active_stop = new_route.get_active_stop()
            vehicle_location = vehicle.current_state.current_location
            active_stop_id = new_route.active_stop_id
            
            # Process stops first
            new_stops = await self._process_stops_for_assignment(
                new_route.stops, 
                stop_assignment,
                pickup_idx,
                dropoff_idx,
                vehicle
            )
            
            if not new_stops:
                logger.warning("Failed to process stops for assignment")
                return None
            
            # Update the route with new stops
            new_route.stops = new_stops
            
            # Update occupancies
            self._update_occupancies(new_stops)
            
            # Update stop timings and travel information
            await self._update_stop_timings_and_travel(new_stops, vehicle)
            
            # Update route properties
            new_route.recalculate_total_distance()
            new_route.recalculate_total_duration()
            
            # Update route processing state
            if active_stop_id:
                # Transfer state from old active stop to new one
                new_route.update_for_reroute(vehicle_location, active_stop_id)
            else:
                # Just recalculate current stop index if no active stop
                new_route.recalc_current_stop_index()
            
            # Mark the route as modified
            new_route.mark_as_modified()
            
            # Validate route consistency
            is_valid, error_msg = new_route.validate_passenger_consistency()
            if not is_valid:
                logger.warning(f"Invalid passenger consistency in modified route: {error_msg}")
                return None
                
            # Validate capacity
            is_valid, error_msg = new_route.validate_capacity(vehicle.capacity)
            if not is_valid:
                logger.warning(f"Vehicle capacity exceeded in modified route: {error_msg}")
                return None
                
            # Final route integrity validation
            is_valid, error_msg = new_route.validate_route_integrity()
            if not is_valid:
                logger.warning(f"Route integrity validation failed: {error_msg}")
                return None
                
            logger.debug(f"Successfully modified route for vehicle {vehicle.id}: total_distance={new_route.total_distance}m, total_duration={new_route.total_duration}s")
            return new_route
            
        except Exception as e:
            logger.error(f"Error modifying route for vehicle {vehicle.id}: {traceback.format_exc()}")
            return None

    async def _process_stops_for_assignment(
        self,
        stops: List[RouteStop],
        stop_assignment: StopAssignment,
        pickup_idx: int,
        dropoff_idx: int,
        vehicle: Vehicle
    ) -> Optional[List[RouteStop]]:
        """
        Process stops for a new assignment, finding compatible stops or creating new ones
        Takes extra care to handle completed stops and preserve route state
        """
        try:
            # Find compatible stops or get insertion points, accounting for completed stops
            existing_pickup_stop, actual_pickup_idx = self._find_compatible_stop(
                stops,
                stop_assignment.origin_stop,
                pickup_idx,
                stop_assignment.request_id,
                True,
                stop_assignment.expected_passenger_origin_stop_arrival_time
            )
            
            existing_dropoff_stop, actual_dropoff_idx = self._find_compatible_stop(
                stops,
                stop_assignment.destination_stop,
                dropoff_idx,
                stop_assignment.request_id,
                False
            )
            
            logger.debug(f"Found pickup stop: existing={existing_pickup_stop is not None}, idx={actual_pickup_idx}")
            logger.debug(f"Found dropoff stop: existing={existing_dropoff_stop is not None}, idx={actual_dropoff_idx}")
            
            # Keep track of original stops for recovery if needed
            original_stops = deepcopy(stops)
            
            # Process pickup stop - either modify existing or create new
            if existing_pickup_stop:
                if stop_assignment.request_id not in existing_pickup_stop.pickup_passengers:
                    existing_pickup_stop.add_pickup(stop_assignment.request_id)
                    existing_pickup_stop.service_time = self._calculate_stop_service_time(
                        existing_pickup_stop, 
                        vehicle.config
                    )
                # Ensure stop is at the right position
                if existing_pickup_stop in stops:
                    # Remove from current position
                    stops.remove(existing_pickup_stop)
                # Insert at the target position
                stops.insert(actual_pickup_idx, existing_pickup_stop)
            else:
                # Create a new pickup stop
                new_pickup_stop = RouteStop(
                    stop=stop_assignment.origin_stop,
                    sequence=actual_pickup_idx,
                    pickup_passengers=[stop_assignment.request_id],
                    dropoff_passengers=[],
                    service_time=vehicle.config.boarding_time
                )
                stops.insert(actual_pickup_idx, new_pickup_stop)
            
            # After pickup is inserted, adjust the dropoff index if necessary
            if actual_dropoff_idx >= actual_pickup_idx and not existing_pickup_stop:
                # Pickup was inserted before dropoff, so increment dropoff index
                actual_dropoff_idx += 1
            
            # Find the updated pickup index after insertion
            pickup_index = -1
            for i, stop in enumerate(stops):
                if stop_assignment.request_id in stop.pickup_passengers:
                    pickup_index = i
                    break
            
            # Ensure dropoff comes after pickup - this is critical for route consistency
            if pickup_index != -1 and actual_dropoff_idx <= pickup_index:
                actual_dropoff_idx = pickup_index + 1
                logger.debug(f"Adjusted dropoff index to {actual_dropoff_idx} to maintain pickup-before-dropoff order")
            
            # Now process the dropoff stop
            if existing_dropoff_stop:
                if stop_assignment.request_id not in existing_dropoff_stop.dropoff_passengers:
                    existing_dropoff_stop.add_dropoff(stop_assignment.request_id)
                    existing_dropoff_stop.service_time = self._calculate_stop_service_time(
                        existing_dropoff_stop, 
                        vehicle.config
                    )
                # Ensure stop is at the right position
                if existing_dropoff_stop in stops:
                    # Remove from current position
                    stops.remove(existing_dropoff_stop)
                # Insert at the target position
                stops.insert(actual_dropoff_idx, existing_dropoff_stop)
            else:
                # Create a new dropoff stop
                new_dropoff_stop = RouteStop(
                    stop=stop_assignment.destination_stop,
                    sequence=actual_dropoff_idx,
                    pickup_passengers=[],
                    dropoff_passengers=[stop_assignment.request_id],
                    service_time=vehicle.config.alighting_time
                )
                stops.insert(actual_dropoff_idx, new_dropoff_stop)
            
            # Update stop sequences
            for i, stop in enumerate(stops):
                stop.sequence = i
            
            # Validate that we didn't mess up any completed stops
            is_valid = True
            for original_stop in original_stops:
                if original_stop.completed:
                    # Find the stop in the new list
                    found = False
                    for new_stop in stops:
                        if new_stop.id == original_stop.id:
                            found = True
                            # Ensure all completed stop properties are preserved
                            if not new_stop.completed:
                                logger.error(f"Completed stop {original_stop.id} is no longer marked as completed")
                                is_valid = False
                            break
                    
                    if not found:
                        logger.error(f"Completed stop {original_stop.id} was removed from the route")
                        is_valid = False
            
            if not is_valid:
                logger.warning("Route modification would invalidate completed stops - reverting")
                return None
                
            return stops
            
        except Exception as e:
            logger.error(f"Error processing stops for assignment: {traceback.format_exc()}")
            return None

    async def _update_stop_timings_and_travel(
        self,
        stops: List[RouteStop],
        vehicle: Vehicle
    ) -> None:
        """
        Updates arrival/departure times and travel information for all stops.
        
        Args:
            stops: List of stops to update
            vehicle: Vehicle operating the route
        """
        try:
            current_time = self.sim_context.current_time
            last_departure_time = current_time
            last_location = vehicle.current_state.current_location
            
            # First, preserve timing for all completed stops
            completed_stop_times = {}
            for stop in stops:
                if stop.completed:
                    completed_stop_times[stop.id] = {
                        'arrival': stop.actual_arrival_time or stop.planned_arrival_time,
                        'departure': stop.actual_departure_time or stop.planned_departure_time
                    }
            
            # Find the last completed stop to start timing from
            last_completed_stop = None
            for stop in stops:
                if stop.completed:
                    last_completed_stop = stop
                    last_location = stop.stop.location
                    if stop.actual_departure_time:
                        last_departure_time = stop.actual_departure_time
                    elif stop.planned_departure_time:
                        last_departure_time = stop.planned_departure_time
            
            # Now update timings for all uncompleted stops
            for i, stop in enumerate(stops):
                if stop.completed:
                    # Preserve timing for completed stops
                    continue
                    
                # Calculate travel time and distance from previous location
                travel_time = await self.network_manager.calculate_travel_time(
                    last_location,
                    stop.stop.location
                )
                
                travel_distance = await self.network_manager.calculate_distance(
                    last_location,
                    stop.stop.location
                )
                
                if travel_time == float('inf') or travel_distance == float('inf'):
                    logger.warning(f"Unable to calculate travel time/distance to stop {stop.stop.id}")
                    # Use a fallback estimate based on straight-line distance
                    distance = self._calculate_haversine_distance(
                        last_location.lat, last_location.lon,
                        stop.stop.location.lat, stop.stop.location.lon
                    )
                    # Assume average speed of 30 km/h (8.33 m/s)
                    travel_time = distance / 8.33
                    travel_distance = distance
                
                # Update stop travel metrics
                stop.origin_location = last_location
                stop.estimated_duration_to_stop = travel_time
                stop.estimated_distance_to_stop = travel_distance
                
                # Update arrival time based on previous departure
                stop.planned_arrival_time = last_departure_time + timedelta(seconds=travel_time)
                
                # Calculate service time based on passengers
                stop.service_time = self._calculate_stop_service_time(stop, vehicle.config)
                
                # Don't exceed maximum dwell time
                effective_service_time = min(stop.service_time, vehicle.config.max_dwell_time)
                
                # Update departure time considering service time
                stop.planned_departure_time = stop.planned_arrival_time + timedelta(seconds=effective_service_time)
                
                # Update tracking variables
                last_departure_time = stop.planned_departure_time
                last_location = stop.stop.location

        except Exception as e:
            logger.error(f"Error updating stop timings and travel info: {traceback.format_exc()}")
            raise

    def _update_occupancies(self, stops: List[RouteStop]) -> None:
        """
        Updates occupancy tracking for all stops.
        Enhanced to handle completed stops correctly.
        
        Args:
            stops: List of stops to update occupancies for
        """
        # Start with actual occupancy based on current passengers
        # Get on-board passengers from completed stops
        onboard = set()
        for stop in stops:
            if stop.completed:
                for request_id in stop.pickup_passengers:
                    if request_id in stop.boarded_request_ids:
                        onboard.add(request_id)
                        
                for request_id in stop.dropoff_passengers:
                    if request_id in onboard:
                        onboard.remove(request_id)
        
        current_occupancy = len(onboard)
        
        # Now update occupancies for all stops
        for stop in stops:
            if stop.completed:
                # For completed stops, just verify consistency
                expected_load = current_occupancy
                if stop.current_load != expected_load:
                    logger.warning(f"Occupancy inconsistency at completed stop {stop.id}: "
                                 f"current_load={stop.current_load}, expected={expected_load}")
                continue
                
            # For uncompleted stops, update the current load
            stop.current_load = current_occupancy
            
            # Calculate load changes at this stop
            pickup_count = len(stop.pickup_passengers)
            dropoff_count = len(stop.dropoff_passengers)
            
            # Compute occupancy after this stop for next stops
            current_occupancy += pickup_count - dropoff_count

    def _calculate_stop_service_time(
        self,
        stop: RouteStop,
        vehicle_config: VehicleConfig
    ) -> float:
        """
        Calculates service time based on passengers
        
        Args:
            stop: Stop to calculate service time for
            vehicle_config: Vehicle configuration with timing parameters
            
        Returns:
            Service time in seconds
        """
        # Calculate basic service time based on boarding/alighting operations
        boarding_time = len(stop.pickup_passengers) * vehicle_config.boarding_time
        alighting_time = len(stop.dropoff_passengers) * vehicle_config.alighting_time
        
        # Apply minimum service time
        return max(boarding_time + alighting_time, vehicle_config.min_dwell_time)

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
        Enhanced to handle route modifications better.
        
        Args:
            stops: List of route stops
            target_stop: Stop to find/insert
            preferred_idx: Preferred insertion index
            request_id: Request ID being processed
            is_pickup: Whether this is a pickup stop
            expected_passenger_origin_stop_arrival_time: Expected passenger arrival time
            
        Returns:
            Tuple of (compatible stop if found, actual insertion index)
        """
        # First ensure preferred_idx is within bounds and not inserting before a completed stop
        max_idx = len(stops)
        min_idx = 0
        
        # Find last completed stop index to avoid inserting before completed stops
        last_completed_idx = -1
        for i, stop in enumerate(stops):
            if stop.completed:
                last_completed_idx = i
                
        # Adjust minimum index to be after the last completed stop
        min_idx = max(min_idx, last_completed_idx + 1)
        
        # Clamp preferred index to valid range
        preferred_idx = max(min_idx, min(preferred_idx, max_idx))
        
        # Look for existing compatible stop
        compatible_stops = []
        for i, stop in enumerate(stops):
            if (
                stop.stop.id == target_stop.id and
                not stop.completed and
                i >= min_idx and
                self._is_stop_time_compatible(stop, is_pickup, expected_passenger_origin_stop_arrival_time)
            ):
                compatible_stops.append((stop, i))
                
        # If we found compatible stops, select the best one
        if compatible_stops:
            # Choose the one closest to the preferred index
            best_stop, best_idx = min(
                compatible_stops,
                key=lambda x: abs(x[1] - preferred_idx)
            )
            logger.debug(f"Found compatible {'pickup' if is_pickup else 'dropoff'} stop at index {best_idx}")
            return best_stop, best_idx
                
        logger.debug(f"No compatible {'pickup' if is_pickup else 'dropoff'} stop found, will insert at index {preferred_idx}")
        return None, preferred_idx

    def _create_fallback_path_info(self, origin: Location, destination: Location) -> Dict[str, Any]:
        """
        Create a fallback path info when actual path calculation fails
        
        Args:
            origin: Origin location
            destination: Destination location
            
        Returns:
            Dictionary with fallback path info
        """
        # Calculate straight-line distance
        distance = self._calculate_haversine_distance(
            origin.lat, origin.lon,
            destination.lat, destination.lon
        )
        
        # Estimate duration based on straight-line distance
        # Assume average speed of 30 km/h (8.33 m/s)
        estimated_speed = 8.33  # m/s
        duration = distance / estimated_speed
        
        # Create simplified waypoints
        waypoints = [
            {'location': origin},
            {'location': destination}
        ]
        
        return {
            'distance': distance,
            'duration': duration,
            'waypoints': waypoints,
            'path': []
        }
    
    def _calculate_haversine_distance(self, lat1, lon1, lat2, lon2):
        """Calculate the great circle distance between two points in meters"""
        import math
        
        R = 6371000  # Earth radius in meters
        
        # Convert latitude and longitude from degrees to radians
        lat1_rad = math.radians(lat1)
        lon1_rad = math.radians(lon1)
        lat2_rad = math.radians(lat2)
        lon2_rad = math.radians(lon2)
        
        # Haversine formula
        dlon = lon2_rad - lon1_rad
        dlat = lat2_rad - lat1_rad
        a = math.sin(dlat/2)**2 + math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(dlon/2)**2
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
        distance = R * c
        
        return distance

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
            is_valid, error_msg = route.validate_capacity(vehicle.capacity)
            if not is_valid:
                logger.debug(f"Capacity validation failed: {error_msg}")
                return False
                
            # Get relevant stops
            pickup_stop = route.find_stop_for_request(
                stop_assignment.request_id,
                is_pickup=True
            )
            dropoff_stop = route.find_stop_for_request(
                stop_assignment.request_id,
                is_pickup=False
            )
            
            if not pickup_stop or not dropoff_stop:
                logger.debug("Could not find pickup or dropoff stop for request")
                return False
            
            # Validate route integrity
            is_valid, error_msg = route.validate_route_integrity()
            if not is_valid:
                logger.debug(f"Route integrity validation failed: {error_msg}")
                return False
                
            logger.debug("Route constraints validation passed")
            return True
            
        except Exception as e:
            logger.error(f"Error validating route constraints: {traceback.format_exc()}")
            return False
        
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