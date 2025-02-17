from datetime import datetime, timedelta
from copy import deepcopy
from typing import Dict, Any, Optional, List, Tuple
import traceback
from drt_sim.models.route import Route, RouteStop, RouteStatus, RouteSegment
from drt_sim.models.vehicle import Vehicle
from drt_sim.models.stop import Stop, StopAssignment
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
            # Calculate initial travel time to pickup
            pickup_travel_time = await self.network_manager.calculate_travel_time(
                vehicle.current_state.current_location,
                stop_assignment.origin_stop.location
            )
            
            if pickup_travel_time == float('inf'):
                return None

            # Calculate arrival times
            pickup_arrival_time = self.sim_context.current_time + timedelta(
                seconds=pickup_travel_time
            )
            
            travel_time_to_dropoff = await self.network_manager.calculate_travel_time(
                stop_assignment.origin_stop.location,
                stop_assignment.destination_stop.location
            )

            if travel_time_to_dropoff == float('inf'):
                return None
            
            dropoff_arrival_time = pickup_arrival_time + timedelta(
                seconds=vehicle.config.boarding_time + travel_time_to_dropoff
            )

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

            # Create segments list to include vehicle's current position
            segments: List[RouteSegment] = []
            
            # Add initial positioning segment (from vehicle's current location to pickup)
            initial_segment = RouteSegment(
                origin=None,  # No RouteStop for vehicle's current position
                destination=pickup_stop,
                estimated_duration=pickup_travel_time,
                estimated_distance=await self.network_manager.calculate_distance(
                    vehicle.current_state.current_location,
                    stop_assignment.origin_stop.location
                ),
                origin_location=vehicle.current_state.current_location,  # Add current location
                destination_location=stop_assignment.origin_stop.location
            )
            segments.append(initial_segment)

            # Add segment between pickup and dropoff
            service_segment = RouteSegment(
                origin=pickup_stop,
                destination=dropoff_stop,
                estimated_duration=travel_time_to_dropoff,
                estimated_distance=await self.network_manager.calculate_distance(
                    stop_assignment.origin_stop.location,
                    stop_assignment.destination_stop.location
                ),
                origin_location=stop_assignment.origin_stop.location,
                destination_location=stop_assignment.destination_stop.location
            )
            segments.append(service_segment)

            return Route(
                vehicle_id=vehicle.id,
                stops=[pickup_stop, dropoff_stop],
                segments=segments,
                status=RouteStatus.CREATED,
                total_distance=sum(segment.estimated_distance for segment in segments),
                total_duration=sum(segment.estimated_duration for segment in segments) + 
                              vehicle.config.boarding_time + 
                              vehicle.config.alighting_time,
                current_segment_index=0
            )
            
        except Exception as e:
            logger.error(f"Error creating new route: {traceback.format_exc()}")
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
        
        Args:
            current_route: Current route to modify
            stop_assignment: Stop assignment to insert
            pickup_idx: Index to insert pickup
            dropoff_idx: Index to insert dropoff
            vehicle: Vehicle operating the route
            
        Returns:
            Modified route if successful, None otherwise
        """
        try:
            new_route = deepcopy(current_route)
            new_stops = new_route.stops
            
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
            
            # Process stops
            new_stops = await self._process_pickup_stop(
                new_stops,
                existing_pickup_stop,
                actual_pickup_idx,
                stop_assignment,
                vehicle
            )
            
            new_stops = await self._process_dropoff_stop(
                new_stops,
                existing_dropoff_stop,
                actual_dropoff_idx,
                stop_assignment,
                vehicle
            )
            
            # Update route properties
            await self._update_stop_timings(new_stops, vehicle)
            self._update_occupancies(new_stops)
            
            # Recreate segments for the modified route
            new_segments = []
            
            # Add initial segment from vehicle's current position if needed
            if not new_stops[0].completed:
                initial_segment = RouteSegment(
                    origin=None,
                    destination=new_stops[0],
                    estimated_duration=await self.network_manager.calculate_travel_time(
                        vehicle.current_state.current_location,
                        new_stops[0].stop.location
                    ),
                    estimated_distance=await self.network_manager.calculate_distance(
                        vehicle.current_state.current_location,
                        new_stops[0].stop.location
                    ),
                    origin_location=vehicle.current_state.current_location,
                    destination_location=new_stops[0].stop.location
                )
                new_segments.append(initial_segment)
            
            # Add segments between stops
            for i in range(len(new_stops) - 1):
                origin_stop = new_stops[i]
                destination_stop = new_stops[i + 1]
                
                segment = RouteSegment(
                    origin=origin_stop,
                    destination=destination_stop,
                    estimated_duration=await self.network_manager.calculate_travel_time(
                        origin_stop.stop.location,
                        destination_stop.stop.location
                    ),
                    estimated_distance=await self.network_manager.calculate_distance(
                        origin_stop.stop.location,
                        destination_stop.stop.location
                    ),
                    origin_location=origin_stop.stop.location,
                    destination_location=destination_stop.stop.location,
                    completed=origin_stop.completed and destination_stop.completed
                )
                new_segments.append(segment)
            
            new_route.stops = new_stops
            new_route.segments = new_segments
            new_route.total_distance = sum(segment.estimated_distance for segment in new_segments)
            new_route.total_duration = sum(segment.estimated_duration for segment in new_segments) + \
                                     sum(stop.service_time for stop in new_stops)
            
            return new_route
            
        except Exception as e:
            logger.warning(f"Could not modify route: {traceback.format_exc()}")
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
        """Finds existing compatible stop or determines insertion point"""
        # Look for existing compatible stop
        for i, stop in enumerate(stops):
            if (
                stop.stop.id == target_stop.id and
                not stop.completed and
                self._is_stop_time_compatible(stop, is_pickup, expected_passenger_origin_stop_arrival_time)
            ):
                return stop, i
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
            if stop_assignment.request_id not in existing_stop.pickup_passengers:
                existing_stop.pickup_passengers.append(stop_assignment.request_id)
                existing_stop.service_time = self._calculate_stop_service_time(
                    existing_stop, 
                    vehicle.config
                )
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
        if existing_stop:
            if stop_assignment.request_id not in existing_stop.dropoff_passengers:
                existing_stop.dropoff_passengers.append(stop_assignment.request_id)
                existing_stop.service_time = self._calculate_stop_service_time(
                    existing_stop, 
                    vehicle.config
                )
            stops.insert(index, existing_stop)
        else:
            new_stop = RouteStop(
                stop=stop_assignment.destination_stop,
                sequence=index,
                pickup_passengers=[],
                dropoff_passengers=[stop_assignment.request_id],
                service_time=vehicle.config.alighting_time,
                current_load=0  # Will be updated by _update_occupancies
            )
            stops.insert(index, new_stop)
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
            passenger_arrival = (
                stop_assignment.assignment_time +
                timedelta(seconds=stop_assignment.walking_time_origin)
            )
            
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
            time_window: Time window for combining stops in seconds
            
        Returns:
            True if stop can be combined, False otherwise
        """
        if stop.completed:
            return False
            
        if not stop.planned_arrival_time:
            return False
            
        current_time = self.sim_context.current_time
        
        if is_pickup:
            # For pickups, ensure that the passenger arrives before the vehicle's planned arrival time + its max dwell time
            return expected_passenger_origin_stop_arrival_time < stop.planned_arrival_time + self.config.vehicle.max_dwell_time
        else:
            # For dropoffs, be more lenient with timing
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