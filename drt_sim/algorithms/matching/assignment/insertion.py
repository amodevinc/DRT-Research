from typing import Dict, List, Optional
from datetime import datetime, timedelta
import asyncio
from dataclasses import dataclass
import traceback
import math
from drt_sim.models.vehicle import Vehicle, VehicleStatus
from drt_sim.models.route import Route, RouteSegment
from drt_sim.models.matching import Assignment
from drt_sim.models.stop import StopAssignment
from drt_sim.network.manager import NetworkManager
from drt_sim.models.location import Location
from drt_sim.core.state.manager import StateManager
from drt_sim.core.simulation.context import SimulationContext
from drt_sim.core.user.manager import UserProfileManager
from drt_sim.core.services.route_service import RouteService
from drt_sim.config.config import MatchingAssignmentConfig
import logging
logger = logging.getLogger(__name__)

@dataclass
class InsertionCost:
    """Represents the cost of inserting a request into a route"""
    costs: Dict[str, float]  # Individual cost components
    total_cost: float
    feasible: bool
    pickup_index: int
    dropoff_index: int
    updated_route: Optional[Route] = None
    vehicle: Optional[Vehicle] = None

class InsertionAssigner:
    """Assigns requests to vehicles using insertion heuristic"""
    
    def __init__(
        self,
        sim_context: SimulationContext,
        config: MatchingAssignmentConfig,
        network_manager: NetworkManager,
        state_manager: StateManager,
        user_profile_manager: UserProfileManager,
        route_service: RouteService
    ):
        self.sim_context = sim_context
        self.config = config
        self.network_manager = network_manager
        self.state_manager = state_manager
        self.user_profile_manager = user_profile_manager
        self.route_service = route_service

    async def assign_request(
        self,
        stop_assignment: StopAssignment,
        available_vehicles: List[Vehicle],
    ) -> Optional[Assignment]:
        """
        Assigns request to best available vehicle using insertion heuristic
        
        Args:
            stop_assignment: Stop assignment to be inserted
            available_vehicles: List of available vehicles
            
        Returns:
            Assignment if feasible match found, None otherwise
        """
        try:
            computation_start = datetime.now()
            
            # Filter compatible vehicles
            compatible_vehicles = [
                vehicle for vehicle in available_vehicles
                if self._is_vehicle_compatible(vehicle)
            ]
            
            if not compatible_vehicles:
                logger.debug(f"No compatible vehicles for request {stop_assignment.request_id}")
                return None
            
            # Create tasks for parallel evaluation
            insertion_tasks = [
                self._evaluate_vehicle_insertion(stop_assignment, vehicle)
                for vehicle in compatible_vehicles
            ]
            
            # Evaluate all vehicles in parallel
            insertion_results = await asyncio.gather(*insertion_tasks)
            
            # Find best feasible insertion
            best_insertion = min(
                (result for result in insertion_results if result and result.feasible),
                key=lambda x: x.total_cost,
                default=None
            )
            
            if best_insertion:
                return await self._create_assignment(
                    stop_assignment,
                    best_insertion,
                    computation_start
                )
            
            return None
            
        except Exception as e:
            logger.error(f"Error in assign_request: {str(e)}", exc_info=True)
            return None

    async def _evaluate_vehicle_insertion(
        self,
        stop_assignment: StopAssignment,
        vehicle: Vehicle,
    ) -> Optional[InsertionCost]:
        try:
            current_route = self.state_manager.route_worker.get_route(vehicle.get_active_route_id())
            if not current_route:
                return await self._evaluate_new_route(stop_assignment, vehicle)

            first_valid_index = self._get_first_valid_index(current_route)
            max_stops = len(current_route.stops)
            best_insertion = None
            min_cost = float('inf')

            for pickup_idx in range(first_valid_index, max_stops + 1):
                for dropoff_idx in range(pickup_idx + 1, max_stops + 2):
                    new_route = await self.route_service.create_modified_route(
                        current_route=current_route,
                        stop_assignment=stop_assignment,
                        pickup_idx=pickup_idx,
                        dropoff_idx=dropoff_idx,
                        vehicle=vehicle
                    )

                    if not new_route:
                        continue

                    if not self.route_service.validate_route_constraints(
                        new_route,
                        vehicle,
                        stop_assignment,
                        self.config
                    ):
                        continue

                    costs = self.route_service.calculate_route_metrics(
                        current_route,
                        new_route,
                        stop_assignment
                    )
                    
                    total_cost = self._calculate_weighted_cost(costs, stop_assignment.request_id)
                    
                    if total_cost < min_cost:
                        min_cost = total_cost
                        best_insertion = InsertionCost(
                            costs=costs,
                            total_cost=total_cost,
                            feasible=True,
                            pickup_index=pickup_idx,
                            dropoff_index=dropoff_idx,
                            updated_route=new_route,
                            vehicle=vehicle
                        )
                        
            return best_insertion

        except Exception as e:
            logger.error(f"Error evaluating vehicle insertion: {str(e)}")
            return None

    async def _evaluate_new_route(
        self,
        stop_assignment: StopAssignment,
        vehicle: Vehicle
    ) -> Optional[InsertionCost]:
        """Evaluates cost of creating new route for empty vehicle"""
        try:
            new_route = await self.route_service.create_new_route(
                vehicle=vehicle,
                stop_assignment=stop_assignment
            )
            
            if not new_route:
                return None

            # Calculate costs using route data
            costs = {
                "waiting_time": (
                    min(0, (new_route.stops[0].planned_arrival_time - stop_assignment.expected_passenger_origin_stop_arrival_time).total_seconds()) / 60
                ),
                "ride_time": new_route.total_duration / 60, # convert to minutes
                "delay": 0.0,  # No existing passengers
                "distance": new_route.total_distance,
            }
            
            total_cost = self._calculate_weighted_cost(
                costs,
                stop_assignment.request_id
            )

            return InsertionCost(
                costs=costs,
                total_cost=total_cost,
                feasible=True,
                pickup_index=0,
                dropoff_index=1,
                updated_route=new_route,
                vehicle=vehicle
            )
        except Exception as e:
            logger.error(f"Error evaluating new route: {str(e)}")
            return None

    def _calculate_weighted_cost(
        self,
        costs: Dict[str, float],
        request_id: str
    ) -> float:
        """Calculates total weighted cost using user preferences"""
        try:
            # Get user-specific weights
            weights = self.user_profile_manager.get_adjusted_weights(
                request_id,
                self.config.weights
            )
            
            # Normalize costs
            normalized_costs = {
                "waiting_time": min(1.0, costs["waiting_time"] / self.config.max_waiting_time_mins),
                "ride_time": min(1.0, costs["ride_time"] / self.config.max_in_vehicle_time_mins),
                "delay": min(1.0, costs["delay"] / self.config.max_delay_mins),
                "distance": min(1.0, costs["distance"] / self.config.max_distance),
            }
            
            # Calculate weighted sum
            total_cost = sum(
                weights.get(factor, self.config.default_weight) * value
                for factor, value in normalized_costs.items()
            )
            
            return total_cost
            
        except Exception as e:
            logger.error(f"Error calculating weighted cost: {traceback.format_exc()}")
            return float('inf')

    async def ensure_segments(self, route: Route) -> None:
        """Ensures route segments have travel times and distances calculated"""
        if not route.segments:
            segment_tasks = []
            for i in range(len(route.stops) - 1):
                # Create tasks for both time and distance calculations
                time_task = self.network_manager.calculate_travel_time(
                    route.stops[i].location,
                    route.stops[i + 1].location
                )
                distance_task = self.network_manager.calculate_distance(
                    route.stops[i].location,
                    route.stops[i + 1].location
                )
                segment_tasks.append(asyncio.gather(time_task, distance_task))
            
            # Calculate all segments in parallel
            segment_results = await asyncio.gather(*segment_tasks)
            
            # Create RouteSegment objects
            route.segments = [
                RouteSegment(
                    origin=route.stops[i],
                    destination=route.stops[i + 1],
                    estimated_duration=segment_results[i][0],  # travel time in seconds
                    estimated_distance=segment_results[i][1],  # distance in meters
                    completed=False
                )
                for i in range(len(route.stops) - 1)
            ]

    async def _create_assignment(
        self,
        stop_assignment: StopAssignment,
        insertion: InsertionCost,
        computation_start: datetime
    ) -> Optional[Assignment]:
        """Creates final assignment from best insertion"""
        try:
            # Ensure route segments are calculated
            await self.ensure_segments(insertion.updated_route)

            
            route_pickup_stop = next(
                stop for stop in insertion.updated_route.stops
                if stop_assignment.request_id in stop.pickup_passengers
            )
            route_dropoff_stop = next(
                stop for stop in insertion.updated_route.stops
                if stop_assignment.request_id in stop.dropoff_passengers
            )
            
            computation_time = (datetime.now() - computation_start).total_seconds()
            
            return Assignment(
                request_id=stop_assignment.request_id,
                vehicle_id=insertion.vehicle.id,
                route=insertion.updated_route,
                stop_assignment_id=stop_assignment.id,
                assignment_time=self.sim_context.current_time,
                estimated_pickup_time=route_pickup_stop.planned_arrival_time,
                estimated_dropoff_time=route_dropoff_stop.planned_arrival_time,
                waiting_time_mins=insertion.costs["waiting_time"],
                in_vehicle_time_mins=insertion.costs["ride_time"],
                detour_time_mins=max(
                    0,
                    insertion.costs["ride_time"] - 
                    await self._calculate_direct_travel_time(
                        stop_assignment.origin_stop.location,
                        stop_assignment.destination_stop.location
                    )
                ),
                assignment_score=1.0 - insertion.total_cost,
                computation_time=computation_time,
                metadata={
                    "insertion_costs": insertion.costs,
                    "pickup_index": insertion.pickup_index,
                    "dropoff_index": insertion.dropoff_index,
                    "route_id": insertion.updated_route.id
                }
            )
            
        except Exception as e:
            logger.error(f"Error creating assignment: {str(e)}\n{traceback.format_exc()}")
            return None

    async def _calculate_direct_travel_time(
        self,
        origin_location: Location,
        destination_location: Location
    ) -> float:
        """Calculates direct travel time between locations in minutes"""
        travel_time = await self.network_manager.calculate_travel_time(
            origin_location,
            destination_location
        )
        return travel_time / 60

    def _get_first_valid_index(self, route: Route) -> int:
        """Gets index of first non-completed stop"""
        for i, stop in enumerate(route.stops):
            if not stop.completed:
                return i
        return len(route.stops)

    def _is_vehicle_compatible(self, vehicle: Vehicle) -> bool:
        """Checks if vehicle is available for assignment"""
        return vehicle.current_state.status not in [
            VehicleStatus.OFF_DUTY,
            VehicleStatus.INACTIVE
        ]
