from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta
import asyncio
from dataclasses import dataclass
import traceback
import math
from drt_sim.models.vehicle import Vehicle, VehicleStatus
from drt_sim.models.route import Route
from drt_sim.models.matching import Assignment
from drt_sim.models.stop import StopAssignment
from drt_sim.network.manager import NetworkManager
from drt_sim.models.location import Location
from drt_sim.core.state.manager import StateManager
from drt_sim.core.simulation.context import SimulationContext
from drt_sim.core.user.user_profile_manager import UserProfileManager
from drt_sim.core.services.route_service import RouteService
from drt_sim.config.config import MatchingAssignmentConfig
from drt_sim.models.rejection import RejectionReason, RejectionMetadata
import logging
logger = logging.getLogger(__name__)

@dataclass
class InsertionCost:
    """Represents the cost of inserting a request into a route"""
    cost_components: Dict[str, float]  # Individual cost components
    total_cost: float
    feasible: bool
    pickup_index: int
    dropoff_index: int
    updated_route: Optional[Route] = None
    vehicle: Optional[Vehicle] = None
    rejection_reason: Optional[RejectionReason] = None
    rejection_details: Optional[Dict[str, Any]] = None

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
        
        # Configuration parameters from constraints
        self.constraints = config.constraints
        self.max_waiting_time = config.constraints["max_waiting_time_secs"]
        self.max_in_vehicle_time = config.constraints["max_in_vehicle_time_secs"]
        self.max_vehicle_access_time = config.constraints["max_vehicle_access_time_secs"]
        self.max_existing_passenger_delay = config.constraints["max_existing_passenger_delay_secs"]
        self.max_distance = config.constraints["max_distance_meters"]
        
        # Configuration parameters from weights
        self.weights = config.weights
        
        # Initialize violation tracking
        self.violation_counts = {
            "time_window": 0,
            "capacity": 0,
            "ride_time": 0,
            "vehicle_access_time": 0,
            "passenger_wait_time": 0,
            "distance": 0,
            "total_duration": 0,
            "passenger_comfort": 0,
            "vehicle_efficiency": 0,
            "operational_cost": 0
        }

    async def assign_request(
        self,
        stop_assignment: StopAssignment,
    ) -> Tuple[Optional[Assignment], Optional[RejectionMetadata]]:
        """
        Assigns request to best available vehicle using insertion heuristic
        
        Args:
            stop_assignment: Stop assignment to be inserted
            
        Returns:
            Tuple of (Assignment if feasible match found, RejectionMetadata if rejected)
        """
        try:
            computation_start = datetime.now()
            available_vehicles = self.state_manager.vehicle_worker.get_available_vehicles()
            logger.info(f"Starting assignment for request {stop_assignment.request_id} with {len(available_vehicles)} available vehicles")
            logger.debug(f"Stop assignment details: pickup={stop_assignment.origin_stop.location}, dropoff={stop_assignment.destination_stop.location}, "
                      f"expected_arrival={stop_assignment.expected_passenger_origin_stop_arrival_time}")
            
            logger.info(f"Found {len(available_vehicles)} compatible vehicles out of {len(available_vehicles)} available")
            for vehicle in available_vehicles:
                logger.debug(f"Compatible vehicle {vehicle.id}: status={vehicle.current_state.status}, "
                          f"location={vehicle.current_state.current_location}, occupancy={vehicle.current_state.current_occupancy}")
            
            if not available_vehicles:
                return None, RejectionMetadata(
                    reason=RejectionReason.NO_VEHICLES_AVAILABLE,
                    timestamp=self.sim_context.current_time.isoformat(),
                    stage="matching",
                    details={
                        "total_vehicles": len(available_vehicles),
                        "compatible_vehicles": 0
                    }
                )
            
            # Create tasks for parallel evaluation
            insertion_tasks = [
                self._evaluate_vehicle_insertion(stop_assignment, vehicle)
                for vehicle in available_vehicles
            ]
            
            # Evaluate all vehicles in parallel
            insertion_results = await asyncio.gather(*insertion_tasks)
            
            # Find best feasible insertion
            best_insertion = min(
                (result for result in insertion_results if result and result.feasible),
                key=lambda x: x.total_cost,
                default=None
            )

            if not best_insertion:
                # If no feasible insertion found, gather rejection reasons
                rejection_reasons = [
                    result.rejection_reason for result in insertion_results 
                    if result and result.rejection_reason
                ]

                logger.info(f"Rejection reasons: {rejection_reasons}")
                
                # Get the most common rejection reason
                most_common_reason = max(
                    set(rejection_reasons),
                    key=rejection_reasons.count,
                    default=RejectionReason.NO_FEASIBLE_INSERTION
                )
                
                # Aggregate rejection details
                aggregated_details = {
                    "evaluated_vehicles": len(available_vehicles),
                    "rejection_counts": {
                        reason.value: rejection_reasons.count(reason)
                        for reason in set(rejection_reasons)
                    },
                    "constraint_violations": {}
                }
                
                # Aggregate constraint violation details from all results
                for result in insertion_results:
                    if result and result.rejection_details:
                        for constraint, value in result.rejection_details.get("violations", {}).items():
                            if constraint not in aggregated_details["constraint_violations"]:
                                aggregated_details["constraint_violations"][constraint] = []
                            aggregated_details["constraint_violations"][constraint].append(value)
                
                return None, RejectionMetadata(
                    reason=most_common_reason,
                    timestamp=self.sim_context.current_time.isoformat(),
                    stage="matching",
                    details=aggregated_details
                )
            
            # Get best vehicle's current route for assignment creation
            best_vehicle_current_route_id = self.state_manager.vehicle_worker.get_vehicle_active_route_id(best_insertion.vehicle.id)
            best_vehicle_current_route = self.state_manager.route_worker.get_route(best_vehicle_current_route_id)
            
            # Create assignment
            assignment = await self._create_assignment(
                best_vehicle_current_route=best_vehicle_current_route,
                stop_assignment=stop_assignment,
                insertion=best_insertion,
                computation_start=computation_start
            )
            return assignment, None
            
        except Exception as e:
            logger.error(f"Error in assign_request: {str(e)}", exc_info=True)
            return None, RejectionMetadata(
                reason=RejectionReason.TECHNICAL_ERROR,
                timestamp=self.sim_context.current_time.isoformat(),
                stage="matching",
                details={"error": str(e)}
            )

    async def _evaluate_vehicle_insertion(
        self,
        stop_assignment: StopAssignment,
        vehicle: Vehicle,
    ) -> Optional[InsertionCost]:
        """Evaluate inserting a stop assignment into a vehicle's route"""
        try:
            logger.info(f"Evaluating insertion for vehicle {vehicle.id} with stop assignment {stop_assignment.id}")
            logger.debug(f"Vehicle state: location={vehicle.current_state.current_location}, "
                      f"occupancy={vehicle.current_state.current_occupancy}/{vehicle.capacity}")
            
            # If vehicle is rebalancing, we can consider a new route from its current location
            if vehicle.current_state.status == VehicleStatus.REBALANCING:
                return await self._evaluate_new_route(stop_assignment, vehicle)
            
            # If vehicle is not rebalancing, we can evaluate the current route
            active_route_id = self.state_manager.vehicle_worker.get_vehicle_active_route_id(vehicle.id)
            current_route = self.state_manager.route_worker.get_route(active_route_id)
            
            if current_route:
                logger.debug(f"Current route {current_route.id}: stops={len(current_route.stops)}, status={current_route.status}")
                logger.debug(f"Route stops: {[str(stop) for stop in current_route.stops]}")
            
            # Handle both cases - vehicle with route and without route
            if not current_route:
                logger.info(f"Vehicle {vehicle.id} has no current route, evaluating new route creation")
                return await self._evaluate_new_route(stop_assignment, vehicle)
            
            logger.info(f"Vehicle {vehicle.id} has existing route, evaluating insertion into current route")
            
            first_valid_index = self._get_first_valid_index(current_route, vehicle)
            max_stops = len(current_route.stops)
            best_insertion = None
            min_cost = float('inf')
            
            logger.info(f"Evaluating insertions for vehicle {vehicle.id}: first_valid_index={first_valid_index}, max_stops={max_stops}")
            
            # Track all constraint violations across attempts
            all_constraint_violations = {}
            route_creation_failed = False

            # Only evaluate insertions if we have valid positions to insert
            if first_valid_index >= max_stops:
                logger.info(f"No valid insertion positions for vehicle {vehicle.id} - all stops completed")
                return InsertionCost(
                    cost_components={},
                    total_cost=float('inf'),
                    feasible=False,
                    pickup_index=-1,
                    dropoff_index=-1,
                    rejection_reason=RejectionReason.NO_FEASIBLE_INSERTION,
                    rejection_details={
                        "vehicle_id": vehicle.id,
                        "reason": "All stops in current route completed"
                    }
                )

            for pickup_idx in range(first_valid_index, max_stops + 1):
                for dropoff_idx in range(pickup_idx + 1, max_stops + 2):
                    logger.debug(f"Attempting insertion at pickup_idx={pickup_idx}, dropoff_idx={dropoff_idx}")
                    
                    new_route = await self.route_service.create_modified_route(
                        current_route=current_route,
                        stop_assignment=stop_assignment,
                        pickup_idx=pickup_idx,
                        dropoff_idx=dropoff_idx,
                        vehicle=vehicle
                    )

                    if not new_route:
                        route_creation_failed = True
                        continue

                    # Check stop time compatibility for pickup
                    pickup_stop = next(
                        (stop for stop in new_route.stops if stop_assignment.request_id in stop.pickup_passengers),
                        None
                    )
                    if not pickup_stop or not self._is_stop_time_compatible(
                        pickup_stop.planned_arrival_time,
                        stop_assignment.expected_passenger_origin_stop_arrival_time
                    ):
                        logger.debug(f"Stop timing incompatible for vehicle {vehicle.id} at pickup_idx={pickup_idx}: " +
                                  f"planned_arrival={pickup_stop.planned_arrival_time if pickup_stop else 'None'}, " +
                                  f"passenger_arrival={stop_assignment.expected_passenger_origin_stop_arrival_time}")
                        continue

                    logger.debug(f"Successfully created modified route for vehicle {vehicle.id}, checking constraints")

                    # Check all constraints
                    violations = self._check_constraints(
                        new_route=new_route,
                        stop_assignment=stop_assignment,
                        pickup_idx=pickup_idx,
                        vehicle=vehicle,
                        current_route=current_route
                    )

                    if violations:
                        logger.debug(f"Constraint violations found for vehicle {vehicle.id}: {violations}")
                        # Accumulate violations for rejection metadata
                        for constraint, value in violations.items():
                            if constraint not in all_constraint_violations:
                                all_constraint_violations[constraint] = []
                            all_constraint_violations[constraint].append(value)
                        continue

                    logger.debug(f"No constraint violations found, calculating cost components")

                    # Calculate weighted cost components
                    cost_components = self._calculate_cost_components(
                        new_route=new_route,
                        current_route=current_route,
                        stop_assignment=stop_assignment,
                        pickup_idx=pickup_idx,
                        vehicle=vehicle
                    )
                    
                    # Calculate total weighted cost
                    total_cost = self._calculate_weighted_cost(cost_components)
                    
                    logger.debug(f"Calculated total cost: {total_cost} for vehicle {vehicle.id}")
                    
                    if total_cost < min_cost:
                        logger.info(f"Found better insertion for vehicle {vehicle.id} with cost {total_cost} (previous best: {min_cost})")
                        min_cost = total_cost
                        best_insertion = InsertionCost(
                            cost_components=cost_components,
                            total_cost=total_cost,
                            feasible=True,
                            pickup_index=pickup_idx,
                            dropoff_index=dropoff_idx,
                            updated_route=new_route,
                            vehicle=vehicle
                        )
            
            if not best_insertion:
                # If we have constraint violations, use those as the reason
                if all_constraint_violations:
                    # Determine primary violation reason based on most frequent violation
                    primary_violation = max(
                        all_constraint_violations.items(),
                        key=lambda x: len(x[1])
                    )[0]
                    
                    logger.info(f"No feasible insertion found for vehicle {vehicle.id}. Primary violation: {primary_violation}")
                    logger.debug(f"All constraint violations: {all_constraint_violations}")
                    
                    return InsertionCost(
                        cost_components={},
                        total_cost=float('inf'),
                        feasible=False,
                        pickup_index=-1,
                        dropoff_index=-1,
                        rejection_reason=self._map_violation_to_reason(primary_violation),
                        rejection_details={
                            "vehicle_id": vehicle.id,
                            "violations": all_constraint_violations
                        }
                    )
                # If route creation failed, set that as the reason
                elif route_creation_failed:
                    logger.info(f"No feasible insertion found for vehicle {vehicle.id} due to route creation failures")
                    return InsertionCost(
                        cost_components={},
                        total_cost=float('inf'),
                        feasible=False,
                        pickup_index=-1,
                        dropoff_index=-1,
                        rejection_reason=RejectionReason.NO_FEASIBLE_INSERTION,
                        rejection_details={
                            "vehicle_id": vehicle.id,
                            "error": "Failed to create modified route"
                        }
                    )
                # If no specific reason found, use generic no feasible insertion
                else:
                    logger.info(f"No feasible insertion found for vehicle {vehicle.id} with no specific violation reason")
                    return InsertionCost(
                        cost_components={},
                        total_cost=float('inf'),
                        feasible=False,
                        pickup_index=-1,
                        dropoff_index=-1,
                        rejection_reason=RejectionReason.NO_FEASIBLE_INSERTION,
                        rejection_details={
                            "vehicle_id": vehicle.id
                        }
                    )
            
            logger.info(f"Successfully found feasible insertion for vehicle {vehicle.id} with final cost {min_cost}")
            return best_insertion

        except Exception as e:
            logger.error(f"Error evaluating vehicle insertion for vehicle {vehicle.id}: {str(e)}", exc_info=True)
            return InsertionCost(
                cost_components={},
                total_cost=float('inf'),
                feasible=False,
                pickup_index=-1,
                dropoff_index=-1,
                rejection_reason=RejectionReason.TECHNICAL_ERROR,
                rejection_details={"error": str(e)}
            )

    def _check_constraints(
        self,
        new_route: Route,
        stop_assignment: StopAssignment,
        pickup_idx: int,
        vehicle: Vehicle,
        current_route: Optional[Route] = None
    ) -> Dict[str, Any]:
        """Check all constraints and return violations if any."""
        violations = {}
        constraints = self.config.constraints
        
        logger.debug(f"Checking constraints for vehicle {vehicle.id} at pickup_idx={pickup_idx}")
        logger.debug(f"New route details: stops={len(new_route.stops)}, duration={new_route.total_duration}, "
                  f"distance={new_route.total_distance}")

        # Validate pickup index
        if not new_route.stops or pickup_idx >= len(new_route.stops):
            logger.error(f"Invalid pickup index {pickup_idx} for route with {len(new_route.stops)} stops")
            return {
                "invalid_index": {
                    "actual": pickup_idx,
                    "max_valid": len(new_route.stops) - 1 if new_route.stops else -1,
                    "reason": "Pickup index out of range"
                }
            }

        # Add detailed constraint checking logs
        try:
            vehicle_access_time = (
                new_route.stops[pickup_idx].planned_arrival_time - 
                self.sim_context.current_time
            ).total_seconds() / 60
            logger.debug(f"Vehicle access time: {vehicle_access_time:.2f} min (limit: {constraints['max_vehicle_access_time_secs'] / 60} mins)")
            if vehicle_access_time > constraints["max_vehicle_access_time_secs"]:
                violations["vehicle_access_time"] = {
                    "actual": vehicle_access_time,
                    "limit": constraints["max_vehicle_access_time_secs"]
                }
        except Exception as e:
            logger.error(f"Error calculating vehicle access time: {str(e)}")
            violations["vehicle_access_time"] = {
                "error": str(e),
                "reason": "Failed to calculate vehicle access time"
            }

        # 2. Check passenger waiting time
        try:
            passenger_wait_time = (
                new_route.stops[pickup_idx].planned_arrival_time - 
                stop_assignment.expected_passenger_origin_stop_arrival_time
            ).total_seconds() / 60
            logger.debug(f"Passenger wait time: {passenger_wait_time:.2f} min (limit: {constraints['max_waiting_time_secs'] / 60} mins)")
            if passenger_wait_time > constraints["max_waiting_time_secs"]:
                violations["waiting_time"] = {
                    "actual": passenger_wait_time,
                    "limit": constraints["max_waiting_time_secs"]
                }
        except Exception as e:
            logger.error(f"Error calculating passenger wait time: {str(e)}")
            violations["waiting_time"] = {
                "error": str(e),
                "reason": "Failed to calculate passenger wait time"
            }

        # 3. Check in-vehicle time for new passenger
        try:
            # Find the dropoff stop index
            dropoff_idx = -1
            for i, stop in enumerate(new_route.stops):
                if stop_assignment.request_id in stop.dropoff_passengers:
                    dropoff_idx = i
                    break
                    
            if dropoff_idx == -1 or dropoff_idx <= pickup_idx:
                # If we can't find the dropoff or it's before pickup, this is invalid
                violations["ride_time"] = {
                    "actual": float('inf'),
                    "limit": constraints["max_in_vehicle_time_secs"],
                    "reason": "No valid dropoff found after pickup"
                }
            else:
                in_vehicle_time = (
                    new_route.stops[dropoff_idx].planned_arrival_time - 
                    new_route.stops[pickup_idx].planned_arrival_time
                ).total_seconds() / 60
                logger.debug(f"In-vehicle time: {in_vehicle_time:.2f} min (limit: {constraints['max_in_vehicle_time_secs'] / 60} mins)")
                if in_vehicle_time > constraints["max_in_vehicle_time_secs"]:
                    violations["ride_time"] = {
                        "actual": in_vehicle_time,
                        "limit": constraints["max_in_vehicle_time_secs"]
                    }
        except Exception as e:
            logger.error(f"Error calculating in-vehicle time: {str(e)}")
            violations["ride_time"] = {
                "error": str(e),
                "reason": "Failed to calculate in-vehicle time"
            }

        # 4. Check existing passenger delay
        try:
            max_existing_passenger_delay_secs = self._calculate_total_passenger_inconvenience(new_route, current_route, vehicle) * 60

            logger.debug(f"Existing passenger delay: {max_existing_passenger_delay_secs / 60} mins (limit: {constraints['max_existing_passenger_delay_secs'] / 60} mins)")
            if max_existing_passenger_delay_secs > constraints["max_existing_passenger_delay_secs"]:
                violations["existing_passenger_delay"] = {
                    "actual": max_existing_passenger_delay_secs,
                    "limit": constraints["max_existing_passenger_delay_secs"]
                }
        except Exception as e:
            logger.error(f"Error calculating existing passenger delay: {str(e)}")
            violations["existing_passenger_delay"] = {
                "error": str(e),
                "reason": "Failed to calculate existing passenger delay"
            }

        # 5. Check distance
        try:
            if new_route.total_distance > constraints["max_distance_meters"]:
                violations["distance"] = {
                    "actual": new_route.total_distance,
                    "limit": constraints["max_distance_meters"]
                }
        except Exception as e:
            logger.error(f"Error checking distance constraint: {str(e)}")
            violations["distance"] = {
                "error": str(e),
                "reason": "Failed to check distance constraint"
            }

        return violations

    def _calculate_cost_components(
        self,
        new_route: Route,
        current_route: Route,
        stop_assignment: StopAssignment,
        pickup_idx: int,
        vehicle: Vehicle
    ) -> Dict[str, float]:
        """Calculate all cost components for optimization."""
        logger.debug(f"Calculating cost components for route modification:")
        logger.debug(f"Current route metrics: duration={current_route.total_duration if current_route else 0}, "
                  f"distance={current_route.total_distance if current_route else 0}")
        logger.debug(f"New route metrics: duration={new_route.total_duration}, distance={new_route.total_distance}")

        # Time-based costs (in seconds)
        passenger_waiting_time = abs((
            new_route.stops[pickup_idx].planned_arrival_time - 
            stop_assignment.expected_passenger_origin_stop_arrival_time
        ).total_seconds())

        # Find the dropoff index
        dropoff_idx = -1
        for i, stop in enumerate(new_route.stops):
            if stop_assignment.request_id in stop.dropoff_passengers:
                dropoff_idx = i
                break

        passenger_in_vehicle_time = (
            new_route.stops[dropoff_idx].planned_arrival_time -  
            new_route.stops[pickup_idx].planned_arrival_time
        ).total_seconds()

        # Convert helper function results from minutes to seconds
        existing_passenger_delay = self._calculate_total_passenger_inconvenience(new_route, current_route, vehicle) * 60

        # Distance (meters)
        distance = new_route.total_distance - (current_route.total_distance if current_route else 0)

        # Operational costs
        operational_cost = (
            self._estimate_fuel_cost(distance) + 
            self._estimate_time_cost(passenger_in_vehicle_time)
        )

        return {
            "passenger_waiting_time": passenger_waiting_time,
            "passenger_in_vehicle_time": passenger_in_vehicle_time,
            "existing_passenger_delay": existing_passenger_delay,
            "distance": distance,
            "operational_cost": operational_cost if "operational_cost" in self.weights.keys() else "N/A"
        }

    def _calculate_weighted_cost(self, cost_components: Dict[str, float]) -> float:
        """Calculate total weighted cost using configured weights."""
        total_cost = 0.0
        
        # Get constraint values from config for normalization
        constraints = self.config.constraints
        
        # Normalize cost components using constraint bounds
        normalized_costs = {}
        
        # Time-based components (convert seconds to ratio of max allowed)
        if "passenger_waiting_time" in cost_components:
            # Ensure waiting time is non-negative
            passenger_waiting_time = max(0, cost_components["passenger_waiting_time"])
            normalized_costs["passenger_waiting_time"] = min(1.0, 
                passenger_waiting_time / constraints["max_waiting_time_secs"])
            
        if "passenger_in_vehicle_time" in cost_components:
            normalized_costs["passenger_in_vehicle_time"] = min(1.0,
                cost_components["passenger_in_vehicle_time"] / constraints["max_in_vehicle_time_secs"])
            
        if "existing_passenger_delay" in cost_components:
            normalized_costs["existing_passenger_delay"] = min(1.0,
                cost_components["existing_passenger_delay"] / constraints["max_existing_passenger_delay_secs"])
            
        # Distance component
        if "distance" in cost_components:
            normalized_costs["distance"] = min(1.0,
                cost_components["distance"] / constraints["max_distance_meters"])
            
        # Operational cost (normalize to range 0-1 based on expected max cost)
        if "operational_cost" in cost_components and cost_components["operational_cost"] != "N/A":
            # Assuming max expected cost of $100
            normalized_costs["operational_cost"] = min(1.0, cost_components["operational_cost"] / 100)
        
        # Apply weights from config
        for component, weight in self.weights.items():
            if component in normalized_costs:
                total_cost += weight * normalized_costs[component]
            else:
                logger.warning(f"Weight defined for {component} but no corresponding cost component found")
        
        return total_cost

    def _calculate_vehicle_idle_time(self, route: Route) -> float:
        """Calculate total vehicle idle time in minutes"""
        if not route or len(route.stops) < 2:
            return 0.0
            
        idle_time = 0.0
        for i in range(len(route.stops) - 1):
            idle_time += (route.stops[i+1].planned_arrival_time - 
                         route.stops[i].planned_departure_time).total_seconds() / 60
        return idle_time
    
    def _estimate_fuel_cost(self, distance: float) -> float:
        """Estimate fuel cost based on distance"""
        # Simplified cost model: $0.15 per km
        return distance * 0.00015  # Convert meters to km and multiply by cost

    def _estimate_time_cost(self, duration: float) -> float:
        """Estimate time-based operational cost"""
        # Simplified cost model: $30 per hour
        return duration * 0.5  # Convert seconds to hours and multiply by cost

    def _map_violation_to_reason(self, violation_type: str) -> RejectionReason:
        """Map constraint violation type to rejection reason."""
        violation_map = {
            # Time-based violations
            "vehicle_access_time": RejectionReason.VEHICLE_ACCESS_TIME_CONSTRAINT,
            "waiting_time": RejectionReason.PASSENGER_WAIT_TIME_CONSTRAINT,
            "ride_time": RejectionReason.RIDE_TIME_CONSTRAINT,
            "existing_passenger_delay": RejectionReason.TIME_WINDOW_CONSTRAINT,
            "total_duration": RejectionReason.TOTAL_JOURNEY_TIME_CONSTRAINT,
            
            # Vehicle-based violations
            "vehicle_range": RejectionReason.VEHICLE_RANGE_CONSTRAINT,
            "shift_end": RejectionReason.VEHICLE_SHIFT_END_CONSTRAINT,
            
            # Passenger-based violations
            "walk_time": RejectionReason.PASSENGER_WALK_TIME_CONSTRAINT,
            "accessibility": RejectionReason.PASSENGER_ACCESSIBILITY_CONSTRAINT,
            
            # Cost-based violations
            "reserve_price": RejectionReason.RESERVE_PRICE_CONSTRAINT,
            "operational_cost": RejectionReason.OPERATIONAL_COST_CONSTRAINT,
            
            # System-based violations
            "system_capacity": RejectionReason.SYSTEM_CAPACITY_CONSTRAINT,
            "geographic": RejectionReason.GEOGRAPHIC_CONSTRAINT,
            
            # Distance-based violations
            "distance": RejectionReason.NO_FEASIBLE_INSERTION
        }
        
        # Log the violation for metrics tracking
        violation_type_normalized = violation_type.lower().replace(" ", "_")
        self.violation_counts[violation_type_normalized] = self.violation_counts.get(violation_type_normalized, 0) + 1
        
        return violation_map.get(violation_type, RejectionReason.NO_FEASIBLE_INSERTION)

    async def _evaluate_new_route(
        self,
        stop_assignment: StopAssignment,
        vehicle: Vehicle
    ) -> Optional[InsertionCost]:
        """Evaluates cost of creating new route for empty vehicle"""
        try:
            logger.info(f"Starting new route evaluation for vehicle {vehicle.id} with stop assignment {stop_assignment.id}")
            
            new_route = await self.route_service.create_new_route(
                vehicle=vehicle,
                stop_assignment=stop_assignment
            )
            
            if not new_route:
                logger.warning(f"Failed to create new route for vehicle {vehicle.id}")
                return None

            logger.debug(f"Successfully created new route for vehicle {vehicle.id}, checking timing compatibility")

            # Check timing compatibility for pickup stop
            pickup_stop = new_route.stops[0]  # First stop is pickup for new route
            if not self._is_stop_time_compatible(
                pickup_stop.planned_arrival_time,
                stop_assignment.expected_passenger_origin_stop_arrival_time
            ):
                logger.info(f"Stop timing incompatible for vehicle {vehicle.id}: " +
                          f"planned_arrival={pickup_stop.planned_arrival_time}, " +
                          f"passenger_arrival={stop_assignment.expected_passenger_origin_stop_arrival_time}")
                return None

            logger.debug(f"Stop timing compatible for vehicle {vehicle.id}, checking constraints")

            # Check all constraints
            violations = self._check_constraints(
                new_route=new_route,
                stop_assignment=stop_assignment,
                pickup_idx=0,  # For new routes, pickup is always first stop
                vehicle=vehicle,
                current_route=None  # No current route for new routes
            )
            if violations:
                logger.info(f"Constraint violations found for new route with vehicle {vehicle.id}: {violations}")
                return None

            logger.debug(f"No constraint violations found for vehicle {vehicle.id}, calculating costs")

            # Calculate costs using route data (all time-based components in seconds)
            cost_components = {
                "passenger_waiting_time": max((
                    new_route.stops[0].planned_arrival_time - 
                    stop_assignment.expected_passenger_origin_stop_arrival_time
                ).total_seconds(), 0.0),
                "passenger_in_vehicle_time": new_route.total_duration,
                "existing_passenger_delay": 0.0,  # No existing passengers
                "distance": new_route.total_distance,
                "operational_cost": (
                    self._estimate_fuel_cost(new_route.total_distance) + 
                    self._estimate_time_cost(new_route.total_duration)
                ) if "operational_cost" in self.weights.keys() else "N/A"
            }
            
            logger.debug(f"Calculated cost components for vehicle {vehicle.id}: {cost_components}")
            
            total_cost = self._calculate_weighted_cost(cost_components)
            
            logger.info(f"Successfully evaluated new route for vehicle {vehicle.id} with total cost {total_cost}")

            return InsertionCost(
                cost_components=cost_components,
                total_cost=total_cost,
                feasible=True,
                pickup_index=0,
                dropoff_index=1,
                updated_route=new_route,
                vehicle=vehicle
            )
        except Exception as e:
            logger.error(f"Error evaluating new route for vehicle {vehicle.id}: {str(e)}", exc_info=True)
            return None

    async def _create_assignment(
        self,
        best_vehicle_current_route: Optional[Route],
        stop_assignment: StopAssignment,
        insertion: InsertionCost,
        computation_start: datetime
    ) -> Optional[Assignment]:
        """Creates final assignment from best insertion"""
        try:
            route_pickup_stop = next(
                stop for stop in insertion.updated_route.stops
                if stop_assignment.request_id in stop.pickup_passengers
            )
            route_dropoff_stop = next(
                stop for stop in insertion.updated_route.stops
                if stop_assignment.request_id in stop.dropoff_passengers
            )
            
            computation_time = (datetime.now() - computation_start).total_seconds()
            
            logger.info(f"Creating assignment for request {stop_assignment.request_id} with vehicle {insertion.vehicle.id}\n Insertion cost components: {insertion.cost_components}")
            logger.debug(f"Is vehicle currently at stop: {insertion.vehicle.current_state.status == VehicleStatus.AT_STOP}")
            logger.debug(f"Vehicle current stop id: {insertion.vehicle.current_state.current_stop_id}")
            if best_vehicle_current_route:
                logger.debug(f"Best vehicle current route stops: {[str(stop) for stop in best_vehicle_current_route.stops]}")
            else:
                logger.debug(f"Best vehicle current route is None")
            logger.debug(f"New Route stops: {[str(stop) for stop in insertion.updated_route.stops]}")

            return Assignment(
                request_id=stop_assignment.request_id,
                vehicle_id=insertion.vehicle.id,
                route=insertion.updated_route,
                stop_assignment_id=stop_assignment.id,
                assignment_time=self.sim_context.current_time,
                estimated_pickup_time=route_pickup_stop.planned_arrival_time,
                estimated_dropoff_time=route_dropoff_stop.planned_arrival_time,
                waiting_time_mins=insertion.cost_components["passenger_waiting_time"] / 60,
                in_vehicle_time_mins=insertion.cost_components["passenger_in_vehicle_time"] / 60,
                assignment_score=1.0 - insertion.total_cost,
                computation_time=computation_time,
                metadata={
                    "insertion_cost_components": insertion.cost_components,
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

    def _get_first_valid_index(self, route: Route, vehicle: Vehicle) -> int:
        """Gets index of first valid insertion point considering vehicle status"""
        first_non_completed_idx = len(route.stops)
        
        for i, stop in enumerate(route.stops):
            if not stop.completed:
                first_non_completed_idx = i
                break
        
        # For vehicles at a stop, we need to ensure we're not inserting before
        # the current active stop is completed
        if vehicle.current_state.status == VehicleStatus.AT_STOP:
            # Find the index of the stop the vehicle is currently at
            for i, stop in enumerate(route.stops):
                if (not stop.completed and 
                    stop.stop.id == vehicle.current_state.current_stop_id):
                    # We can only insert after this stop
                    return i + 1
        return first_non_completed_idx


    def _is_stop_time_compatible(
        self,
        planned_arrival_time: datetime,
        expected_passenger_arrival_time: datetime
    ) -> bool:
        """
        Checks if stop timing is compatible for pickup
        
        Args:
            planned_arrival_time: Vehicle's planned arrival time at stop
            expected_passenger_arrival_time: Expected passenger arrival time at stop
            
        Returns:
            True if timing is compatible, False otherwise
        """
        if not planned_arrival_time:
            return False
            
        # For pickups, ensure passenger arrives before vehicle's planned arrival
        return expected_passenger_arrival_time < planned_arrival_time
    
    def _calculate_total_passenger_inconvenience(self, new_route: Route, current_route: Optional[Route] = None, vehicle: Optional[Vehicle] = None) -> float:
        """
        Calculate the maximum total inconvenience for any existing passenger by combining
        pickup delay and in-vehicle (ride) delay.
        
        Returns the maximum additional delay in minutes.
        """
        if not current_route:
            return 0.0  # No inconvenience if there is no current route

        first_valid_idx = self._get_first_valid_index(current_route, vehicle)
        max_inconvenience = 0.0

        # Gather all passengers from the original route (from the first non-completed stop onward)
        original_passengers = set()
        for stop in current_route.stops[first_valid_idx:]:
            original_passengers.update(stop.pickup_passengers)
            original_passengers.update(stop.dropoff_passengers)

        for passenger_request_id in original_passengers:
            # Original pickup and dropoff in the current route
            orig_pickup = next(
                (stop for stop in current_route.stops[first_valid_idx:] if passenger_request_id in stop.pickup_passengers),
                None
            )
            orig_dropoff = next(
                (stop for stop in current_route.stops[first_valid_idx:] if passenger_request_id in stop.dropoff_passengers),
                None
            )
            # New pickup and dropoff in the modified route
            new_pickup = next(
                (stop for stop in new_route.stops if passenger_request_id in stop.pickup_passengers),
                None
            )
            new_dropoff = next(
                (stop for stop in new_route.stops if passenger_request_id in stop.dropoff_passengers),
                None
            )
            
            if not (orig_pickup and orig_dropoff and new_pickup and new_dropoff):
                logger.warning(f"Missing stops for passenger {passenger_request_id} in current or new route")
                continue

            # Calculate pickup delay (only count if the new pickup is later)
            pickup_delay = (new_pickup.planned_arrival_time - orig_pickup.planned_arrival_time).total_seconds() / 60
            pickup_delay = max(pickup_delay, 0)

            # Calculate ride delay: the extra time added during the ride segment
            orig_ride_time = (orig_dropoff.planned_arrival_time - orig_pickup.planned_arrival_time).total_seconds() / 60
            new_ride_time = (new_dropoff.planned_arrival_time - new_pickup.planned_arrival_time).total_seconds() / 60
            ride_delay = max(new_ride_time - orig_ride_time, 0)

            total_delay = pickup_delay + ride_delay
            max_inconvenience = max(max_inconvenience, total_delay)

        return max_inconvenience