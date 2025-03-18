from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta
import asyncio
from dataclasses import dataclass
import traceback
import math
import logging

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
from drt_sim.models.request import Request
from drt_sim.core.user.acceptance_context import AcceptanceContext
from drt_sim.algorithms.user_acceptance.logit import LogitModel

# Import InsertionAssigner as the base class
from drt_sim.algorithms.matching.assignment.insertion import InsertionAssigner, InsertionCost

logger = logging.getLogger(__name__)

@dataclass
class ProbabilityInsertionCost(InsertionCost):
    """
    Extends InsertionCost with acceptance probability information.
    """
    acceptance_probability: float = 0.0
    combined_score: float = 0.0

class ProbabilityInsertionAssigner(InsertionAssigner):
    """
    Extends the insertion heuristic to consider user acceptance probability.
    
    This algorithm combines the traditional insertion cost with the probability
    that a user will accept the ride, aiming to maximize the expected value of
    successful matches.
    """
    
    def __init__(
        self,
        sim_context: SimulationContext,
        config: MatchingAssignmentConfig,
        network_manager: NetworkManager,
        state_manager: StateManager,
        user_profile_manager: UserProfileManager,
        route_service: RouteService,
        acceptance_model: Optional[LogitModel] = None,
        probability_weight: float = 0.5
    ):
        """
        Initialize the probability-based insertion assigner.
        
        Args:
            sim_context: Simulation context
            config: Matching assignment configuration
            network_manager: Network manager
            state_manager: State manager
            user_profile_manager: User profile manager
            route_service: Route service
            acceptance_model: Acceptance model to use for probability calculation
            probability_weight: Weight for acceptance probability in the combined score (0-1)
        """
        super().__init__(
            sim_context=sim_context,
            config=config,
            network_manager=network_manager,
            state_manager=state_manager,
            user_profile_manager=user_profile_manager,
            route_service=route_service
        )
        
        # Initialize acceptance model if not provided
        self.acceptance_model = acceptance_model or LogitModel()
        
        # Configure the weight of probability vs. cost in the combined score
        self.probability_weight = probability_weight
        
        # Track additional statistics
        self.stats = {
            "total_assignments": 0,
            "avg_probability": 0.0,
            "high_probability_assignments": 0,  # Probability > 0.8
            "medium_probability_assignments": 0,  # 0.5 < Probability <= 0.8
            "low_probability_assignments": 0,  # Probability <= 0.5
        }
        
        logger.info(f"ProbabilityInsertionAssigner initialized with probability_weight={probability_weight}")
    
    async def assign_request(
        self,
        stop_assignment: StopAssignment,
    ) -> Tuple[Optional[Assignment], Optional[RejectionMetadata]]:
        """
        Assigns request to best available vehicle considering both insertion cost and acceptance probability.
        
        Args:
            stop_assignment: Stop assignment to be inserted
            
        Returns:
            Tuple of (Assignment if feasible match found, RejectionMetadata if rejected)
        """
        try:
            computation_start = datetime.now()
            available_vehicles = self.state_manager.vehicle_worker.get_available_vehicles()
            logger.info(f"Starting assignment for request {stop_assignment.request_id} with {len(available_vehicles)} available vehicles")
            
            # Get request from state manager
            request = self.state_manager.request_worker.get_request(stop_assignment.request_id)
            if not request:
                logger.error(f"Request {stop_assignment.request_id} not found")
                return None, RejectionMetadata(
                    reason=RejectionReason.TECHNICAL_ERROR,
                    timestamp=self.sim_context.current_time.isoformat(),
                    stage="matching",
                    details={"error": "Request not found"}
                )
            
            # Get user profile
            user_id = request.user_id
            user_profile = self.user_profile_manager.get_profile(user_id)
            if not user_profile:
                logger.warning(f"User profile for {user_id} not found")
                # Continue without user profile - will use default probability
            
            # Create tasks for parallel evaluation
            insertion_tasks = [
                self._evaluate_vehicle_insertion_with_probability(
                    stop_assignment=stop_assignment,
                    vehicle=vehicle,
                    request=request,
                    user_profile=user_profile
                )
                for vehicle in available_vehicles
            ]
            
            # Evaluate all vehicles in parallel
            insertion_results = await asyncio.gather(*insertion_tasks)
            
            # Filter out None results and keep only feasible insertions
            feasible_insertions = [
                result for result in insertion_results 
                if result and result.feasible
            ]
            
            if not feasible_insertions:
                # Handle rejection similar to parent class
                rejection_reasons = [
                    result.rejection_reason for result in insertion_results 
                    if result and result.rejection_reason
                ]
                
                most_common_reason = max(
                    set(rejection_reasons),
                    key=rejection_reasons.count,
                    default=RejectionReason.UNKNOWN
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
                
                # Add constraint violation details
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
            
            # Find the best insertion based on combined score
            best_insertion = max(
                feasible_insertions,
                key=lambda x: x.combined_score
            )
            
            # Update stats
            self.stats["total_assignments"] += 1
            self.stats["avg_probability"] = (
                (self.stats["avg_probability"] * (self.stats["total_assignments"] - 1) + 
                 best_insertion.acceptance_probability) / self.stats["total_assignments"]
            )
            
            if best_insertion.acceptance_probability > 0.8:
                self.stats["high_probability_assignments"] += 1
            elif best_insertion.acceptance_probability > 0.5:
                self.stats["medium_probability_assignments"] += 1
            else:
                self.stats["low_probability_assignments"] += 1
            
            # Get best vehicle's current route for assignment creation
            best_vehicle_current_route_id = self.state_manager.vehicle_worker.get_vehicle_active_route_id(best_insertion.vehicle.id)
            best_vehicle_current_route = self.state_manager.route_worker.get_route(best_vehicle_current_route_id)
            
            # Create assignment
            assignment = await self._create_assignment(
                best_vehicle_current_route=best_vehicle_current_route,
                stop_assignment=stop_assignment,
                insertion=best_insertion,
                computation_start=computation_start,
                acceptance_probability=best_insertion.acceptance_probability
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
    
    async def _evaluate_vehicle_insertion_with_probability(
        self,
        stop_assignment: StopAssignment,
        vehicle: Vehicle,
        request: Request,
        user_profile: Optional[Any] = None
    ) -> Optional[ProbabilityInsertionCost]:
        """
        Evaluate inserting a stop assignment into a vehicle's route, considering acceptance probability.
        
        Args:
            stop_assignment: Stop assignment to evaluate
            vehicle: Vehicle to consider
            request: Original request
            user_profile: User profile for probability calculation
            
        Returns:
            Optional[ProbabilityInsertionCost]: Extended insertion cost with probability information
        """
        try:
            # Get standard insertion evaluation result
            insertion_result = await self._evaluate_vehicle_insertion(stop_assignment, vehicle)
            
            if not insertion_result or not insertion_result.feasible:
                # If not feasible, return as is (with default probability values if it's our subclass)
                if isinstance(insertion_result, ProbabilityInsertionCost):
                    return insertion_result
                
                # Convert to our subclass if it's a standard InsertionCost
                if insertion_result:
                    return ProbabilityInsertionCost(
                        cost_components=insertion_result.cost_components,
                        total_cost=insertion_result.total_cost,
                        feasible=insertion_result.feasible,
                        pickup_index=insertion_result.pickup_index,
                        dropoff_index=insertion_result.dropoff_index,
                        updated_route=insertion_result.updated_route,
                        vehicle=insertion_result.vehicle,
                        rejection_reason=insertion_result.rejection_reason,
                        rejection_details=insertion_result.rejection_details,
                        acceptance_probability=0.0,
                        combined_score=0.0
                    )
                    
                return None
            
            # Calculate acceptance probability
            service_attributes = self._extract_service_attributes(insertion_result, stop_assignment)
            
            # Create acceptance context from the evaluated insertion
            context = AcceptanceContext.from_assignment(
                request=request,
                service_attributes=service_attributes,
                user_profile=user_profile
            )
            
            # Calculate acceptance probability using our model
            probability = self.acceptance_model.calculate_acceptance_probability(context)
            
            # Calculate combined score: weighted average of normalized cost and probability
            # Normalize cost to a score in [0,1] range (higher is better)
            cost_score = 1.0 - min(insertion_result.total_cost, 1.0)
            
            # Combine scores using the configured weight
            # combined_score = (1-w) * cost_score + w * probability
            combined_score = (
                (1.0 - self.probability_weight) * cost_score + 
                self.probability_weight * probability
            )
            
            logger.debug(f"Vehicle {vehicle.id}: cost_score={cost_score:.4f}, probability={probability:.4f}, "
                      f"combined_score={combined_score:.4f}")
            
            # Create extended result
            result = ProbabilityInsertionCost(
                cost_components=insertion_result.cost_components,
                total_cost=insertion_result.total_cost,
                feasible=insertion_result.feasible,
                pickup_index=insertion_result.pickup_index,
                dropoff_index=insertion_result.dropoff_index,
                updated_route=insertion_result.updated_route,
                vehicle=insertion_result.vehicle,
                rejection_reason=insertion_result.rejection_reason,
                rejection_details=insertion_result.rejection_details,
                acceptance_probability=probability,
                combined_score=combined_score
            )
            
            return result
            
        except Exception as e:
            logger.error(f"Error evaluating vehicle insertion with probability for vehicle {vehicle.id}: {str(e)}", 
                       exc_info=True)
            
            # Return a failed result
            return ProbabilityInsertionCost(
                cost_components={},
                total_cost=float('inf'),
                feasible=False,
                pickup_index=-1,
                dropoff_index=-1,
                rejection_reason=RejectionReason.TECHNICAL_ERROR,
                rejection_details={"error": str(e)},
                acceptance_probability=0.0,
                combined_score=0.0
            )
    
    def _extract_service_attributes(
        self,
        insertion_result: InsertionCost,
        stop_assignment: StopAssignment
    ) -> Dict[str, Any]:
        """
        Extract service attributes from an insertion result for acceptance modeling.
        
        Args:
            insertion_result: Insertion evaluation result
            stop_assignment: Stop assignment being inserted
            
        Returns:
            Dict[str, Any]: Service attributes for acceptance context
        """
        # Find pickup and dropoff stops
        pickup_stop = None
        dropoff_stop = None
        
        for stop in insertion_result.updated_route.stops:
            if stop_assignment.request_id in stop.pickup_passengers:
                pickup_stop = stop
            if stop_assignment.request_id in stop.dropoff_passengers:
                dropoff_stop = stop
            
            # If found both, break
            if pickup_stop and dropoff_stop:
                break
        
        # Calculate time-related attributes (convert to minutes for user model)
        walking_time_to_origin = 0  # Placeholder - would need data from request
        
        waiting_time = (
            (pickup_stop.planned_arrival_time - stop_assignment.expected_passenger_origin_stop_arrival_time).total_seconds() / 60
            if pickup_stop and stop_assignment.expected_passenger_origin_stop_arrival_time 
            else insertion_result.cost_components.get("passenger_waiting_time", 0) / 60
        )
        
        in_vehicle_time = (
            (dropoff_stop.planned_arrival_time - pickup_stop.planned_arrival_time).total_seconds() / 60
            if pickup_stop and dropoff_stop
            else insertion_result.cost_components.get("passenger_in_vehicle_time", 0) / 60
        )
        
        walking_time_from_destination = 0  # Placeholder - would need data from request
        
        # Additional attributes
        distance_to_pickup = insertion_result.updated_route.stops[0].distance_from_previous / 1000  # km
        
        # Create service attributes dictionary
        service_attributes = {
            "walking_time_to_origin": walking_time_to_origin,
            "waiting_time": max(0.0, waiting_time),  # Ensure non-negative
            "in_vehicle_time": max(0.0, in_vehicle_time),  # Ensure non-negative
            "walking_time_from_destination": walking_time_from_destination,
            "distance_to_pickup": distance_to_pickup,
            # Add time of day if available
            "time_of_day": self.sim_context.current_time.hour + self.sim_context.current_time.minute / 60.0,
            # Add day of week (0=Monday, 6=Sunday)
            "day_of_week": self.sim_context.current_time.weekday()
        }
        
        return service_attributes
    
    async def _create_assignment(
        self,
        best_vehicle_current_route: Optional[Route],
        stop_assignment: StopAssignment,
        insertion: InsertionCost,
        computation_start: datetime,
        acceptance_probability: float = 0.0
    ) -> Optional[Assignment]:
        """
        Creates final assignment from best insertion.
        
        Args:
            best_vehicle_current_route: Current route of the best vehicle
            stop_assignment: Stop assignment to insert
            insertion: Insertion cost result
            computation_start: Start time of computation for tracking
            acceptance_probability: Probability of user accepting this assignment
            
        Returns:
            Optional[Assignment]: Created assignment
        """
        # Use parent implementation to create the base assignment
        assignment = await super()._create_assignment(
            best_vehicle_current_route=best_vehicle_current_route,
            stop_assignment=stop_assignment,
            insertion=insertion,
            computation_start=computation_start
        )
        
        if not assignment:
            return None
        
        # Add probability information to metadata
        if not assignment.metadata:
            assignment.metadata = {}
            
        assignment.metadata.update({
            "acceptance_probability": acceptance_probability,
            "probability_weight": self.probability_weight,
        })
        
        if isinstance(insertion, ProbabilityInsertionCost):
            assignment.metadata.update({
                "combined_score": insertion.combined_score
            })
        
        return assignment
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the probability-based assignments.
        
        Returns:
            Dict[str, Any]: Statistics
        """
        return self.stats.copy() 