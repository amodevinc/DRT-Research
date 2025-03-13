from abc import ABC
from typing import List, Dict, Any, Union

from drt_sim.models.vehicle import Vehicle
from drt_sim.network.manager import NetworkManager
from drt_sim.models.matching import (
    Assignment, AssignmentMethod
)
from drt_sim.config.config import MatchingConfig
from drt_sim.core.state.manager import StateManager
from drt_sim.algorithms.matching.assignment.insertion import InsertionAssigner
from drt_sim.algorithms.matching.assignment.auction import AuctionAssigner
from drt_sim.core.simulation.context import SimulationContext

from drt_sim.core.user.user_profile_manager import UserProfileManager
from drt_sim.core.services.route_service import RouteService
from drt_sim.models.stop import StopAssignment
import logging
logger = logging.getLogger(__name__)

class MatchingStrategy(ABC):
    """Abstract base class for matching strategies."""
    
    def __init__(
        self,
        sim_context: SimulationContext,
        config: MatchingConfig,
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
        self.performance_metrics = self._initialize_metrics()
        
        # Initialize components based on configuration
        self.assigner = self._initialize_assigner()
        
        logger.info(f"Initialized MatchingStrategy with Assignment={self.config.assignment_method.value}")

    def _initialize_assigner(self) -> Union[InsertionAssigner, AuctionAssigner]:
        """Initialize assignment component based on configuration."""
        assigners = {
            AssignmentMethod.INSERTION: InsertionAssigner,
            AssignmentMethod.AUCTION: AuctionAssigner,
        }
        
        assigner_class = assigners.get(self.config.assignment_method)
        if not assigner_class:
            raise ValueError(f"Unknown assignment method: {self.config.assignment_method}")
            
        return assigner_class(
            sim_context=self.sim_context,
            config=self.config.assignment_config,
            network_manager=self.network_manager,
            state_manager=self.state_manager,
            user_profile_manager=self.user_profile_manager,
            route_service=self.route_service
        )

    async def match_stop_assignment_to_vehicle(
        self,
        stop_assignment: StopAssignment,
    ) -> List[Assignment]:
        """
        Quick matching of requests to vehicles using configured methods.
        This is the main method used for immediate request handling.
        """
        try:
            # Step 1: Quick Assignment
            logger.debug("Performing quick assignment")
            # Call the async assign_requests method
            assignment = await self.assigner.assign_request(
                stop_assignment
            )

            return assignment
            
        except Exception as e:
            import traceback
            logger.error(f"Error in match_requests_to_vehicles: {str(e)}\n{traceback.format_exc()}")
            raise

    def _initialize_metrics(self) -> Dict[str, Any]:
        """Initialize performance metrics."""
        return {
            'total_requests': 0,
            'successful_matches': 0,
            'rejected_requests': 0,
            'average_waiting_time': 0.0,
            'average_computation_time': 0.0,
        }