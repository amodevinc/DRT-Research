from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Set, Tuple, TYPE_CHECKING
from dataclasses import dataclass
from enum import Enum
import numpy as np
import uuid

from drt_sim.models.stop import Stop, StopStatus
from drt_sim.models.location import Location
from drt_sim.models.request import Request
from drt_sim.network.manager import NetworkManager
from drt_sim.models.simulation import SimulationState
from drt_sim.core.simulation.context import SimulationContext
class StopSelectorStrategy(Enum):
    """Enumeration of different stop selection strategies."""
    COVERAGE_BASED = "coverage_based"
    DEMAND_BASED = "demand_based"
    CLUSTERING = "clustering"
    HYBRID = "hybrid"
    OPTIMIZATION = "optimization"

@dataclass
class StopSelectorConfig:
    """Configuration parameters for stop selection."""
    strategy: StopSelectorStrategy
    max_walking_distance: float  # meters
    min_stop_spacing: float  # meters
    default_virtual_stop_capacity: int = 10
    max_stops: Optional[int] = None
    coverage_radius: Optional[float] = None  # meters
    min_demand_threshold: Optional[float] = None
    accessibility_weights: Optional[Dict[str, float]] = None
    optimization_constraints: Optional[Dict[str, Any]] = None
    custom_params: Optional[Dict[str, Any]] = None

class StopSelector(ABC):
    """Abstract base class for stop selection algorithms."""
    
    def __init__(self, sim_context: SimulationContext, config: StopSelectorConfig, network_manager: NetworkManager):
        """
        Initialize the stop selector with configuration parameters.
        
        Args:
            config: StopSelectorConfig object containing selection parameters
        """
        self.sim_context = sim_context
        self.config = config
        self.network_manager = network_manager
        self.selected_stops: Set[str] = set()  # Track selected stop IDs
        self._validate_config()
    @abstractmethod
    def select_stops(self, 
                    candidate_locations: List[Location],
                    demand_points: Optional[List[Location]] = None,
                    existing_stops: Optional[List[Stop]] = None,
                    constraints: Optional[Dict[str, Any]] = None) -> List[Stop]:
        """
        Select optimal stop locations based on the implementation strategy.
        
        Args:
            candidate_locations: List of possible stop locations
            demand_points: Optional list of demand point locations
            existing_stops: Optional list of existing stops to consider
            constraints: Optional additional constraints for selection
            
        Returns:
            List of selected Stop objects
        """
        pass

    @abstractmethod
    def update_stops(self,
                    current_stops: List[Stop],
                    demand_changes: Dict[str, float],
                    system_state: SimulationState) -> Tuple[List[Stop], List[str]]:
        """
        Update stop selection based on changes in demand or system state.
        
        Args:
            current_stops: List of currently active stops
            demand_changes: Dictionary mapping areas to demand change factors
            system_state: Optional current state of the system
            
        Returns:
            Tuple of (updated stop list, list of modified stop IDs)
        """
        pass

    @abstractmethod
    async def create_virtual_stops_for_request(self,
                                       request: Request,
                                       existing_stops: List[Stop],
                                       system_state: SimulationState) -> Tuple[Stop, Stop]:
        """
        Create optimally placed virtual stops for a specific request when no existing stops are viable.
        """
        pass

    def validate_stop_placement(self, 
                              proposed_stop: Location, 
                              existing_stops: List[Stop]) -> bool:
        """
        Validate if a proposed stop location meets spacing and other requirements.
        
        Args:
            proposed_stop: Location object for the proposed stop
            existing_stops: List of existing Stop objects
            
        Returns:
            Boolean indicating if the placement is valid
        """
        # Check minimum spacing requirements
        for stop in existing_stops:
            distance = self._calculate_distance(proposed_stop, stop.location)
            if distance < self.config.min_stop_spacing:
                return False
        return True

    def _validate_config(self) -> None:
        """Validate the configuration parameters."""
        if self.config.max_walking_distance <= 0:
            raise ValueError("max_walking_distance must be positive")
        if self.config.min_stop_spacing <= 0:
            raise ValueError("min_stop_spacing must be positive")
        if self.config.max_stops is not None and self.config.max_stops <= 0:
            raise ValueError("max_stops must be positive if specified")
        
        # Strategy-specific validation
        if self.config.strategy == StopSelectorStrategy.COVERAGE_BASED:
            if not self.config.coverage_radius:
                raise ValueError("coverage_radius required for COVERAGE_BASED strategy")
        elif self.config.strategy == StopSelectorStrategy.DEMAND_BASED:
            if not self.config.min_demand_threshold:
                raise ValueError("min_demand_threshold required for DEMAND_BASED strategy")

    def _calculate_distance(self, loc1: Location, loc2: Location) -> float:
        """
        Calculate the distance between two locations using the network manager.
        
        Args:
            loc1: First location
            loc2: Second location
            
        Returns:
            Distance in meters
        """
        R = 6371000  # Earth's radius in meters
        lat1, lon1 = np.radians(loc1.lat), np.radians(loc1.lon)
        lat2, lon2 = np.radians(loc2.lat), np.radians(loc2.lon)
        
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        
        a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
        c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))
        
        return R * c

    async def _calculate_network_distance(self, 
                                  loc1: Location, 
                                  loc2: Location, 
                                  network_type: str = 'drive') -> float:
        """
        Calculate the distance between two locations using the network manager.
        
        Args:
            loc1: First location
            loc2: Second location
            network_type: Type of network to use ('drive' or 'walk')
            
        Returns:
            Distance in meters between locations using the specified network.
            Returns infinity if no path is found.
        """
        if network_type not in ['drive', 'walk']:
            raise ValueError("network_type must be either 'drive' or 'walk'")
        # Get shortest path length between nodes
        _, distance = await self.network_manager.get_shortest_path(
            loc1, 
            loc2, 
            network_type=network_type,
            weight='distance'
        )
        return distance

    def _create_stop(self,
                    location: Location,
                    name: Optional[str] = None,
                    metadata: Optional[Dict[str, Any]] = None) -> Stop:
        """
        Create a new Stop object with generated ID and provided parameters.
        
        Args:
            location: Location object for the stop
            name: Optional name for the stop
            metadata: Optional metadata dictionary
            
        Returns:
            New Stop object
        """
        stop_id = str(uuid.uuid4())
        return Stop(
            id=stop_id,
            name=name or f"Stop_{stop_id[:8]}",
            location=location,
            status=StopStatus.ACTIVE,
            metadata=metadata or {}
        )