from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from enum import Enum
from drt_sim.models.stop import Stop, StopAssignment
from drt_sim.models.request import Request
from drt_sim.network.manager import NetworkManager
from drt_sim.core.simulation.context import SimulationContext
from drt_sim.core.state.manager import StateManager

class StopAssignmentStrategy(Enum):
    """Available strategies for stop assignment."""
    NEAREST = "nearest"
    ACCESSIBILITY = "accessibility"
    MULTI_OBJECTIVE = "multi_objective"

@dataclass
class StopAssignerConfig:
    """Configuration for stop assignment."""
    strategy: StopAssignmentStrategy
    max_alternatives: int = 3
    thresholds: Dict[str, float] = field(default_factory=lambda: {
        "max_walking_distance": 400.0,
        "max_driving_time": 900.0,
    })
    weights: Dict[str, float] = field(default_factory=lambda: {
        "vehicle_access_time": 0.3,
        "passenger_access_time": 0.7,
    })
    custom_params: Dict[str, Any] = field(default_factory=dict)


class StopAssigner(ABC):
    """Abstract base class for operational stop assignment."""
    
    def __init__(self, sim_context: SimulationContext, config: StopAssignerConfig, network_manager: NetworkManager, state_manager: StateManager):
        self.sim_context = sim_context
        self.config = config
        self.network_manager = network_manager
        self.state_manager = state_manager
        self._validate_config()

    @abstractmethod
    def assign_stops(self, 
                    request: Request,
                    available_stops: List[Stop],
                    system_state: Optional[Dict[str, Any]] = None) -> StopAssignment:
        """
        Assign optimal stops for a specific request.
        
        Args:
            request: Trip request to assign stops for
            available_stops: Currently available stops
            system_state: Current system state
            
        Returns:
            StopAssignment with selected stops and metrics
        
        Raises:
            ValueError: If no viable stops found
        """
        pass

    def _validate_config(self) -> None:
        """Validate configuration parameters."""
        if sum(self.config.weights.values()) != 1.0:
            raise ValueError("weights must sum to 1.0")