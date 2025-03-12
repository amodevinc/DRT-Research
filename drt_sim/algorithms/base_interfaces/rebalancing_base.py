from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Tuple
from enum import Enum
from datetime import datetime

from drt_sim.models.vehicle import Vehicle, VehicleStatus
from drt_sim.models.stop import Stop
from drt_sim.models.route import Route
from drt_sim.network.manager import NetworkManager
from drt_sim.core.state.manager import StateManager
from drt_sim.core.simulation.context import SimulationContext
from drt_sim.core.services.route_service import RouteService

import logging
logger = logging.getLogger(__name__)

class RebalancingStrategy(Enum):
    """Available strategies for vehicle rebalancing."""
    NAIVE = "naive"
    DEMAND_BASED = "demand_based"
    PREDICTIVE = "predictive"
    ZONE_BASED = "zone_based"
    OPTIMIZATION = "optimization"

class RebalancingConfig:
    """Configuration for rebalancing algorithms."""
    
    def __init__(
        self,
        strategy: RebalancingStrategy,
        min_battery_level: Optional[float] = 20.0,
        weights: Dict[str, float] = None,
        **kwargs
    ):
        """
        Initialize rebalancing configuration.
        
        Args:
            strategy: The rebalancing strategy to use
            min_battery_level: Minimum battery level required for rebalancing (for electric vehicles)
            weights: Dictionary of weights for different factors in rebalancing decisions
            **kwargs: Additional strategy-specific parameters
        """
        self.strategy = strategy
        self.min_battery_level = min_battery_level
        self.weights = weights or {"distance": 0.5, "demand": 0.5}
        
        # Store any additional parameters
        for key, value in kwargs.items():
            setattr(self, key, value)
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'RebalancingConfig':
        """
        Create a RebalancingConfig instance from a dictionary.
        
        Args:
            config_dict: Dictionary containing configuration parameters
            
        Returns:
            RebalancingConfig instance
        """
        # Extract known parameters
        strategy_value = config_dict.get('strategy', 'naive')
        # Convert string to enum if needed
        if isinstance(strategy_value, str):
            strategy = RebalancingStrategy(strategy_value)
        else:
            strategy = strategy_value
            
        min_battery_level = config_dict.get('min_battery_level', 20.0)
        weights = config_dict.get('weights', {"distance": 0.5, "demand": 0.5})
        
        # Create kwargs dict with remaining parameters
        kwargs = {k: v for k, v in config_dict.items() 
                 if k not in ['strategy', 'min_battery_level', 'weights']}
        
        # Create and return instance
        return cls(
            strategy=strategy,
            min_battery_level=min_battery_level,
            weights=weights,
            **kwargs
        )

class RebalancingAlgorithm(ABC):
    """Abstract base class for vehicle rebalancing algorithms."""
    
    def __init__(
        self,
        sim_context: SimulationContext,
        config: Dict[str, Any] | RebalancingConfig,
        network_manager: NetworkManager,
        state_manager: StateManager,
        route_service: RouteService
    ):
        """
        Initialize the rebalancing algorithm.
        
        Args:
            sim_context: Simulation context
            config: Rebalancing configuration (dict or RebalancingConfig)
            network_manager: Network manager for distance calculations
            state_manager: State manager for accessing system state
            route_service: Route service for creating routes
        """
        self.sim_context = sim_context
        
        # Convert dict config to RebalancingConfig object if needed
        if isinstance(config, dict):
            self.config = RebalancingConfig.from_dict(config)
        else:
            self.config = config
            
        self.network_manager = network_manager
        self.state_manager = state_manager
        self.route_service = route_service
        self._validate_config()
        
        logger.info(f"Initialized RebalancingAlgorithm with strategy={self.config.strategy.value}")
    
    @abstractmethod
    def get_rebalancing_target(
        self,
        vehicle: Vehicle,
        current_time: Optional[datetime] = None,
        available_stops: Optional[List[Stop]] = None,
        system_state: Optional[Dict[str, Any]] = None
    ) -> Optional[Stop]:
        """
        Determine the target stop for rebalancing a vehicle.
        
        Args:
            vehicle: The vehicle to rebalance
            current_time: Current simulation time
            available_stops: List of available stops for rebalancing
            system_state: Current system state information
            
        Returns:
            Target stop for rebalancing or None if no rebalancing needed
        """
        pass
    
    @abstractmethod
    async def create_rebalancing_route(
        self,
        vehicle: Vehicle,
        target_stop: Stop,
        current_time: datetime
    ) -> Route:
        """
        Create a route for rebalancing a vehicle to the target stop.
        
        Args:
            vehicle: The vehicle to rebalance
            target_stop: The target stop for rebalancing
            current_time: Current simulation time
            
        Returns:
            Route object for the rebalancing journey
        """
        pass
    
    @abstractmethod
    def should_rebalance(
        self,
        vehicle: Vehicle,
        current_time: datetime,
        system_state: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Determine if a vehicle should be rebalanced.
        
        Args:
            vehicle: The vehicle to potentially rebalance
            current_time: Current simulation time
            system_state: Current system state information
            
        Returns:
            Boolean indicating whether the vehicle should be rebalanced
        """
        pass
    
    def get_idle_vehicles(
        self,
        vehicles: List[Vehicle],
        current_time: datetime
    ) -> List[Vehicle]:
        """
        Get a list of idle vehicles that are candidates for rebalancing.
        
        Args:
            vehicles: List of all vehicles in the system
            current_time: Current simulation time
            
        Returns:
            List of idle vehicles that are candidates for rebalancing
        """
        idle_vehicles = []
        
        for vehicle in vehicles:
            if self.should_rebalance(vehicle, current_time):
                idle_vehicles.append(vehicle)
                
        return idle_vehicles
    
    def _validate_config(self) -> None:
        """Validate the configuration parameters."""
        
        if self.config.min_battery_level is not None and (self.config.min_battery_level < 0 or self.config.min_battery_level > 100):
            raise ValueError("min_battery_level must be between 0 and 100 if specified")
        
        # Validate weights
        if sum(self.config.weights.values()) != 1.0:
            raise ValueError("weights must sum to 1.0")
