from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta

from drt_sim.algorithms.base_interfaces.rebalancing_base import RebalancingAlgorithm, RebalancingConfig, RebalancingStrategy
from drt_sim.models.vehicle import Vehicle, VehicleStatus
from drt_sim.models.stop import Stop
from drt_sim.models.route import Route, RouteStatus, RouteType
from drt_sim.network.manager import NetworkManager
from drt_sim.core.state.manager import StateManager
from drt_sim.core.simulation.context import SimulationContext
from drt_sim.core.services.route_service import RouteService

import logging
logger = logging.getLogger(__name__)

class NaiveRebalancingAlgorithm(RebalancingAlgorithm):
    """
    A naive rebalancing algorithm that simply routes idle vehicles back to their depot stops.
    
    This algorithm checks for idle vehicles and sends them back to their home depot
    when they've been idle for longer than the configured threshold.
    """
    
    def __init__(
        self,
        sim_context: SimulationContext,
        config: RebalancingConfig,
        network_manager: NetworkManager,
        state_manager: StateManager,
        route_service: RouteService
    ):
        """Initialize the naive rebalancing algorithm."""
        super().__init__(sim_context, config, network_manager, state_manager, route_service)
    
    def get_rebalancing_target(
        self,
        vehicle: Vehicle,
        current_time: Optional[datetime] = None,
        available_stops: Optional[List[Stop]] = None,
        system_state: Optional[Dict[str, Any]] = None
    ) -> Optional[Stop]:
        """
        Get the rebalancing target for a vehicle, which is simply its depot stop.
        
        Args:
            vehicle: The vehicle to rebalance
            current_time: Current simulation time
            available_stops: List of available stops (not used in naive implementation)
            system_state: Current system state (not used in naive implementation)
            
        Returns:
            The vehicle's depot stop or None if no depot is assigned
        """
        if not vehicle.depot_stop:
            logger.warning(f"Vehicle {vehicle.id} has no depot stop assigned, cannot rebalance")
            return None
        
        # Check if vehicle is already at its depot
        if (vehicle.current_state.current_stop_id and 
            vehicle.current_state.current_stop_id == vehicle.depot_stop.id):
            logger.debug(f"Vehicle {vehicle.id} is already at its depot stop, no rebalancing needed")
            return None
        
        return vehicle.depot_stop
    
    async def create_rebalancing_route(
        self,
        vehicle: Vehicle,
        target_stop: Stop,
        current_time: datetime
    ) -> Route:
        """
        Create a route for rebalancing a vehicle to its depot stop.
        
        Args:
            vehicle: The vehicle to rebalance
            target_stop: The target stop (depot)
            current_time: Current simulation time
            
        Returns:
            Route object for the rebalancing journey
        """
        # Use the route service to create a route from current location to depot
        route_id = f"rebalance_{vehicle.id}_{current_time.strftime('%Y%m%d%H%M%S')}"
        
        # Create a simple route with just the destination stop
        route = await self.route_service.create_route(
            route_id=route_id,
            vehicle=vehicle,
            stops=[target_stop],
            start_time=current_time,
            route_type=RouteType.REBALANCING
        )
        
        logger.info(f"Created rebalancing route {route_id} for vehicle {vehicle.id} to depot {target_stop.id}")
        return route
    
    def should_rebalance(
        self,
        vehicle: Vehicle,
        current_time: datetime,
        system_state: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Determine if a vehicle should be rebalanced based on idle time.
        
        Args:
            vehicle: The vehicle to potentially rebalance
            current_time: Current simulation time
            system_state: Current system state (not used in naive implementation)
            
        Returns:
            Boolean indicating whether the vehicle should be rebalanced
        """
        # Only rebalance vehicles that are idle
        if vehicle.current_state.status != VehicleStatus.IDLE:
            return False
        
        # Don't rebalance if no depot is assigned
        if not vehicle.depot_stop:
            return False
        
        # Don't rebalance if already at depot
        if (vehicle.current_state.current_stop_id and 
            vehicle.current_state.current_stop_id == vehicle.depot_stop.id):
            return False
        
        # Check if vehicle has been idle for longer than the threshold
        last_updated = vehicle.current_state.last_updated
        idle_duration = (current_time - last_updated).total_seconds() / 60.0  # Convert to minutes
        
        return idle_duration >= self.config.idle_threshold_minutes
