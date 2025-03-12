from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime, timedelta
from collections import defaultdict
import numpy as np

from drt_sim.algorithms.base_interfaces.rebalancing_base import RebalancingAlgorithm, RebalancingConfig, RebalancingStrategy
from drt_sim.models.vehicle import Vehicle, VehicleStatus
from drt_sim.models.stop import Stop
from drt_sim.models.route import Route, RouteStatus, RouteType
from drt_sim.models.location import Location
from drt_sim.network.manager import NetworkManager
from drt_sim.core.state.manager import StateManager
from drt_sim.core.simulation.context import SimulationContext
from drt_sim.core.services.route_service import RouteService

import logging
logger = logging.getLogger(__name__)

class LoadBalancingConfig(RebalancingConfig):
    """Configuration for load balancing rebalancing algorithm."""
    
    def __init__(
        self,
        strategy: RebalancingStrategy = RebalancingStrategy.ZONE_BASED,
        idle_threshold_minutes: float = 5.0,
        max_rebalance_distance_km: Optional[float] = 10.0,
        min_battery_level: Optional[float] = 20.0,
        rebalance_trigger_occupancy_threshold: float = 0.3,
        weights: Dict[str, float] = None,
        grid_size_km: float = 2.0,  # Size of each grid cell in kilometers
        target_balance_threshold: float = 0.2,  # Threshold for determining imbalance (0-1)
        min_vehicles_per_cell: int = 1,  # Minimum vehicles per cell
        max_vehicles_per_cell: Optional[int] = None,  # Maximum vehicles per cell
        **kwargs
    ):
        """
        Initialize load balancing configuration.
        
        Args:
            grid_size_km: Size of each grid cell in kilometers
            target_balance_threshold: Threshold for determining imbalance (0-1)
            min_vehicles_per_cell: Minimum vehicles per cell
            max_vehicles_per_cell: Maximum vehicles per cell
            **kwargs: Additional parameters passed to RebalancingConfig
        """
        weights = weights or {"balance": 0.7, "distance": 0.3}
        super().__init__(
            strategy=strategy,
            idle_threshold_minutes=idle_threshold_minutes,
            max_rebalance_distance_km=max_rebalance_distance_km,
            min_battery_level=min_battery_level,
            rebalance_trigger_occupancy_threshold=rebalance_trigger_occupancy_threshold,
            weights=weights,
            **kwargs
        )
        self.grid_size_km = grid_size_km
        self.target_balance_threshold = target_balance_threshold
        self.min_vehicles_per_cell = min_vehicles_per_cell
        self.max_vehicles_per_cell = max_vehicles_per_cell


class LoadBalancingRebalancingAlgorithm(RebalancingAlgorithm):
    """
    A load balancing rebalancing algorithm that distributes idle vehicles evenly across the service area.
    
    This algorithm divides the service area into grid cells and attempts to maintain an even
    distribution of vehicles across all cells, moving vehicles from over-supplied areas to
    under-supplied areas.
    """
    
    def __init__(
        self,
        sim_context: SimulationContext,
        config: Optional[LoadBalancingConfig],
        network_manager: NetworkManager,
        state_manager: StateManager,
        route_service: RouteService
    ):
        """Initialize the load balancing rebalancing algorithm."""
        if not config:
            config = LoadBalancingConfig()
        super().__init__(sim_context, config, network_manager, state_manager, route_service)
        
        # Store the grid cells and their associated stops
        self.grid_cells = {}  # Maps cell_id to list of stops in that cell
        self.stop_to_cell = {}  # Maps stop_id to cell_id
        
        # Initialize the grid cells
        self._initialize_grid_cells()
        
        # Cache for vehicle distribution
        self.vehicle_distribution_cache = None
        self.last_distribution_update = None
        self.distribution_cache_ttl = timedelta(minutes=5)  # Update distribution every 5 minutes
    
    def _initialize_grid_cells(self):
        """
        Initialize the grid cells based on the service area and available stops.
        This divides the service area into a grid and assigns stops to cells.
        """
        # Get all stops from the state manager
        all_stops = self._get_all_stops()
        
        if not all_stops:
            logger.warning("No stops available for grid cell initialization")
            return
        
        # Determine the bounds of the service area
        min_lat, max_lat, min_lon, max_lon = self._get_service_area_bounds(all_stops)
        
        # Calculate the approximate size of a grid cell in degrees
        # This is a simplification and assumes a flat earth for small areas
        # For more accuracy, a proper geospatial library should be used
        km_per_degree_lat = 111.0  # Approximate km per degree of latitude
        km_per_degree_lon = 111.0 * np.cos(np.radians((min_lat + max_lat) / 2))  # Approximate km per degree of longitude
        
        cell_size_lat = self.config.grid_size_km / km_per_degree_lat
        cell_size_lon = self.config.grid_size_km / km_per_degree_lon
        
        # Create grid cells
        for stop in all_stops:
            cell_row = int((stop.location.latitude - min_lat) / cell_size_lat)
            cell_col = int((stop.location.longitude - min_lon) / cell_size_lon)
            cell_id = f"{cell_row}_{cell_col}"
            
            if cell_id not in self.grid_cells:
                self.grid_cells[cell_id] = []
            
            self.grid_cells[cell_id].append(stop)
            self.stop_to_cell[stop.id] = cell_id
        
        logger.info(f"Initialized {len(self.grid_cells)} grid cells for load balancing rebalancing")
    
    def _get_all_stops(self) -> List[Stop]:
        """
        Get all stops from the state manager.
        
        Returns:
            List of all stops in the system
        """
        # This is a placeholder - implement based on your state manager's API
        # In a real implementation, you would get all stops from the state manager
        stop_state_worker = self.state_manager.get_worker("stop")
        if stop_state_worker:
            return stop_state_worker.get_all_stops()
        return []
    
    def _get_service_area_bounds(self, stops: List[Stop]) -> Tuple[float, float, float, float]:
        """
        Get the bounds of the service area based on stop locations.
        
        Args:
            stops: List of stops
            
        Returns:
            Tuple of (min_latitude, max_latitude, min_longitude, max_longitude)
        """
        if not stops:
            # Default to a small area if no stops are available
            return (0.0, 0.1, 0.0, 0.1)
        
        min_lat = min(stop.location.latitude for stop in stops)
        max_lat = max(stop.location.latitude for stop in stops)
        min_lon = min(stop.location.longitude for stop in stops)
        max_lon = max(stop.location.longitude for stop in stops)
        
        return (min_lat, max_lat, min_lon, max_lon)
    
    def _get_current_vehicle_distribution(self, current_time: datetime) -> Dict[str, List[Vehicle]]:
        """
        Get the current distribution of vehicles across grid cells.
        
        Args:
            current_time: Current simulation time
            
        Returns:
            Dictionary mapping cell_id to list of vehicles in that cell
        """
        # Check if we can use the cached distribution
        if (self.vehicle_distribution_cache is not None and 
            self.last_distribution_update is not None and
            current_time - self.last_distribution_update < self.distribution_cache_ttl):
            return self.vehicle_distribution_cache
        
        # Get all vehicles from the state manager
        all_vehicles = self._get_all_vehicles()
        
        # Initialize distribution
        distribution = defaultdict(list)
        
        # Assign vehicles to cells based on their current location
        for vehicle in all_vehicles:
            cell_id = self._get_cell_for_location(vehicle.current_state.current_location)
            if cell_id:
                distribution[cell_id].append(vehicle)
        
        # Update cache
        self.vehicle_distribution_cache = distribution
        self.last_distribution_update = current_time
        
        return distribution
    
    def _get_all_vehicles(self) -> List[Vehicle]:
        """
        Get all vehicles from the state manager.
        
        Returns:
            List of all vehicles in the system
        """
        # This is a placeholder - implement based on your state manager's API
        vehicle_state_worker = self.state_manager.get_worker("vehicle")
        if vehicle_state_worker:
            return vehicle_state_worker.get_all_vehicles()
        return []
    
    def _get_cell_for_location(self, location: Location) -> Optional[str]:
        """
        Get the grid cell ID for a given location.
        
        Args:
            location: Location to find cell for
            
        Returns:
            Cell ID or None if location is outside the grid
        """
        # This is a simplified implementation
        # In a real system, you would use the same grid calculation as in _initialize_grid_cells
        
        # Find the nearest stop and use its cell
        nearest_stop = self._find_nearest_stop(location)
        if nearest_stop and nearest_stop.id in self.stop_to_cell:
            return self.stop_to_cell[nearest_stop.id]
        
        return None
    
    def _find_nearest_stop(self, location: Location) -> Optional[Stop]:
        """
        Find the nearest stop to a given location.
        
        Args:
            location: Location to find nearest stop for
            
        Returns:
            Nearest stop or None if no stops are available
        """
        # This is a placeholder - implement based on your network manager's API
        all_stops = self._get_all_stops()
        if not all_stops:
            return None
        
        # Find the stop with the minimum distance to the location
        nearest_stop = min(
            all_stops, 
            key=lambda stop: self.network_manager.calculate_distance(location, stop.location)
        )
        
        return nearest_stop
    
    def _calculate_target_distribution(self, current_distribution: Dict[str, List[Vehicle]]) -> Dict[str, int]:
        """
        Calculate the target number of vehicles for each cell based on even distribution.
        
        Args:
            current_distribution: Current distribution of vehicles
            
        Returns:
            Dictionary mapping cell_id to target number of vehicles
        """
        total_vehicles = sum(len(vehicles) for vehicles in current_distribution.values())
        total_cells = len(self.grid_cells)
        
        if total_cells == 0:
            return {}
        
        # Calculate the target number of vehicles per cell
        base_target = total_vehicles // total_cells
        remainder = total_vehicles % total_cells
        
        # Distribute vehicles evenly, with any remainder going to the first cells
        target_distribution = {}
        for i, cell_id in enumerate(self.grid_cells.keys()):
            target = base_target + (1 if i < remainder else 0)
            
            # Apply min/max constraints
            if target < self.config.min_vehicles_per_cell:
                target = self.config.min_vehicles_per_cell
            
            if self.config.max_vehicles_per_cell is not None and target > self.config.max_vehicles_per_cell:
                target = self.config.max_vehicles_per_cell
                
            target_distribution[cell_id] = target
        
        return target_distribution
    
    def _identify_imbalanced_cells(
        self, 
        current_distribution: Dict[str, List[Vehicle]], 
        target_distribution: Dict[str, int]
    ) -> Tuple[List[str], List[str]]:
        """
        Identify cells with too many or too few vehicles.
        
        Args:
            current_distribution: Current distribution of vehicles
            target_distribution: Target distribution of vehicles
            
        Returns:
            Tuple of (oversupplied_cells, undersupplied_cells)
        """
        oversupplied_cells = []
        undersupplied_cells = []
        
        for cell_id, target in target_distribution.items():
            current = len(current_distribution.get(cell_id, []))
            
            # Calculate the imbalance ratio
            if target > 0:
                imbalance = abs(current - target) / target
            else:
                imbalance = 1.0 if current > 0 else 0.0
            
            # Check if the imbalance exceeds the threshold
            if imbalance > self.config.target_balance_threshold:
                if current > target:
                    oversupplied_cells.append(cell_id)
                elif current < target:
                    undersupplied_cells.append(cell_id)
        
        return oversupplied_cells, undersupplied_cells
    
    def _get_rebalancing_candidates(
        self, 
        oversupplied_cells: List[str], 
        current_distribution: Dict[str, List[Vehicle]],
        current_time: datetime
    ) -> List[Tuple[Vehicle, str]]:
        """
        Get vehicles that are candidates for rebalancing and their source cells.
        
        Args:
            oversupplied_cells: List of oversupplied cell IDs
            current_distribution: Current distribution of vehicles
            current_time: Current simulation time
            
        Returns:
            List of (vehicle, source_cell_id) tuples
        """
        candidates = []
        
        for cell_id in oversupplied_cells:
            vehicles = current_distribution.get(cell_id, [])
            
            # Filter for idle vehicles that meet rebalancing criteria
            for vehicle in vehicles:
                if self.should_rebalance(vehicle, current_time):
                    candidates.append((vehicle, cell_id))
        
        return candidates
    
    def _get_target_stop_for_cell(self, cell_id: str) -> Optional[Stop]:
        """
        Get a representative stop for a cell to use as a rebalancing target.
        
        Args:
            cell_id: Cell ID
            
        Returns:
            A stop in the cell or None if no stops are available
        """
        stops = self.grid_cells.get(cell_id, [])
        if not stops:
            return None
        
        # For simplicity, use the first stop in the cell
        # In a more sophisticated implementation, you might choose a central stop
        # or one with high historical demand
        return stops[0]
    
    def get_rebalancing_target(
        self,
        vehicle: Vehicle,
        current_time: datetime,
        available_stops: List[Stop],
        system_state: Optional[Dict[str, Any]] = None
    ) -> Optional[Stop]:
        """
        Get the rebalancing target for a vehicle based on load balancing.
        
        Args:
            vehicle: The vehicle to rebalance
            current_time: Current simulation time
            available_stops: List of available stops
            system_state: Current system state
            
        Returns:
            Target stop for rebalancing or None if no rebalancing needed
        """
        # Get current vehicle distribution
        current_distribution = self._get_current_vehicle_distribution(current_time)
        
        # Calculate target distribution
        target_distribution = self._calculate_target_distribution(current_distribution)
        
        # Identify imbalanced cells
        oversupplied_cells, undersupplied_cells = self._identify_imbalanced_cells(
            current_distribution, target_distribution
        )
        
        if not undersupplied_cells:
            logger.debug("No undersupplied cells found, no rebalancing needed")
            return None
        
        # Get the current cell for the vehicle
        current_cell = self._get_cell_for_location(vehicle.current_state.current_location)
        
        # If the vehicle is not in an oversupplied cell, no rebalancing needed
        if current_cell not in oversupplied_cells:
            return None
        
        # Find the best undersupplied cell to move to
        best_cell = None
        best_score = float('inf')
        
        for cell_id in undersupplied_cells:
            # Get a representative stop for the cell
            target_stop = self._get_target_stop_for_cell(cell_id)
            if not target_stop:
                continue
            
            # Calculate distance to the cell
            distance = self.network_manager.calculate_distance(
                vehicle.current_state.current_location, target_stop.location
            )
            
            # Skip if beyond max rebalance distance
            if (self.config.max_rebalance_distance_km is not None and 
                distance > self.config.max_rebalance_distance_km):
                continue
            
            # Calculate imbalance score (how undersupplied the cell is)
            current = len(current_distribution.get(cell_id, []))
            target = target_distribution.get(cell_id, 0)
            imbalance = (target - current) / max(1, target)
            
            # Calculate combined score (weighted sum of distance and imbalance)
            # Lower is better for distance, higher is better for imbalance
            distance_weight = self.config.weights.get("distance", 0.3)
            balance_weight = self.config.weights.get("balance", 0.7)
            
            # Normalize distance to 0-1 range using max_rebalance_distance
            normalized_distance = distance / (self.config.max_rebalance_distance_km or 10.0)
            
            # Combine scores (lower is better)
            score = distance_weight * normalized_distance - balance_weight * imbalance
            
            if score < best_score:
                best_score = score
                best_cell = cell_id
        
        if best_cell:
            return self._get_target_stop_for_cell(best_cell)
        
        return None
    
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
            target_stop: The target stop
            current_time: Current simulation time
            
        Returns:
            Route object for the rebalancing journey
        """
        # Use the route service to create a route from current location to target stop
        route_id = f"rebalance_lb_{vehicle.id}_{current_time.strftime('%Y%m%d%H%M%S')}"
        
        # Create a route with just the destination stop
        route = await self.route_service.create_route(
            route_id=route_id,
            vehicle=vehicle,
            stops=[target_stop],
            start_time=current_time,
            route_type=RouteType.REBALANCING
        )
        
        logger.info(f"Created load balancing rebalancing route {route_id} for vehicle {vehicle.id} to stop {target_stop.id}")
        return route
    
    def should_rebalance(
        self,
        vehicle: Vehicle,
        current_time: datetime,
        system_state: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Determine if a vehicle should be rebalanced based on idle time and other criteria.
        
        Args:
            vehicle: The vehicle to potentially rebalance
            current_time: Current simulation time
            system_state: Current system state
            
        Returns:
            Boolean indicating whether the vehicle should be rebalanced
        """
        # Only rebalance vehicles that are idle
        if vehicle.current_state.status != VehicleStatus.IDLE:
            return False
        
        # Check battery level for electric vehicles
        if (vehicle.type.value == "electric" and 
            vehicle.current_state.battery_level is not None and
            self.config.min_battery_level is not None and
            vehicle.current_state.battery_level < self.config.min_battery_level):
            return False
        
        # Check if vehicle has been idle for longer than the threshold
        last_updated = vehicle.current_state.last_updated
        idle_duration = (current_time - last_updated).total_seconds() / 60.0  # Convert to minutes
        
        return idle_duration >= self.config.idle_threshold_minutes 