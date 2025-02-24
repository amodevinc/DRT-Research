from typing import List, Dict, Any, Optional, Tuple, Set
import numpy as np
from math import cos, radians, ceil
from datetime import timedelta
import asyncio
from sklearn.cluster import DBSCAN
import pandas as pd

from drt_sim.models.stop import Stop, StopStatus, StopType
from drt_sim.models.request import Request
from drt_sim.models.location import Location
from drt_sim.algorithms.base_interfaces.stop_selector_base import StopSelector
from drt_sim.models.vehicle import Vehicle
from drt_sim.core.state.manager import StateManager
from drt_sim.core.monitoring.visualization.manager import VisualizationManager

def latlon_to_meters(lat: float, lon: float, ref_lat: float = 0.0, ref_lon: float = 0.0) -> Tuple[float, float]:
    # Equirectangular projection (acceptable for small areas)
    meters_per_deg_lat = 111320
    meters_per_deg_lon = 111320 * cos(radians(ref_lat))
    x = (lon - ref_lon) * meters_per_deg_lon
    y = (lat - ref_lat) * meters_per_deg_lat
    return x, y

class CoverageBasedStopSelector(StopSelector):
    """
    A stop selector that blends immediate demand, historical demand, and (in future)
    predictive analysis to identify candidate locations for virtual stops.
    
    Configurable parameters (all can be set via the provided config):
      - immediate_demand_window: timedelta for recent requests (e.g. 30 minutes)
      - historical_demand_window: timedelta for historical patterns (e.g. 24 hours)
      - recent_weight: weight of immediate demand in the combined score (0-1)
      - historical_weight: weight of historical demand in the combined score (0-1)
      - coverage_radius: radius (in meters) within which demand is "close"
      - min_requests_for_cluster: minimum requests needed to form a cluster
      - min_stop_spacing: minimum distance between stops (meters)
      - max_walking_distance: maximum distance a passenger should walk (meters)
      - hourly_update_min_demand: minimum demand for creating a new stop in hourly updates
      - hourly_update_max_stops: maximum number of new stops to add during hourly updates
    
    Additional ML-based prediction can later be integrated via _predict_future_demand.
    """
    def __init__(self, sim_context, config, network_manager, state_manager: StateManager,
                 visualization_manager: Optional[VisualizationManager] = None):
        super().__init__(sim_context, config, network_manager)
        self.state_manager = state_manager
        self.visualization_manager = visualization_manager

        # Load candidate stops from CSV if provided
        self.candidate_stop_locations: List[Location] = []
        if self.config.candidate_stops_file:
            try:
                stops_df = pd.read_csv(self.config.candidate_stops_file)
                self.candidate_stop_locations = [
                    Location(lat=float(row['y']), lon=float(row['x']))
                    for _, row in stops_df.iterrows()
                ]
            except Exception as e:
                raise ValueError(f"Failed to load candidate stops from {self.config.candidate_stops_file}: {str(e)}")

        # Demand time windows: immediate and historical.
        self.immediate_window = timedelta(
            minutes=self.config.custom_params.get('immediate_demand_window_minutes', 30)
        )
        self.historical_window = timedelta(
            hours=self.config.custom_params.get('historical_demand_window_hours', 24)
        )
        # Weights to combine immediate and historical demand scores.
        self.recent_weight = self.config.custom_params.get('recent_weight', 0.6)
        self.historical_weight = self.config.custom_params.get('historical_weight', 0.4)
        self.min_requests_for_cluster = self.config.custom_params.get('min_requests_for_cluster', 3)

        self.max_walking_distance = self.config.max_walking_distance
        self.min_stop_spacing = self.config.min_stop_spacing
        self.coverage_radius = self.config.coverage_radius  # in meters; must be provided
        if not self.coverage_radius:
            raise ValueError("coverage_radius must be specified in config for CoverageBasedStopSelector")

        # Parameters for hourly stop selection
        self.hourly_update_min_demand = self.config.custom_params.get('hourly_update_min_demand', 2)
        self.hourly_update_max_stops = self.config.custom_params.get('hourly_update_max_stops', 25)
        self.last_stop_selection_time = None
        self.virtual_stops_by_area = {}  # Dictionary to track stops by area
        
        # Reference for projections (could be center of service area)
        self.ref_lat = 0.0
        self.ref_lon = 0.0

    def _project_location(self, location: Location) -> Tuple[float, float]:
        return latlon_to_meters(location.lat, location.lon, self.ref_lat, self.ref_lon)

    async def select_stops(self) -> List[Stop]:
        """
        Analyze demand patterns and existing stops to identify coverage gaps.
        Called hourly to ensure proper stop coverage across the service area.
        
        Returns:
            List[Stop]: New virtual stops to fill coverage gaps
        """
        current_time = self.sim_context.current_time
        if self.last_stop_selection_time is not None:
            time_since_last_selection = current_time - self.last_stop_selection_time
            if time_since_last_selection < timedelta(hours=1):
                return []  # Only run hourly
        
        self.last_stop_selection_time = current_time
        
        # Get demand patterns for the last hour
        hourly_demand_window = timedelta(hours=1)
        hourly_demand = self.state_manager.request_worker.get_recent_requests(
            hourly_demand_window, current_time
        )
        
        # Get existing active stops 
        existing_stops = self.state_manager.stop_worker.get_active_stops()
        existing_stop_locations = [stop.location for stop in existing_stops]
        
        # Get service area from network manager
        service_area = None
        if hasattr(self.network_manager, 'service_area'):
            service_area = self.network_manager.service_area
        
        # Analyze hourly demand for hot spots
        demand_clusters = self._cluster_demand_points(hourly_demand)
        
        # Filter out clusters that already have stops nearby
        new_stop_locations = []
        for cluster_center in demand_clusters:
            # Skip if any existing stop is within coverage radius
            if any(self._calculate_distance(
                Location(lat=cluster_center[0], lon=cluster_center[1]), 
                stop_loc) < self.coverage_radius 
                   for stop_loc in existing_stop_locations):
                continue
            
            # If we have a service area, check if cluster is within it
            if service_area is not None:
                from shapely.geometry import Point
                point = Point(cluster_center[1], cluster_center[0])  # lon, lat order for Shapely
                if not service_area.contains(point):
                    continue
            
            # Find accessible location near cluster center
            candidate = Location(lat=cluster_center[0], lon=cluster_center[1])
            candidates = await self._generate_candidates(candidate, 5)
            
            if candidates:
                # Score candidates and select best
                scores = []
                for loc in candidates:
                    vehicle_score = await self._calculate_vehicle_accessibility(
                        loc, self.state_manager.vehicle_worker.get_available_vehicles()
                    )
                    demand_score = self._calculate_demand_density(loc, hourly_demand)
                    scores.append(vehicle_score * 0.4 + demand_score * 0.6)
                
                if scores:
                    best_candidate = candidates[scores.index(max(scores))]
                    new_stop_locations.append(best_candidate)
                    
        # Limit the number of new stops per hourly update
        if len(new_stop_locations) > self.hourly_update_max_stops:
            new_stop_locations = new_stop_locations[:self.hourly_update_max_stops]
            
        # Create new virtual stops
        new_stops = []
        for i, location in enumerate(new_stop_locations):
            new_stop = Stop(
                location=location,
                type=StopType.VIRTUAL,
                status=StopStatus.ACTIVE,
                capacity=self.config.default_virtual_stop_capacity,
                current_load=0,
                metadata={
                    'virtual': True,
                    'creation_time': current_time,
                    'selection_method': 'hourly_coverage_update',
                    'area_id': self._get_area_id(location)
                }
            )
            
            # Track this stop by area
            area_id = self._get_area_id(location)
            if area_id not in self.virtual_stops_by_area:
                self.virtual_stops_by_area[area_id] = []
            self.virtual_stops_by_area[area_id].append(new_stop)
            
            new_stops.append(new_stop)
            
        # Visualize new stops if visualization is enabled
        if self.visualization_manager and new_stops:
            self.visualization_manager.add_frame(
                component_id='stop_handler',
                module_id='stop_selector',
                frame_type='hourly_virtual_stops',
                data={
                    'new_stops': [[s.location.lat, s.location.lon] for s in new_stops],
                    'existing_stops': [[s.location.lat, s.location.lon] for s in existing_stops],
                    'demand_clusters': [[c[0], c[1]] for c in demand_clusters]
                },
                metadata={'time': current_time.isoformat()},
                description=f"Hourly virtual stop selection at {current_time.isoformat()}"
            )
            
        return new_stops

    def _cluster_demand_points(self, requests: List[Request]) -> List[Tuple[float, float]]:
        """Cluster demand points to find hot spots"""
        if not requests or len(requests) < self.hourly_update_min_demand:
            return []
            
        # Extract points from requests
        points = []
        for req in requests:
            points.append((req.origin.lat, req.origin.lon))
            points.append((req.destination.lat, req.destination.lon))
            
        if not points:
            return []
            
        # Project points to meters for clustering
        projected = np.array([self._project_location(Location(lat, lon)) for lat, lon in points])
        
        # Run DBSCAN clustering
        clustering = DBSCAN(
            eps=self.coverage_radius / 2,  # Half the coverage radius for density
            min_samples=self.hourly_update_min_demand
        ).fit(projected)
        
        labels = clustering.labels_
        clusters = []
        
        for label in set(labels):
            if label == -1:  # Skip noise points
                continue
            mask = labels == label
            if sum(mask) >= self.hourly_update_min_demand:
                center = projected[mask].mean(axis=0)
                # Convert back to lat/lon
                meters_per_deg_lat = 111320
                meters_per_deg_lon = 111320 * cos(radians(self.ref_lat))
                lat = center[1] / meters_per_deg_lat + self.ref_lat
                lon = center[0] / meters_per_deg_lon + self.ref_lon
                clusters.append((lat, lon))
                
        return clusters

    def _calculate_demand_density(self, location: Location, requests: List[Request]) -> float:
        """Calculate demand density around a location based on recent requests"""
        count = 0
        for req in requests:
            if (self._calculate_distance(location, req.origin) <= self.coverage_radius or
                self._calculate_distance(location, req.destination) <= self.coverage_radius):
                count += 1
                
        # Normalize by theoretical maximum (all requests within radius)
        max_count = len(requests) * 2  # Both origins and destinations
        if max_count == 0:
            return 0.0
        return min(1.0, count / max_count)
    
    async def create_virtual_stops_for_request(self, request: Request) -> Tuple[Stop, Stop]:
        """
        Create optimally placed virtual stops for the given request using predefined candidates.
        """
        # Find existing stops in the vicinity that could be reused
        nearby_origin_stops = self._find_nearby_stops(request.origin)
        nearby_dest_stops = self._find_nearby_stops(request.destination)

        # Generate candidates for visualization
        origin_candidates = await self._generate_candidates(request.origin, num_candidates=5)
        dest_candidates = await self._generate_candidates(request.destination, num_candidates=5)
        
        # Visualize candidate generation
        if self.visualization_manager:
            self.visualization_manager.add_frame(
                component_id='stop_handler',
                module_id='stop_selector',
                frame_type='virtual_stop_candidates',
                data={
                    'request_origin': [request.origin.lat, request.origin.lon],
                    'request_destination': [request.destination.lat, request.destination.lon],
                    'origin_candidates': [[c.lat, c.lon] for c in origin_candidates],
                    'destination_candidates': [[c.lat, c.lon] for c in dest_candidates],
                    'nearby_origin_stops': [[s.location.lat, s.location.lon] for s in nearby_origin_stops],
                    'nearby_dest_stops': [[s.location.lat, s.location.lon] for s in nearby_dest_stops],
                    'coverage_radius': self.coverage_radius,
                    'min_stop_spacing': self.min_stop_spacing
                },
                metadata={'request_id': request.id},
                description=f"Generated candidate stops for request {request.id}"
            )
        
        # First try to reuse nearby stops if they're within acceptable walking distance
        origin_stop = None
        origin_min_distance = float('inf')
        dest_stop = None
        dest_min_distance = float('inf')
        
        # Check origin stops
        origin_distances = await asyncio.gather(*[
            self._calculate_network_distance(stop.location, request.origin, network_type='walk')
            for stop in nearby_origin_stops
        ])
        for stop, distance in zip(nearby_origin_stops, origin_distances):
            if distance <= self.max_walking_distance and distance < origin_min_distance:
                origin_stop = stop
                origin_min_distance = distance
                
        # Check destination stops  
        dest_distances = await asyncio.gather(*[
            self._calculate_network_distance(stop.location, request.destination, network_type='walk')
            for stop in nearby_dest_stops
        ])
        for stop, distance in zip(nearby_dest_stops, dest_distances):
            if distance <= self.max_walking_distance and distance < dest_min_distance:
                dest_stop = stop
                dest_min_distance = distance
        
        # If no suitable existing stops, find best candidates from predefined locations
        if origin_stop is None:
            origin_location, origin_min_distance = await self._select_optimal_virtual_location(
                request.origin, origin_candidates, is_pickup=True
            )
            if origin_min_distance > self.max_walking_distance:
                return (None, float('inf')), (None, float('inf'))
            origin_stop = Stop(
                location=origin_location,
                type=StopType.VIRTUAL,
                status=StopStatus.ACTIVE,
                capacity=self.config.default_virtual_stop_capacity,
                current_load=0,
                metadata={
                    'virtual': True,
                    'request_id': request.id,
                    'creation_time': self.sim_context.current_time,
                    'type': 'origin',
                    'selection_method': 'coverage_based',
                    'area_id': self._get_area_id(origin_location)
                }
            )
            
            # Track this stop by area
            area_id = self._get_area_id(origin_location)
            if area_id not in self.virtual_stops_by_area:
                self.virtual_stops_by_area[area_id] = []
            self.virtual_stops_by_area[area_id].append(origin_stop)
            
        if dest_stop is None:
            dest_location, dest_min_distance = await self._select_optimal_virtual_location(
                request.destination, dest_candidates, is_pickup=False
            )
            if dest_min_distance > self.max_walking_distance:
                return (origin_stop, origin_min_distance), (None, float('inf'))
            dest_stop = Stop(
                location=dest_location,
                type=StopType.VIRTUAL,
                status=StopStatus.ACTIVE,
                capacity=self.config.default_virtual_stop_capacity,
                current_load=0,
                metadata={
                    'virtual': True,
                    'request_id': request.id,
                    'creation_time': self.sim_context.current_time,
                    'type': 'destination',
                    'selection_method': 'coverage_based',
                    'area_id': self._get_area_id(dest_location)
                }
            )
            
            # Track this stop by area
            area_id = self._get_area_id(dest_location)
            if area_id not in self.virtual_stops_by_area:
                self.virtual_stops_by_area[area_id] = []
            self.virtual_stops_by_area[area_id].append(dest_stop)

        return (origin_stop, origin_min_distance), (dest_stop, dest_min_distance)

    def _find_nearby_stops(self, location: Location) -> List[Stop]:
        """Find existing virtual stops near a location"""
        nearby_stops = []
        all_stops = self.state_manager.stop_worker.get_active_stops()
        for stop in all_stops:
            if stop.type == StopType.VIRTUAL:
                distance = self._calculate_distance(location, stop.location)
                if distance <= self.coverage_radius:
                    nearby_stops.append(stop)
        return nearby_stops

    def _analyze_demand_patterns(self) -> Dict[str, Any]:
        """
        Analyze demand by combining immediate requests, historical requests, and future predictions.
        Returns a dictionary with:
         - 'demand_clusters': List of (lat, lon) cluster centers (weighted combination)
         - 'area_densities': Mapping of area IDs to combined demand density.
        """
        # Get current simulation time
        current_time = self.sim_context.current_time
        
        # Retrieve immediate and historical requests from state manager
        immediate_requests = self.state_manager.request_worker.get_recent_requests(
            self.immediate_window, 
            current_time
        )
        historical_requests = self.state_manager.request_worker.get_historical_requests(
            self.historical_window,
            current_time
        )

        # For now, our future demand prediction is a stub.
        predicted_requests = self._predict_future_demand()

        # Weight and combine the requests:
        def extract_points(requests: List[Request]) -> List[Tuple[float, float]]:
            points = []
            for req in requests:
                # Include both origin and destination points.
                points.append((req.origin.lat, req.origin.lon))
                points.append((req.destination.lat, req.destination.lon))
            return points

        immediate_points = extract_points(immediate_requests)
        historical_points = extract_points(historical_requests)
        predicted_points = extract_points(predicted_requests)

        # Combine points with weights by simply replicating points by weight factor.
        weighted_points = (immediate_points * int(self.recent_weight * 10) +
                           historical_points * int(self.historical_weight * 10) +
                           predicted_points)  # prediction weight can be tuned later

        if not weighted_points:
            combined_clusters = []
            area_densities = {}
        else:
            # Project points to meters.
            projected = np.array([self._project_location(Location(lat, lon)) for lat, lon in weighted_points])
            clustering = DBSCAN(eps=self.coverage_radius, min_samples=self.min_requests_for_cluster).fit(projected)
            labels = clustering.labels_
            combined_clusters = []
            for label in set(labels):
                if label == -1:
                    continue
                mask = labels == label
                center = projected[mask].mean(axis=0)
                # Convert back to lat/lon (approximation)
                meters_per_deg_lat = 111320
                meters_per_deg_lon = 111320 * cos(radians(self.ref_lat))
                lat = center[1] / meters_per_deg_lat + self.ref_lat
                lon = center[0] / meters_per_deg_lon + self.ref_lon
                combined_clusters.append((lat, lon))
            # Also compute area densities using a simple grid.
            area_densities = {}
            grid_size = 0.01  # approx. 1km cells
            for lat, lon in weighted_points:
                area_id = f"area_{int(lat/grid_size)}_{int(lon/grid_size)}"
                area_densities[area_id] = area_densities.get(area_id, 0) + 1

        # Optionally visualize the demand analysis.
        if self.visualization_manager:
            self.visualization_manager.add_frame(
                component_id='stop_handler',
                module_id='stop_selector',
                frame_type='demand_analysis',
                data={
                    'weighted_points': weighted_points,
                    'clusters': combined_clusters,
                    'area_densities': area_densities,
                    'num_immediate': len(immediate_points),
                    'num_historical': len(historical_points),
                    'num_predicted': len(predicted_points)
                },
                metadata={'coverage_radius': self.coverage_radius},
                description="Combined demand analysis using immediate, historical, and predicted requests."
            )

        return {
            'demand_clusters': combined_clusters,
            'area_densities': area_densities
        }

    def _predict_future_demand(self) -> List[Request]:
        """
        Stub for future ML-based demand prediction.
        For now, simply returns an empty list.
        """
        # In the future, integrate ML model here.
        return []

    async def _generate_candidates(self, center: Location, num_candidates: int = 5) -> List[Location]:
        """
        Generate candidate locations around a given center using predefined candidates from CSV.
        
        Args:
            center: The center location to generate candidates around
            num_candidates: Number of candidates to return
            
        Returns:
            List of candidate locations closest to the center point
        """
        if not self.candidate_stop_locations:
            return [center]  # If no candidates loaded, return the center point
            
        # Calculate distances to all candidates
        distances = [(stop, self._calculate_distance(center, stop)) 
                    for stop in self.candidate_stop_locations]
        
        # Sort by distance and return the closest num_candidates
        sorted_candidates = sorted(distances, key=lambda x: x[1])
        return [candidate[0] for candidate in sorted_candidates[:num_candidates]]

    async def _select_optimal_virtual_location(self, request_point: Location,
                                                 candidates: List[Location],
                                                 is_pickup: bool) -> Tuple[Location, float]:
        """
        Given a set of candidate locations, select the one with the highest combined score.
        """
        if not candidates:
            return request_point, 0.0  # fallback to original location

        scores = await asyncio.gather(*[
            self._score_candidate(candidate, request_point, is_pickup)
            for candidate in candidates
        ])
        best_candidate = max(zip(candidates, scores), key=lambda x: x[1])[0]
        best_candidate_distance = await self._calculate_network_distance(best_candidate, request_point, network_type='walk')
        return best_candidate, best_candidate_distance

    async def _score_candidate(self, candidate: Location, request_point: Location,
                               is_pickup: bool) -> float:
        """
        Score a candidate location based on:
          - Walking distance (closer is better)
          - Vehicle accessibility (using network distance)
          - Demand density (proximity to other requests)
          - Safety (stub for future integration)
        """
        # 1. Walking distance (normalized)
        walking_distance = await self._calculate_network_distance(candidate, request_point, network_type='walk')
        walk_score = max(0.0, 1.0 - walking_distance / self.max_walking_distance)

        # 2. Vehicle accessibility
        vehicles = self.state_manager.vehicle_worker.get_available_vehicles()
        vehicle_score = await self._calculate_vehicle_accessibility(candidate, vehicles)

        # 3. Demand density
        hourly_demand = self.state_manager.request_worker.get_recent_requests(
            timedelta(hours=1), self.sim_context.current_time
        )
        demand_score = self._calculate_demand_density(candidate, hourly_demand)
        
        # 4. Safety (stub for future implementation)
        safety_score = 1.0  # Placeholder

        # Different weights for pickup vs dropoff
        if is_pickup:
            # For pickup, prioritize vehicle accessibility and walking distance
            weights = {
                'walk': 0.4,
                'vehicle': 0.4,
                'demand': 0.1,
                'safety': 0.1
            }
        else:
            # For dropoff, prioritize walking distance and safety
            weights = {
                'walk': 0.5,
                'vehicle': 0.2,
                'demand': 0.1,
                'safety': 0.2
            }
            
        total_score = (weights['walk'] * walk_score +
                       weights['vehicle'] * vehicle_score +
                       weights['demand'] * demand_score +
                       weights['safety'] * safety_score)
        return total_score

    def _calculate_demand_score(self, location: Location, demand_patterns: Dict[str, Any]) -> float:
        """
        Calculate a combined demand score based on:
          - Proximity to demand clusters
          - Area density (from grid counts)
        """
        cluster_scores = []
        for cluster_lat, cluster_lon in demand_patterns.get('demand_clusters', []):
            cluster_loc = Location(cluster_lat, cluster_lon)
            distance = self._calculate_distance(location, cluster_loc)
            if distance <= self.coverage_radius:
                cluster_scores.append(1.0 - distance / self.coverage_radius)
        cluster_score = max(cluster_scores) if cluster_scores else 0.0

        area_id = self._get_area_id(location)
        area_demand = demand_patterns.get('area_densities', {}).get(area_id, 0)
        max_density = max(demand_patterns.get('area_densities', {}).values(), default=1)
        area_score = area_demand / max_density

        # Combine the two with a configurable cluster weight.
        cluster_weight = self.config.custom_params.get('cluster_weight', 0.7)
        return cluster_weight * cluster_score + (1 - cluster_weight) * area_score

    async def _calculate_vehicle_accessibility(self, location: Location,
                                                vehicles: List[Vehicle]) -> float:
        """
        Calculate accessibility for vehicles by determining the network distance to the nearest vehicle.
        """
        if not vehicles:
            return 0.5
        distance_tasks = [
            self._calculate_network_distance(location, vehicle.current_state.current_location, network_type='drive')
            for vehicle in vehicles
        ]
        distances = await asyncio.gather(*distance_tasks)
        min_distance = min(distances)
        max_vehicle_distance = self.config.custom_params.get('max_vehicle_distance', 1500)
        return max(0.0, 1.0 - min_distance / max_vehicle_distance)

    async def _calculate_network_distance(self, loc1: Location, loc2: Location,
                                          network_type: str = 'drive') -> float:
        """
        Asynchronously compute network distance between two locations.
        Here we call the network manager's distance function.
        """
        return await self.network_manager.calculate_distance(loc1, loc2, network_type=network_type)

    def _calculate_distance(self, loc1: Location, loc2: Location) -> float:
        """
        Calculate Euclidean distance (in meters) between two locations.
        """
        x1, y1 = self._project_location(loc1)
        x2, y2 = self._project_location(loc2)
        return np.hypot(x2 - x1, y2 - y1)

    def _get_area_id(self, location: Location) -> str:
        """
        Map a location to an area ID using a grid (e.g. 1km cells).
        """
        grid_size = 0.01  # roughly 1km
        lat_grid = int(location.lat / grid_size)
        lon_grid = int(location.lon / grid_size)
        return f"area_{lat_grid}_{lon_grid}"