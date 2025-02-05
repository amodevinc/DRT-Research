'''Under Construction'''
from typing import List, Dict, Any, Optional, Tuple, Set
import numpy as np
from math import ceil
from sklearn.cluster import DBSCAN

from drt_sim.models.stop import Stop, StopStatus, StopType
from drt_sim.models.request import Request
from drt_sim.models.location import Location
from drt_sim.algorithms.base_interfaces.stop_selector_base import (
    StopSelector,
)
from drt_sim.models.simulation import SimulationState, VehicleState
import asyncio

class CoverageBasedStopSelector(StopSelector):
    """
    Selects stops to maximize area coverage while minimizing walking distances.
    Uses DBSCAN clustering to identify demand-dense areas and ensures coverage.
    """

    def select_stops(self,
                    candidate_locations: List[Location],
                    demand_points: Optional[List[Location]] = None,
                    existing_stops: Optional[List[Stop]] = None,
                    constraints: Optional[Dict[str, Any]] = None) -> List[Stop]:
        """Select stops to maximize coverage of the service area."""
        if not candidate_locations:
            raise ValueError("No candidate locations provided")

        # Convert locations to numpy arrays for clustering
        points = np.array([[loc.lat, loc.lon] for loc in candidate_locations])
        
        # Determine DBSCAN parameters based on coverage radius
        eps = self.config.coverage_radius / 111000  # Convert meters to approximate degrees
        min_samples = 3  # Minimum points to form a cluster
        
        # Perform clustering
        clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(points)
        labels = clustering.labels_

        # Process each cluster and noise points
        selected_stops = []
        processed_areas = set()

        # Handle clustered points
        for cluster_id in set(labels):
            if cluster_id == -1:  # Skip noise points for now
                continue
                
            # Get points in this cluster
            cluster_mask = labels == cluster_id
            cluster_points = points[cluster_mask]
            cluster_candidates = [loc for loc, mask in zip(candidate_locations, cluster_mask) if mask]
            
            # Select stops for this cluster
            cluster_stops = self._select_cluster_stops(cluster_points, cluster_candidates)
            selected_stops.extend(cluster_stops)
            
            # Track covered area
            cluster_center = np.mean(cluster_points, axis=0)
            processed_areas.add(tuple(cluster_center))

        # Handle noise points - they might be important isolated locations
        noise_points = points[labels == -1]
        noise_candidates = [loc for loc, label in zip(candidate_locations, labels) if label == -1]
        
        if noise_points.size > 0:
            noise_stops = self._handle_noise_points(
                noise_points, 
                noise_candidates,
                processed_areas
            )
            selected_stops.extend(noise_stops)

        # Apply maximum stops constraint if specified
        if self.config.max_stops and len(selected_stops) > self.config.max_stops:
            selected_stops = self._prioritize_stops(selected_stops, self.config.max_stops)

        return selected_stops

    def update_stops(self,
                    current_stops: List[Stop],
                    demand_changes: Dict[str, float],
                    system_state: Optional[Dict[str, Any]] = None) -> Tuple[List[Stop], List[str]]:
        """Update stop network based on demand changes."""
        if not current_stops:
            return [], []

        modified_stops = []
        modified_ids = []

        # Create spatial index of current stops
        stop_locations = np.array([[s.location.lat, s.location.lon] for s in current_stops])
        
        # Process areas with significant demand changes
        for area_id, change_factor in demand_changes.items():
            area_info = self._get_area_info(area_id, system_state)
            if not area_info:
                continue

            area_center = np.array([area_info['latitude'], area_info['longitude']])
            
            # Find stops in this area
            distances = np.linalg.norm(stop_locations - area_center, axis=1)
            area_stops = [stop for stop, dist in zip(current_stops, distances)
                         if dist <= self.config.coverage_radius / 111000]

            if change_factor > 1.5 and not area_stops:
                # Significant increase in demand with no coverage - add a stop
                new_stop = self._create_stop(
                    Location(lat=area_info['latitude'], lon=area_info['longitude'])
                )
                modified_stops.append(new_stop)
                modified_ids.append(new_stop.id)
                
            elif change_factor < 0.5 and area_stops:
                # Significant decrease in demand - consider removing stops
                for stop in area_stops:
                    if self._can_remove_stop(stop, current_stops, system_state):
                        stop.status = StopStatus.INACTIVE
                        modified_stops.append(stop)
                        modified_ids.append(stop.id)

        # Update complete stop list
        updated_stops = [
            stop if stop.id not in modified_ids else
            next(s for s in modified_stops if s.id == stop.id)
            for stop in current_stops
        ] + [s for s in modified_stops if s.id not in {stop.id for stop in current_stops}]

        return updated_stops, modified_ids

    def _select_cluster_stops(self,
                            cluster_points: np.ndarray,
                            cluster_candidates: List[Location]) -> List[Stop]:
        """Select stops for a cluster of points."""
        # Calculate required number of stops based on area and coverage radius
        cluster_diameter = np.max([
            np.linalg.norm(p1 - p2) for p1 in cluster_points for p2 in cluster_points
        ])
        required_stops = max(1, ceil(cluster_diameter / (2 * self.config.coverage_radius / 111000)))
        
        # Use k-means to position stops if we need multiple
        if required_stops > 1:
            from sklearn.cluster import KMeans
            kmeans = KMeans(n_clusters=required_stops).fit(cluster_points)
            centers = kmeans.cluster_centers_
        else:
            centers = [np.mean(cluster_points, axis=0)]

        # Find closest candidate locations to computed centers
        stops = []
        for center in centers:
            distances = [
                (candidate, np.linalg.norm([center[0] - candidate.lat,
                                          center[1] - candidate.lon]))
                for candidate in cluster_candidates
            ]
            best_candidate = min(distances, key=lambda x: x[1])[0]
            
            # Create stop if it meets spacing requirements
            if not stops or all(
                self._calculate_distance(best_candidate, s.location) >= self.config.min_stop_spacing
                for s in stops
            ):
                stops.append(self._create_stop(best_candidate))

        return stops

    def _handle_noise_points(self,
                           noise_points: np.ndarray,
                           noise_candidates: List[Location],
                           processed_areas: Set[Tuple[float, float]]) -> List[Stop]:
        """Process noise points that might need coverage."""
        stops = []
        
        for point, candidate in zip(noise_points, noise_candidates):
            # Check if point is already covered by a processed area
            if any(
                np.linalg.norm(point - np.array(center)) <= self.config.coverage_radius / 111000
                for center in processed_areas
            ):
                continue
            
            # Check if this point should have a stop
            if self._should_place_noise_stop(point, candidate, noise_points):
                stops.append(self._create_stop(candidate))
                processed_areas.add(tuple(point))

        return stops

    def _should_place_noise_stop(self,
                                point: np.ndarray,
                                candidate: Location,
                                all_noise_points: np.ndarray) -> bool:
        """Determine if a noise point should have a stop."""
        # Count nearby points
        nearby_points = np.sum(
            np.linalg.norm(all_noise_points - point, axis=1) <= self.config.coverage_radius / 111000
        )
        
        # Place stop if:
        # 1. Point has some nearby neighbors (but not enough for a cluster)
        # 2. Point is isolated but might be an important location
        return nearby_points >= 2

    def _prioritize_stops(self, stops: List[Stop], max_stops: int) -> List[Stop]:
        """Prioritize stops when exceeding maximum allowed."""
        # Score each stop based on coverage and spacing
        scores = []
        for stop in stops:
            # Count other stops within minimum spacing
            crowding = sum(
                1 for other in stops
                if other != stop and
                self._calculate_distance(stop.location, other.location) < self.config.min_stop_spacing
            )
            
            # Higher score = better stop (less crowding)
            scores.append((stop, -crowding))
            
        # Sort by score and take top max_stops
        scores.sort(key=lambda x: x[1], reverse=True)
        return [stop for stop, _ in scores[:max_stops]]

    def _get_area_info(self, 
                      area_id: str, 
                      system_state: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """Get information about an area from system state."""
        if not system_state or 'areas' not in system_state:
            return None
        return system_state['areas'].get(area_id)

    def _can_remove_stop(self,
                        stop: Stop,
                        all_stops: List[Stop],
                        system_state: Optional[Dict[str, Any]]) -> bool:
        """Determine if a stop can be safely removed."""
        # Don't remove if it's the only stop in its area
        nearby_stops = [
            other for other in all_stops
            if other != stop and
            self._calculate_distance(stop.location, other.location) <= self.config.coverage_radius
        ]
        
        if not nearby_stops:
            return False
            
        # Check current usage if available
        if system_state and 'stop_usage' in system_state:
            usage = system_state['stop_usage'].get(stop.id, 0)
            if usage > 0:  # Don't remove stops with recent usage
                return False
                
        return True
    
    async def create_virtual_stops_for_request(self,
                                       request: Request,
                                       existing_stops: List[Stop],
                                       system_state: SimulationState) -> Tuple[Stop, Stop]:
        """
        Create optimally placed virtual stops for a specific request when no existing stops are viable.
        Uses similar principles as coverage-based selection but optimized for immediate needs.
        
        Args:
            request: The request needing virtual stops
            existing_stops: List of all current stops
            system_state: Current system state
            
        Returns:
            Tuple of (origin_stop, destination_stop)
        """
        # Get candidate points around request locations
        origin_candidates = self._generate_candidates_around_point(
            request.origin,
            existing_stops,
            is_pickup=True
        )
        
        dest_candidates = self._generate_candidates_around_point(
            request.destination,
            existing_stops,
            is_pickup=False
        )
        
        # Select optimal locations asynchronously
        origin_location, dest_location = await asyncio.gather(
            self._select_optimal_virtual_location(
                request.origin,
                origin_candidates,
                existing_stops,
                is_pickup=True,
                system_state=system_state
            ),
            self._select_optimal_virtual_location(
                request.destination,
                dest_candidates,
                existing_stops,
                is_pickup=False,
                system_state=system_state
            )
        )
        
        # Create virtual stops
        origin_stop = Stop(
            id=f"virtual_origin_{request.id}",
            location=origin_location,
            type=StopType.VIRTUAL,
            status=StopStatus.ACTIVE,
            capacity=self.config.default_virtual_stop_capacity,
            current_load=0,
            metadata={
                'virtual': True,
                'request_id': request.id,
                'creation_time': system_state.current_time,
                'type': 'origin',
                'selection_method': 'coverage_based'
            }
        )
        
        dest_stop = Stop(
            id=f"virtual_dest_{request.id}",
            location=dest_location,
            type=StopType.VIRTUAL,
            status=StopStatus.ACTIVE,
            capacity=self.config.default_virtual_stop_capacity,
            current_load=0,
            metadata={
                'virtual': True,
                'request_id': request.id,
                'creation_time': system_state.current_time,
                'type': 'destination',
                'selection_method': 'coverage_based'
            }
        )
        
        return origin_stop, dest_stop

    def _generate_candidates_around_point(self,
                                       center: Location,
                                       existing_stops: List[Stop],
                                       is_pickup: bool) -> List[Location]:
        """Generate candidate locations around a center point."""
        candidates = []
        
        # Define search grid
        radius = self.config.coverage_radius
        grid_size = radius / 5  # Create a 5x5 grid within radius
        
        for i in range(-2, 3):
            for j in range(-2, 3):
                lat = center.lat + (i * grid_size / 111000)  # Convert to degrees
                lon = center.lon + (j * grid_size / 111000 / np.cos(np.radians(center.lat)))
                
                candidate = Location(lat=lat, lon=lon)
                
                # Check if location is suitable
                if self._is_candidate_suitable(
                    candidate,
                    existing_stops,
                    is_pickup
                ):
                    candidates.append(candidate)
        
        return candidates

    async def _select_optimal_virtual_location(self,
                                      request_point: Location,
                                      candidates: List[Location],
                                      existing_stops: List[Stop],
                                      is_pickup: bool,
                                      system_state: Optional[Dict[str, Any]] = None) -> Location:
        """Select optimal location from candidates for virtual stop placement."""
        if not candidates:
            return request_point  # Fallback to original location if no suitable candidates
            
        scored_candidates = []
        scoring_tasks = []
        
        # Create tasks for scoring all candidates concurrently
        for candidate in candidates:
            task = self._score_virtual_location(
                candidate,
                request_point,
                existing_stops,
                is_pickup,
                system_state
            )
            scoring_tasks.append(task)
            
        # Wait for all scoring tasks to complete
        scores = await asyncio.gather(*scoring_tasks)
        
        # Combine candidates with their scores
        scored_candidates = list(zip(candidates, scores))
            
        # Select best candidate
        return max(scored_candidates, key=lambda x: x[1])[0]

    def _is_candidate_suitable(self,
                            location: Location,
                            existing_stops: List[Stop],
                            is_pickup: bool) -> bool:
        """Check if a candidate location is suitable for a virtual stop."""
        # Check minimum spacing from existing stops
        for stop in existing_stops:
            if self._calculate_distance(location, stop.location) < self.config.min_stop_spacing:
                return False
                
        # Additional checks could include:
        # - Street network accessibility
        # - Safety considerations
        # - Vehicle access restrictions
        # These would typically use external data sources or APIs
        
        return True

    async def _score_virtual_location(self,
                             candidate: Location,
                             request_point: Location,
                             existing_stops: List[Stop],
                             is_pickup: bool,
                             system_state: SimulationState) -> float:
        """
        Score a candidate location for virtual stop placement.
        Considers multiple factors weighted by their importance.
        """
        scores = {}
        
        # 1. Distance from request point (closer is better)
        distance = await self._calculate_network_distance(candidate, request_point, network_type='walk')
        scores['distance'] = 1.0 - (distance / self.config.coverage_radius)
        
        # 2. Coverage optimization (how well it fits with existing stops)
        coverage_score = await self._calculate_coverage_score(candidate, existing_stops)
        scores['coverage'] = coverage_score
        
        # 3. Operational efficiency
        efficiency_score = await self._calculate_operational_efficiency(
            candidate,
            system_state.vehicles.vehicles,
            is_pickup
        )
        scores['efficiency'] = efficiency_score
        
        # 4. Future demand potential
        # demand_score = self._calculate_demand_potential(
        #     candidate,
        #     system_state.demand_patterns
        # )
        # scores['demand'] = demand_score
        
        # Weight and combine scores
        weights = {
            'distance': 0.5,
            'coverage': 0.3,
            'efficiency': 0.2,
            # 'demand': 0.1
        }
        
        return sum(score * weights[factor] for factor, score in scores.items())

    async def _calculate_coverage_score(self,
                               candidate: Location,
                               existing_stops: List[Stop]) -> float:
        """Calculate how well a candidate location optimizes coverage."""
        if not existing_stops:
            return 1.0
            
        # Calculate distances to existing stops concurrently
        distance_tasks = [
            self._calculate_network_distance(candidate, stop.location, network_type='walk')
            for stop in existing_stops
        ]
        distances = await asyncio.gather(*distance_tasks)
        
        # Prefer locations that fill gaps in coverage
        min_distance = min(distances)
        
        if min_distance < self.config.min_stop_spacing:
            return 0.0
        elif min_distance > self.config.coverage_radius:
            return 1.0
        else:
            # Score based on optimal spacing
            optimal_distance = (self.config.min_stop_spacing + self.config.coverage_radius) / 2
            return 1.0 - abs(min_distance - optimal_distance) / optimal_distance

    async def _calculate_operational_efficiency(self,
                                      location: Location,
                                      vehicles: Dict[str, VehicleState],
                                      is_pickup: bool) -> float:
        """Calculate operational efficiency score based on vehicle positions."""
        if not vehicles:
            return 0.5
        
        
        # Calculate distances to all vehicles concurrently
        distance_tasks = [
            self._calculate_network_distance(location, vehicle.current_location, network_type='drive')
            for vehicle in vehicles.values()
        ]
        distances = await asyncio.gather(*distance_tasks)
        
        min_distance = min(distances)
        
        # Score based on vehicle proximity
        max_desired_distance = 1500  # 1.5km
        
        if min_distance > max_desired_distance:
            return 0.0
        else:
            return 1.0 - (min_distance / max_desired_distance)

    def _calculate_demand_potential(self,
                                 location: Location,
                                 demand_patterns: Dict[str, Any]) -> float:
        """Calculate potential future demand score for a location."""
        # Extract relevant area demand
        area_key = self._get_area_key(location)
        area_demand = demand_patterns.get(area_key, {})
        
        if not area_demand:
            return 0.5
        
        # Consider both current and predicted demand
        current_demand = area_demand.get('current_demand', 0)
        predicted_demand = area_demand.get('predicted_demand', 0)
        
        # Normalize against system-wide demand
        max_demand = max(
            max(d.get('current_demand', 0) for d in demand_patterns.values()),
            max(d.get('predicted_demand', 0) for d in demand_patterns.values())
        )
        
        if max_demand == 0:
            return 0.5
            
        current_score = current_demand / max_demand
        predicted_score = predicted_demand / max_demand
        
        # Weight current vs predicted demand
        return 0.7 * current_score + 0.3 * predicted_score