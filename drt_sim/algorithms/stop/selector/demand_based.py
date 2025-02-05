from typing import List, Dict, Any, Optional, Tuple, Set
from collections import defaultdict

from drt_sim.models.stop import Stop, StopStatus
from drt_sim.models.location import Location
from drt_sim.algorithms.base_interfaces.stop_selector_base import (
    StopSelector,
)
from drt_sim.models.request import Request
import numpy as np

class DemandBasedStopSelector(StopSelector):
    """
    Selects stops based on demand patterns and density.
    Prioritizes areas with higher demand while ensuring minimum coverage.
    """

    def select_stops(self,
                    candidate_locations: List[Location],
                    demand_points: Optional[List[Location]] = None,
                    existing_stops: Optional[List[Stop]] = None,
                    constraints: Optional[Dict[str, Any]] = None) -> List[Stop]:
        """Select stops based on demand patterns."""
        if not candidate_locations or not demand_points:
            raise ValueError("Both candidate locations and demand points are required")

        # Create demand density map
        demand_density = self._calculate_demand_density(demand_points)
        
        # Score candidate locations
        scored_candidates = []
        for location in candidate_locations:
            score = self._score_location(location, demand_density)
            if score >= self.config.min_demand_threshold:
                scored_candidates.append((location, score))
        
        # Sort candidates by score
        scored_candidates.sort(key=lambda x: x[1], reverse=True)
        
        # Select stops ensuring minimum spacing
        selected_stops = []
        for location, score in scored_candidates:
            if self._respects_spacing(location, selected_stops):
                stop = self._create_stop(
                    location,
                    metadata={'demand_score': score}
                )
                selected_stops.append(stop)
                
                # Check if we've reached maximum stops
                if self.config.max_stops and len(selected_stops) >= self.config.max_stops:
                    break

        # Ensure minimum coverage if needed
        if existing_stops:
            selected_stops = self._ensure_coverage(
                selected_stops,
                existing_stops,
                candidate_locations
            )

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

        # Get current demand patterns
        current_demand = self._get_current_demand(system_state)
        if not current_demand:
            return current_stops, []

        # Process each stop
        for stop in current_stops:
            demand_score = self._calculate_current_demand_score(
                stop.location,
                current_demand,
                demand_changes
            )
            
            if demand_score < self.config.min_demand_threshold:
                # Check if stop can be safely removed
                if self._can_remove_stop(stop, current_stops, current_demand):
                    stop.status = StopStatus.INACTIVE
                    modified_stops.append(stop)
                    modified_ids.append(stop.id)
            else:
                # Update stop metadata
                stop.metadata['demand_score'] = demand_score
                modified_stops.append(stop)
                modified_ids.append(stop.id)

        # Check for new high-demand areas
        new_stops = self._add_high_demand_stops(
            current_stops,
            current_demand,
            system_state
        )
        modified_stops.extend(new_stops)
        modified_ids.extend(s.id for s in new_stops)

        # Update complete stop list
        updated_stops = [
            stop if stop.id not in modified_ids else
            next(s for s in modified_stops if s.id == stop.id)
            for stop in current_stops
        ] + [s for s in modified_stops if s.id not in {stop.id for stop in current_stops}]

        return updated_stops, modified_ids
    
    def create_virtual_stops_for_request(self,
                                    request: Request,
                                    existing_stops: List[Stop],
                                    system_state: Optional[Dict[str, Any]] = None) -> Tuple[Stop, Stop]:
        """
        Create virtual stops for a request, optimizing placement based on demand patterns.
        
        Args:
            request: The request needing virtual stops
            existing_stops: List of all current stops
            system_state: Current system state including demand patterns
            
        Returns:
            Tuple of (origin_stop, destination_stop)
        """
        # Get current demand patterns
        current_demand = self._get_current_demand(system_state) or {}
        
        # Generate candidate locations around request points
        origin_candidates = self._generate_candidates_around_point(
            request.origin,
            existing_stops
        )
        
        dest_candidates = self._generate_candidates_around_point(
            request.destination,
            existing_stops
        )
        
        # Select optimal locations considering demand
        origin_location = self._select_optimal_virtual_location(
            request.origin,
            origin_candidates,
            existing_stops,
            current_demand,
            system_state
        )
        
        dest_location = self._select_optimal_virtual_location(
            request.destination,
            dest_candidates,
            existing_stops,
            current_demand,
            system_state
        )
        
        # Create virtual stops
        origin_stop = Stop(
            id=f"virtual_origin_{request.id}",
            location=origin_location,
            stop_type="virtual",
            status=StopStatus.ACTIVE,
            capacity=self.config.default_virtual_stop_capacity,
            current_load=0,
            metadata={
                'virtual': True,
                'request_id': request.id,
                'creation_time': system_state.get('current_time') if system_state else None,
                'type': 'origin',
                'selection_method': 'demand_based'
            }
        )
        
        dest_stop = Stop(
            id=f"virtual_dest_{request.id}",
            location=dest_location,
            stop_type="virtual",
            status=StopStatus.ACTIVE,
            capacity=self.config.default_virtual_stop_capacity,
            current_load=0,
            metadata={
                'virtual': True,
                'request_id': request.id,
                'creation_time': system_state.get('current_time') if system_state else None,
                'type': 'destination',
                'selection_method': 'demand_based'
            }
        )
        
        return origin_stop, dest_stop

    def _generate_candidates_around_point(self,
                                    center: Location,
                                    existing_stops: List[Stop]) -> List[Location]:
        """Generate candidate locations around a center point."""
        candidates = []
        
        # Define search grid
        GRID_SIZE = 0.001  # Approximately 100m
        search_radius_cells = int(self.config.coverage_radius / 100)  # Convert meters to grid cells
        
        # Generate grid points
        for dx in range(-search_radius_cells, search_radius_cells + 1):
            for dy in range(-search_radius_cells, search_radius_cells + 1):
                # Skip points outside circular radius
                if dx*dx + dy*dy > search_radius_cells*search_radius_cells:
                    continue
                    
                lat = center.lat + (dy * GRID_SIZE)
                lon = center.lon + (dx * GRID_SIZE / np.cos(np.radians(center.lat)))
                
                candidate = Location(lat=lat, lon=lon)
                
                # Check spacing requirements
                if self._respects_spacing(candidate, existing_stops):
                    candidates.append(candidate)
        
        return candidates

    def _select_optimal_virtual_location(self,
                                    request_point: Location,
                                    candidates: List[Location],
                                    existing_stops: List[Stop],
                                    current_demand: Dict[str, Any],
                                    system_state: Optional[Dict[str, Any]] = None) -> Location:
        """Select optimal location from candidates for virtual stop placement."""
        if not candidates:
            return request_point
            
        scored_candidates = []
        for candidate in candidates:
            # Calculate combined score based on multiple factors
            scores = {
                'demand': self._calculate_demand_score(candidate, current_demand),
                'distance': self._calculate_distance_score(candidate, request_point),
                'accessibility': self._calculate_accessibility_score(candidate, system_state)
            }
            
            # Weight factors
            weights = {
                'demand': 0.5,  # Prioritize demand patterns
                'distance': 0.35,  # Important but not primary
                'accessibility': 0.15  # Minor factor
            }
            
            total_score = sum(score * weights[factor] for factor, score in scores.items())
            scored_candidates.append((candidate, total_score))
        
        # Select location with highest score
        return max(scored_candidates, key=lambda x: x[1])[0]

    def _calculate_demand_score(self,
                            location: Location,
                            current_demand: Dict[str, Any]) -> float:
        """Calculate demand-based score for a location."""
        if 'density_map' in current_demand:
            return self._score_location(location, current_demand['density_map'])
        return 0.5

    def _calculate_distance_score(self,
                            candidate: Location,
                            request_point: Location) -> float:
        """Calculate distance-based score for a location."""
        distance = self._calculate_distance(candidate, request_point)
        if distance > self.config.coverage_radius:
            return 0.0
        return 1.0 - (distance / self.config.coverage_radius)

    def _calculate_accessibility_score(self,
                                    location: Location,
                                    system_state: Optional[Dict[str, Any]]) -> float:
        """Calculate accessibility score based on system state."""
        if not system_state or 'vehicle_positions' not in system_state:
            return 0.5
            
        # Find closest vehicle
        min_distance = float('inf')
        for vehicle_loc in system_state['vehicle_positions'].values():
            distance = self._calculate_distance(location, vehicle_loc)
            min_distance = min(min_distance, distance)
        
        # Score based on vehicle proximity
        max_desired_distance = 2000  # 2km
        if min_distance > max_desired_distance:
            return 0.0
        return 1.0 - (min_distance / max_desired_distance)

    def _calculate_demand_density(self, demand_points: List[Location]) -> Dict[Tuple[int, int], int]:
        """Calculate demand density using a grid-based approach."""
        # Use grid cells of approximately 100m x 100m
        GRID_SIZE = 0.001  # roughly 100m in decimal degrees
        
        density_map = defaultdict(int)
        for point in demand_points:
            # Convert to grid coordinates
            grid_x = int(point.lon / GRID_SIZE)
            grid_y = int(point.lat / GRID_SIZE)
            density_map[(grid_x, grid_y)] += 1
            
        return density_map

    def _score_location(self, 
                       location: Location, 
                       density_map: Dict[Tuple[int, int], int]) -> float:
        """Score a location based on nearby demand density."""
        GRID_SIZE = 0.001
        grid_x = int(location.lon / GRID_SIZE)
        grid_y = int(location.lat / GRID_SIZE)
        
        # Sum demand in surrounding cells
        score = 0
        for dx in range(-2, 3):  # Check 5x5 grid around point
            for dy in range(-2, 3):
                cell = (grid_x + dx, grid_y + dy)
                # Weight by distance from center
                distance_weight = 1 / (1 + abs(dx) + abs(dy))
                score += density_map.get(cell, 0) * distance_weight
                
        return score

    def _respects_spacing(self, 
                         location: Location, 
                         existing_stops: List[Stop]) -> bool:
        """Check if location respects minimum spacing requirements."""
        return all(
            self._calculate_distance(location, stop.location) >= self.config.min_stop_spacing
            for stop in existing_stops
        )

    def _ensure_coverage(self,
                        selected_stops: List[Stop],
                        existing_stops: List[Stop],
                        candidate_locations: List[Location]) -> List[Stop]:
        """Ensure minimum coverage is maintained when updating stops."""
        # Find areas that lost coverage
        covered_areas = set()
        for stop in selected_stops:
            covered_areas.update(self._get_covered_cells(stop.location))
            
        previously_covered = set()
        for stop in existing_stops:
            previously_covered.update(self._get_covered_cells(stop.location))
            
        uncovered = previously_covered - covered_areas
        
        if uncovered:
            # Add stops to maintain coverage
            additional_stops = []
            for location in candidate_locations:
                if len(selected_stops) + len(additional_stops) >= self.config.max_stops:
                    break
                    
                cell = self._get_cell(location)
                if cell in uncovered and self._respects_spacing(location, selected_stops + additional_stops):
                    stop = self._create_stop(location)
                    additional_stops.append(stop)
                    covered_areas.update(self._get_covered_cells(location))
                    
            selected_stops.extend(additional_stops)
            
        return selected_stops

    def _get_current_demand(self, 
                           system_state: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """Extract current demand patterns from system state."""
        if not system_state or 'demand_patterns' not in system_state:
            return None
        return system_state['demand_patterns']

    def _calculate_current_demand_score(self,
                                      location: Location,
                                      current_demand: Dict[str, Any],
                                      demand_changes: Dict[str, float]) -> float:
        """Calculate current demand score for a location."""
        base_score = 0
        
        # Calculate base demand score from current patterns
        if 'density_map' in current_demand:
            base_score = self._score_location(location, current_demand['density_map'])
            
        # Apply demand changes
        for area_id, change_factor in demand_changes.items():
            if self._is_location_in_area(location, area_id, current_demand):
                base_score *= change_factor
                
        return base_score

    def _can_remove_stop(self,
                        stop: Stop,
                        all_stops: List[Stop],
                        current_demand: Dict[str, Any]) -> bool:
        """Determine if a stop can be safely removed."""
        # Check if removing would create coverage gaps
        covered_by_others = False
        stop_coverage = self._get_covered_cells(stop.location)
        
        for other in all_stops:
            if other.id != stop.id and other.status == StopStatus.ACTIVE:
                other_coverage = self._get_covered_cells(other.location)
                if stop_coverage.issubset(other_coverage):
                    covered_by_others = True
                    break
                    
        if not covered_by_others:
            return False
            
        # Check if stop has significant recent usage
        if 'stop_usage' in current_demand:
            recent_usage = current_demand['stop_usage'].get(stop.id, 0)
            if recent_usage > self.config.min_demand_threshold:
                return False
                
        return True

    def _add_high_demand_stops(self,
                              current_stops: List[Stop],
                              current_demand: Dict[str, Any],
                              system_state: Optional[Dict[str, Any]]) -> List[Stop]:
        """Add new stops in high demand areas."""
        new_stops = []
        
        if 'high_demand_areas' in current_demand:
            for area in current_demand['high_demand_areas']:
                # Check if area already has coverage
                if self._area_has_coverage(area, current_stops):
                    continue
                    
                # Find best location in area
                location = self._find_best_location_in_area(
                    area,
                    current_stops + new_stops,
                    system_state
                )
                
                if location:
                    stop = self._create_stop(location)
                    new_stops.append(stop)
                    
        return new_stops

    def _get_covered_cells(self, location: Location) -> Set[Tuple[int, int]]:
        """Get grid cells covered by a stop."""
        GRID_SIZE = 0.001
        center_x = int(location.lon / GRID_SIZE)
        center_y = int(location.lat / GRID_SIZE)
        
        coverage_radius_cells = int(self.config.coverage_radius / 100)  # Convert meters to grid cells
        
        covered = set()
        for dx in range(-coverage_radius_cells, coverage_radius_cells + 1):
            for dy in range(-coverage_radius_cells, coverage_radius_cells + 1):
                if dx*dx + dy*dy <= coverage_radius_cells*coverage_radius_cells:
                    covered.add((center_x + dx, center_y + dy))
                    
        return covered

    def _get_cell(self, location: Location) -> Tuple[int, int]:
        """Get grid cell for a location."""
        GRID_SIZE = 0.001
        return (int(location.lon / GRID_SIZE),
                int(location.lat / GRID_SIZE))

    def _is_location_in_area(self,
                            location: Location,
                            area_id: str,
                            demand_data: Dict[str, Any]) -> bool:
        """Check if location falls within an area."""
        if 'areas' not in demand_data:
            return False
            
        area = demand_data['areas'].get(area_id)
        if not area or 'bounds' not in area:
            return False
            
        bounds = area['bounds']
        return (bounds['min_lat'] <= location.lat <= bounds['max_lat'] and
                bounds['min_lon'] <= location.lon <= bounds['max_lon'])

    def _area_has_coverage(self, area: Dict[str, Any], stops: List[Stop]) -> bool:
        """Check if an area has stop coverage."""
        if 'center' not in area:
            return False
            
        center_location = Location(
            lat=area['center']['lat'],
            lon=area['center']['lon']
        )
        
        return any(
            self._calculate_distance(center_location, stop.location) <= self.config.coverage_radius
            for stop in stops if stop.status == StopStatus.ACTIVE
        )

    def _find_best_location_in_area(self,
                                   area: Dict[str, Any],
                                   existing_stops: List[Stop],
                                   system_state: Optional[Dict[str, Any]]) -> Optional[Location]:
        """Find best location for a new stop in an area."""
        if 'candidate_locations' not in area:
            return None
            
        best_location = None
        best_score = -float('inf')
        
        for candidate in area['candidate_locations']:
            location = Location(
                lat=candidate['lat'],
                lon=candidate['lon']
            )
            
            if not self._respects_spacing(location, existing_stops):
                continue
                
            score = self._score_location(
                location,
                system_state.get('demand_patterns', {}).get('density_map', {})
                if system_state else {}
            )
            
            if score > best_score:
                best_score = score
                best_location = location
                
        return best_location