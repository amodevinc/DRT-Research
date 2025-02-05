'''Under Construction'''
from typing import List, Dict, Any, Optional
from drt_sim.algorithms.base_interfaces.stop_assigner_base import (
    StopAssigner,
    StopAssignment,
)
from drt_sim.models.stop import Stop, StopStatus
from drt_sim.models.request import Request

class AccessibilityFocusedAssigner(StopAssigner):
    """Assigner that prioritizes accessibility while maintaining distance constraints."""
    
    def assign_stops(self,
                    request: Request,
                    available_stops: List[Stop],
                    system_state: Optional[Dict[str, Any]] = None) -> StopAssignment:
        # Consider accessibility needs if specified in request
        accessibility_needs = request.metadata.get('accessibility_needs', {})
        
        # Filter stops by walking distance and get accessibility scores
        viable_stops = []
        for stop in available_stops:
            if stop.status != StopStatus.ACTIVE:
                continue
                
            # Check distances
            origin_dist = self._calculate_distance(request.origin, stop.location)
            dest_dist = self._calculate_distance(request.destination, stop.location)
            
            if origin_dist <= self.config.max_walking_distance:
                accessibility_score = self._calculate_accessibility_score(
                    stop, accessibility_needs)
                viable_stops.append((stop, 'origin', origin_dist, accessibility_score))
                
            if dest_dist <= self.config.max_walking_distance:
                accessibility_score = self._calculate_accessibility_score(
                    stop, accessibility_needs)
                viable_stops.append((stop, 'dest', dest_dist, accessibility_score))
        
        if not viable_stops:
            raise ValueError("No viable stops found within walking distance")
        
        # Separate and sort origin and destination candidates
        origin_candidates = [
            (s, d, a) for s, t, d, a in viable_stops if t == 'origin'
        ]
        dest_candidates = [
            (s, d, a) for s, t, d, a in viable_stops if t == 'dest'
        ]
        
        # Sort by accessibility score, then distance
        origin_candidates.sort(key=lambda x: (-x[2], x[1]))
        dest_candidates.sort(key=lambda x: (-x[2], x[1]))
        
        if not origin_candidates or not dest_candidates:
            raise ValueError("No valid origin-destination pair found")
        
        # Select best stops
        origin_stop, origin_dist, _ = origin_candidates[0]
        dest_stop, dest_dist, _ = dest_candidates[0]
        
        # Get alternatives
        alt_origins = [stop for stop, _, _ in origin_candidates[1:self.config.max_alternatives+1]]
        alt_dests = [stop for stop, _, _ in dest_candidates[1:self.config.max_alternatives+1]]
        
        return StopAssignment(
            request_id=request.id,
            origin_stop=origin_stop,
            destination_stop=dest_stop,
            walking_distance_origin=origin_dist,
            walking_distance_destination=dest_dist,
            walking_time_origin=origin_dist / self.config.walking_speed,
            walking_time_destination=dest_dist / self.config.walking_speed,
            total_score=self.score_assignment(request, origin_stop, dest_stop, system_state),
            alternative_origins=alt_origins,
            alternative_destinations=alt_dests,
            assignment_time=system_state.get('current_time', 0) if system_state else 0,
            metadata={'accessibility_scores': {
                'origin': self._calculate_accessibility_score(origin_stop, accessibility_needs),
                'destination': self._calculate_accessibility_score(dest_stop, accessibility_needs)
            }}
        )

    def _calculate_accessibility_score(self, 
                                    stop: Stop, 
                                    needs: Dict[str, Any]) -> float:
        """Calculate accessibility score based on stop features and user needs."""
        if not needs or 'accessibility' not in stop.metadata:
            return 1.0
            
        score = 0.0
        features = stop.metadata['accessibility']
        
        # Score based on matching features to needs
        for need, importance in needs.items():
            if need in features:
                score += importance * features[need]
                
        return score / sum(needs.values()) if needs else 1.0

    def score_assignment(self,
                        request: Request,
                        origin_stop: Stop,
                        destination_stop: Stop,
                        system_state: Optional[Dict[str, Any]] = None) -> float:
        # Get accessibility needs
        accessibility_needs = request.metadata.get('accessibility_needs', {})
        
        # Calculate scores
        origin_access = self._calculate_accessibility_score(origin_stop, accessibility_needs)
        dest_access = self._calculate_accessibility_score(destination_stop, accessibility_needs)
        
        # Distance scores
        origin_dist = self._calculate_distance(request.origin, origin_stop.location)
        dest_dist = self._calculate_distance(request.destination, destination_stop.location)
        distance_score = 1 - ((origin_dist + dest_dist) / (2 * self.config.max_walking_distance))
        
        # Weighted combination (prioritize accessibility)
        return 0.7 * ((origin_access + dest_access) / 2) + 0.3 * distance_score