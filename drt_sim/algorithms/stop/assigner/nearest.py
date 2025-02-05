'''Under Construction'''
from typing import List, Dict, Any, Optional, Tuple, Set
import numpy as np
from dataclasses import dataclass
from collections import defaultdict

from drt_sim.models.stop import Stop, StopStatus
from drt_sim.models.location import Location
from drt_sim.models.request import Request
from drt_sim.algorithms.base_interfaces.stop_assigner_base import (
    StopAssigner, 
    StopAssignment,
)

class NearestStopAssigner(StopAssigner):
    """Simple assigner that selects the nearest stops within walking distance."""
    
    def assign_stops(self, 
                    request: Request,
                    available_stops: List[Stop],
                    system_state: Optional[Dict[str, Any]] = None) -> StopAssignment:
        # Get viable stops within walking distance
        origin_candidates = [
            (stop, self._calculate_distance(request.origin, stop.location))
            for stop in available_stops
            if stop.status == StopStatus.ACTIVE
        ]
        
        dest_candidates = [
            (stop, self._calculate_distance(request.destination, stop.location))
            for stop in available_stops
            if stop.status == StopStatus.ACTIVE
        ]
        
        # Filter by walking distance
        origin_candidates = [
            (stop, dist) for stop, dist in origin_candidates
            if dist <= self.config.max_walking_distance
        ]
        dest_candidates = [
            (stop, dist) for stop, dist in dest_candidates
            if dist <= self.config.max_walking_distance
        ]
        
        if not origin_candidates or not dest_candidates:
            raise ValueError("No viable stops found within walking distance")
            
        # Sort by distance
        origin_candidates.sort(key=lambda x: x[1])
        dest_candidates.sort(key=lambda x: x[1])
        
        # Select best stops
        origin_stop, origin_dist = origin_candidates[0]
        dest_stop, dest_dist = dest_candidates[0]
        
        # Get alternatives
        alt_origins = [stop for stop, _ in origin_candidates[1:self.config.max_alternatives+1]]
        alt_dests = [stop for stop, _ in dest_candidates[1:self.config.max_alternatives+1]]
        
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
            assignment_time=system_state.get('current_time', 0) if system_state else 0
        )

    def score_assignment(self,
                        request: Request,
                        origin_stop: Stop,
                        destination_stop: Stop,
                        system_state: Optional[Dict[str, Any]] = None) -> float:
        # Simple distance-based score
        origin_dist = self._calculate_distance(request.origin, origin_stop.location)
        dest_dist = self._calculate_distance(request.destination, destination_stop.location)
        
        # Normalize distances to 0-1 scale (1 is best)
        origin_score = 1 - (origin_dist / self.config.max_walking_distance)
        dest_score = 1 - (dest_dist / self.config.max_walking_distance)
        
        return (origin_score + dest_score) / 2