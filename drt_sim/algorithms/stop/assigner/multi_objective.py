from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Tuple
import asyncio
from concurrent.futures import ThreadPoolExecutor
import math
from drt_sim.algorithms.base_interfaces.stop_assigner_base import (
    StopAssigner,
    StopAssignment,
)
from drt_sim.models.stop import Stop, StopStatus
from drt_sim.models.request import Request
from drt_sim.models.vehicle import VehicleStatus
from drt_sim.models.vehicle import Vehicle
from drt_sim.models.location import Location
from drt_sim.core.logging_config import setup_logger
from datetime import timedelta
logger = setup_logger(__name__)

@dataclass
class StopScore:
    """Simplified scoring breakdown for stop assignments"""
    vehicle_access_time_score: float
    passenger_access_time_score: float
    final_score: float

def haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """
    Calculate the great circle distance between two points 
    on the earth (specified in decimal degrees)
    """
    # Convert decimal degrees to radians
    lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])

    # Haversine formula
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
    c = 2 * math.asin(math.sqrt(a))
    # Radius of earth in kilometers
    r = 6371
    return c * r * 1000  # Convert to meters

class MultiObjectiveStopAssigner(StopAssigner):
    """
    Modified stop assigner that evaluates origin and destination stops independently
    using three main criteria: vehicle access time, passenger access time, and stop utilization.
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._thread_pool = ThreadPoolExecutor(max_workers=4)
    
    async def assign_stops(self,
                    request: Request,
                    available_stops: List[Stop]) -> StopAssignment:
        # Get all viable active stops
        viable_stops = [
            stop for stop in available_stops
            if stop.status == StopStatus.ACTIVE
        ]
        
        if not viable_stops:
            raise ValueError("No active stops available")

        # Pre-filter stops using fast haversine distance calculation
        max_walking_distance = self.config.thresholds["max_walking_time"] * self.network_manager.config.walking_speed
        
        # Calculate haversine distances for quick filtering
        origin_candidates = []
        dest_candidates = []
        
        for stop in viable_stops:
            # Origin distance
            origin_dist = haversine_distance(
                request.origin.lat, 
                request.origin.lon,
                stop.location.lat, 
                stop.location.lon
            )
            if origin_dist <= max_walking_distance * 1.2:  # Add 20% buffer for straight-line vs network distance
                origin_candidates.append((stop, origin_dist))
                
            # Destination distance
            dest_dist = haversine_distance(
                request.destination.lat, 
                request.destination.lon,
                stop.location.lat, 
                stop.location.lon
            )
            if dest_dist <= max_walking_distance * 1.2:  # Add 20% buffer for straight-line vs network distance
                dest_candidates.append((stop, dest_dist))
        
        # Sort and take top candidates
        origin_candidates = sorted(origin_candidates, key=lambda x: x[1])[:3]
        dest_candidates = sorted(dest_candidates, key=lambda x: x[1])[:3]

        if not origin_candidates or not dest_candidates:
            raise ValueError("No viable stop combinations found within walking distance")

        # Extract just the stops for scoring
        filtered_origin_stops = [stop for stop, _ in origin_candidates]
        filtered_dest_stops = [stop for stop, _ in dest_candidates]

        # Concurrently score filtered origin and destination stops
        origin_scores, dest_scores = await asyncio.gather(
            self._score_stops_batch(request.origin, filtered_origin_stops),
            self._score_stops_batch(request.destination, filtered_dest_stops)
        )

        logger.info(f"Origin scores: {origin_scores}")
        logger.info(f"Destination scores: {dest_scores}")
        
        if not origin_scores or not dest_scores:
            raise ValueError("No viable stop combinations found")
        
        # Sort scores in thread pool
        sorted_scores = await asyncio.gather(
            self._run_in_thread(lambda: sorted(origin_scores, key=lambda x: x[2].final_score, reverse=True)),
            self._run_in_thread(lambda: sorted(dest_scores, key=lambda x: x[2].final_score, reverse=True))
        )
        origin_scores, dest_scores = sorted_scores
        
        # Select best stops
        best_origin, origin_dist, origin_score = origin_scores[0]
        best_dest, dest_dist, dest_score = dest_scores[0]
        
        # Get alternatives safely (handling cases with fewer alternatives than max)
        alt_origins = [stop for stop, _, _ in origin_scores[1:min(len(origin_scores), self.config.max_alternatives + 1)]]
        alt_dests = [stop for stop, _, _ in dest_scores[1:min(len(dest_scores), self.config.max_alternatives + 1)]]
        
        walking_time_origin = origin_dist / self.network_manager.config.walking_speed
        walking_time_destination = dest_dist / self.network_manager.config.walking_speed
        
        return StopAssignment(
            request_id=request.id,
            origin_stop=best_origin,
            destination_stop=best_dest,
            walking_distance_origin=origin_dist,
            walking_distance_destination=dest_dist,
            walking_time_origin=walking_time_origin,
            walking_time_destination=walking_time_destination,
            expected_passenger_origin_stop_arrival_time=self.sim_context.current_time + timedelta(seconds=walking_time_origin),
            total_score=(origin_score.final_score + dest_score.final_score) / 2,
            alternative_origins=alt_origins,
            alternative_destinations=alt_dests,
            assignment_time=self.sim_context.current_time,
            metadata={
                'origin_score_breakdown': {
                    'vehicle_access_time': origin_score.vehicle_access_time_score,
                    'passenger_access_time': origin_score.passenger_access_time_score,
                },
                'destination_score_breakdown': {
                    'vehicle_access_time': dest_score.vehicle_access_time_score,
                    'passenger_access_time': dest_score.passenger_access_time_score,
                }
            }
        )

    async def _score_stops_batch(self,
                         location: Location,
                         stops: List[Stop]) -> List[Tuple[Stop, float, StopScore]]:
        """Calculate scores for multiple stops concurrently"""
        # Calculate distances in parallel
        times = await asyncio.gather(*[
            self.network_manager.calculate_travel_time(
                location,
                stop.location,
                network_type='walk'
            ) for stop in stops
        ])
        
        # Filter and score valid stops concurrently
        valid_stops = [
            (stop, time) for stop, time in zip(stops, times)
            if time <= self.config.thresholds["max_walking_time"]
        ]
        
        if not valid_stops:
            return []
            
        scores = await asyncio.gather(*[
            self._calculate_detailed_scores(
                stop=stop,  
                time=time
            ) for stop, time in valid_stops
        ])
        
        return [(stop, time, score) for (stop, time), score in zip(valid_stops, scores)]

    async def _calculate_detailed_scores(self,
                                 stop: Stop,
                                 time: float) -> StopScore:
        """Calculate detailed scoring breakdown for a single stop."""
        # Calculate passenger access time score (based on walking distance)
        passenger_access_time_score = 1 - (time / self.config.thresholds["max_walking_time"])
        
        # Calculate vehicle access time score
        vehicle_access_time_score = await self._calculate_vehicle_access_score(stop)
        
        # Calculate final weighted score using config weights
        final_score = (
            self.config.weights["vehicle_access_time"] * vehicle_access_time_score +
            self.config.weights["passenger_access_time"] * passenger_access_time_score
        )
        
        return StopScore(
            vehicle_access_time_score=vehicle_access_time_score,
            passenger_access_time_score=passenger_access_time_score,
            final_score=final_score
        )

    async def _calculate_vehicle_access_score(self, stop: Stop) -> float:
        """Calculate accessibility score based on available vehicles and their access to the stop."""
        # Get available vehicles
        available_vehicles = self.state_manager.vehicle_worker.get_available_vehicles()
        if not available_vehicles:
            return 0.0
        
        # Filter active vehicles
        active_vehicles = [
            vehicle for vehicle in available_vehicles
            if vehicle.current_state.status not in [
                VehicleStatus.CHARGING,
                VehicleStatus.OFF_DUTY,
            ]
        ]
        
        if not active_vehicles:
            return 0.0
        
        # Calculate distances concurrently
        distances = await asyncio.gather(*[
            self.network_manager.calculate_distance(
                vehicle.current_state.current_location,
                stop.location,
                network_type='drive'
            ) for vehicle in active_vehicles
        ])
        
        # Calculate scores in thread pool
        def calculate_scores(vehicles: List[Vehicle], distances: List[float]) -> List[float]:
            scores = []
            for vehicle, distance in zip(vehicles, distances):
                # Convert distance to time based on average speed
                access_time = distance / self.network_manager.config.driving_speed
                time_score = max(0.0, 1.0 - (access_time / self.config.thresholds["max_driving_time"]))
                
                capacity_utilization = vehicle.current_state.current_occupancy / vehicle.capacity
                capacity_score = max(0.0, 1.0 - capacity_utilization)
                
                vehicle_score = 0.8 * time_score + 0.3 * capacity_score
                scores.append(vehicle_score)
            return scores
        
        vehicle_scores = await self._run_in_thread(
            lambda: calculate_scores(active_vehicles, distances)
        )
        
        if not vehicle_scores:
            return 0.0
        
        # Take the average of the top 3 vehicle scores
        top_scores = sorted(vehicle_scores, reverse=True)[:3]
        return sum(top_scores) / len(top_scores)

    async def _run_in_thread(self, func):
        """Run CPU-bound operations in thread pool."""
        return await asyncio.get_event_loop().run_in_executor(self._thread_pool, func)