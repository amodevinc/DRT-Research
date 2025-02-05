# drt_sim/models/metrics.py
from dataclasses import dataclass, field
from typing import Dict, List
from .base import ModelBase
from datetime import datetime
@dataclass
class OperationalMetrics:
    total_requests: int
    completed_requests: int
    cancelled_requests: int
    average_wait_time: float
    average_detour_time: float
    vehicle_utilization: float
    total_distance: float
    empty_distance: float
    fuel_consumption: float
    co2_emissions: float

@dataclass
class ServiceMetrics:
    on_time_performance: float
    average_passenger_rating: float
    service_reliability: float
    coverage_area: float
    accessibility_score: float
    cost_per_trip: float

@dataclass
class RequestMetrics:
    """Metrics tracked for requests"""
    total_requests: int = 0
    active_requests: int = 0
    accepted_requests: int = 0
    rejected_requests: int = 0
    completed_requests: int = 0
    cancelled_requests: int = 0
    average_wait_time: float = 0.0
    average_service_time: float = 0.0
    total_wait_time: float = 0.0
    total_service_time: float = 0.0

@dataclass
class IndividualVehicleMetrics:
    """Metrics tracked for an individual vehicle"""
    # Distance metrics
    total_distance: float = 0.0
    empty_distance: float = 0.0
    loaded_distance: float = 0.0
    
    # Time metrics
    total_time: float = 0.0
    idle_time: float = 0.0
    service_time: float = 0.0
    
    # Service metrics
    service_count: int = 0
    passenger_count: int = 0
    current_occupancy: int = 0
    total_stops_visited: int = 0
    
    # Efficiency metrics
    occupancy_rate: float = 0.0
    utilization_rate: float = 0.0
    
    # Energy/fuel metrics
    fuel_consumption: float = 0.0
    energy_usage: float = 0.0
    
    def update_distance(self, distance: float, loaded: bool = False) -> None:
        self.total_distance += distance
        if loaded:
            self.loaded_distance += distance
        else:
            self.empty_distance += distance
            
    def update_time(self, duration: float, is_idle: bool = False) -> None:
        self.total_time += duration
        if is_idle:
            self.idle_time += duration
        else:
            self.service_time += duration
            
    def update_occupancy(self, passengers: int) -> None:
        self.current_occupancy = passengers
        self.passenger_count += passengers
        if self.total_time > 0:
            self.occupancy_rate = self.service_time / self.total_time


@dataclass
class VehicleMetrics:
    """Aggregated metrics for all vehicles in the system"""
    # Fleet size metrics
    total_vehicles: int = 0
    active_vehicles: int = 0
    idle_vehicles: int = 0
    
    # Aggregated distance metrics
    total_fleet_distance: float = 0.0
    total_empty_distance: float = 0.0
    total_loaded_distance: float = 0.0
    average_daily_distance: float = 0.0
    
    # Aggregated time metrics
    total_fleet_time: float = 0.0
    total_fleet_idle_time: float = 0.0
    total_fleet_service_time: float = 0.0
    average_utilization_rate: float = 0.0
    
    # Aggregated service metrics
    total_services: int = 0
    total_passengers: int = 0
    total_stops_visited: int = 0
    average_occupancy_rate: float = 0.0
    
    # Aggregated efficiency metrics
    fleet_utilization_rate: float = 0.0
    total_fuel_consumption: float = 0.0
    total_energy_usage: float = 0.0
    
    # Storage for individual metrics
    individual_metrics: Dict[str, IndividualVehicleMetrics] = field(default_factory=dict)
    
    def update_aggregated_metrics(self) -> None:
        """Update all aggregated metrics based on individual metrics"""
        active_metrics = [m for m in self.individual_metrics.values()]
        n_active = len(active_metrics)
        
        if n_active == 0:
            return
            
        # Update distance metrics
        self.total_fleet_distance = sum(m.total_distance for m in active_metrics)
        self.total_empty_distance = sum(m.empty_distance for m in active_metrics)
        self.total_loaded_distance = sum(m.loaded_distance for m in active_metrics)
        
        # Update time metrics
        self.total_fleet_time = sum(m.total_time for m in active_metrics)
        self.total_fleet_idle_time = sum(m.idle_time for m in active_metrics)
        self.total_fleet_service_time = sum(m.service_time for m in active_metrics)
        
        # Update service metrics
        self.total_services = sum(m.service_count for m in active_metrics)
        self.total_passengers = sum(m.passenger_count for m in active_metrics)
        self.total_stops_visited = sum(m.total_stops_visited for m in active_metrics)
        
        # Calculate averages
        if self.total_fleet_time > 0:
            self.fleet_utilization_rate = self.total_fleet_service_time / self.total_fleet_time
        self.average_occupancy_rate = sum(m.occupancy_rate for m in active_metrics) / n_active
        self.average_utilization_rate = sum(m.utilization_rate for m in active_metrics) / n_active
        
        # Update efficiency metrics
        self.total_fuel_consumption = sum(m.fuel_consumption for m in active_metrics)
        self.total_energy_usage = sum(m.energy_usage for m in active_metrics)

@dataclass
class IndividualPassengerMetrics:
    """Metrics tracked for an individual passenger"""
    # Timing metrics
    request_time: datetime
    total_wait_time: float = 0.0
    total_in_vehicle_time: float = 0.0
    total_walking_time: float = 0.0
    total_journey_time: float = 0.0
    
    # Distance metrics
    access_walking_distance: float = 0.0
    egress_walking_distance: float = 0.0
    in_vehicle_distance: float = 0.0
    route_deviation_ratio: float = 0.0
    
    # Service quality metrics
    service_level_violations: List[str] = field(default_factory=list)
    on_time_pickup: bool = False
    on_time_dropoff: bool = False

@dataclass
class PassengerMetrics:
    """Metrics tracked for all passengers"""
    # Counts
    total_passengers: int = 0
    active_passengers: int = 0
    completed_passengers: int = 0
    cancelled_passengers: int = 0
    
    # Aggregated timing metrics
    average_wait_time: float = 0.0
    average_in_vehicle_time: float = 0.0
    average_walking_time: float = 0.0
    average_journey_time: float = 0.0
    
    # Aggregated distance metrics
    average_access_distance: float = 0.0
    average_egress_distance: float = 0.0
    average_in_vehicle_distance: float = 0.0
    average_route_deviation: float = 0.0
    
    # Service quality metrics
    service_level_violations: int = 0
    on_time_pickup_rate: float = 0.0
    on_time_dropoff_rate: float = 0.0
    
    # Individual metrics storage
    individual_metrics: Dict[str, IndividualPassengerMetrics] = field(default_factory=dict)
    
    def update_aggregated_metrics(self) -> None:
        """Update all aggregated metrics based on individual metrics"""
        completed_metrics = [m for m in self.individual_metrics.values() 
                           if m.total_journey_time > 0]
        n_completed = len(completed_metrics)
        
        if n_completed == 0:
            return
            
        # Update timing averages
        self.average_wait_time = sum(m.total_wait_time for m in completed_metrics) / n_completed
        self.average_in_vehicle_time = sum(m.total_in_vehicle_time for m in completed_metrics) / n_completed
        self.average_walking_time = sum(m.total_walking_time for m in completed_metrics) / n_completed
        self.average_journey_time = sum(m.total_journey_time for m in completed_metrics) / n_completed
        
        # Update distance averages
        self.average_access_distance = sum(m.access_walking_distance for m in completed_metrics) / n_completed
        self.average_egress_distance = sum(m.egress_distance for m in completed_metrics) / n_completed
        self.average_in_vehicle_distance = sum(m.in_vehicle_distance for m in completed_metrics) / n_completed
        self.average_route_deviation = sum(m.route_deviation_ratio for m in completed_metrics) / n_completed
        
        # Update service quality metrics
        self.service_level_violations = sum(len(m.service_level_violations) for m in self.individual_metrics.values())
        self.on_time_pickup_rate = sum(1 for m in completed_metrics if m.on_time_pickup) / n_completed
        self.on_time_dropoff_rate = sum(1 for m in completed_metrics if m.on_time_dropoff) / n_completed


@dataclass
class RouteMetrics:
    """Metrics tracked for routes"""
    total_routes: int = 0
    active_routes: int = 0
    completed_routes: int = 0
    average_route_length: float = 0.0
    average_stops_per_route: float = 0.0

@dataclass
class IndividualStopMetrics:
    """Metrics tracked for individual stops"""
    # Current state metrics
    current_occupancy: int = 0
    waiting_passengers: int = 0
    vehicles_queued: int = 0
    
    # Cumulative metrics
    total_boardings: int = 0
    total_alightings: int = 0
    total_stop_visits: int = 0
    cumulative_dwell_time: float = 0.0
    total_congestion_events: int = 0
    total_capacity_exceeded_events: int = 0
    
    # Derived metrics
    @property
    def average_dwell_time(self) -> float:
        if self.total_stop_visits > 0:
            return self.cumulative_dwell_time / self.total_stop_visits
        return 0.0
    
    def update_dwell_time(self, new_dwell_time: float) -> None:
        self.cumulative_dwell_time += new_dwell_time
        
    def update_occupancy(self, boarding: bool = False, alighting: bool = False) -> None:
        if boarding:
            self.current_occupancy += 1
            self.total_boardings += 1
            if self.waiting_passengers > 0:
                self.waiting_passengers -= 1
        if alighting:
            self.current_occupancy = max(0, self.current_occupancy - 1)
            self.total_alightings += 1
            
    def record_vehicle_visit(self) -> None:
        self.total_stop_visits += 1
        self.vehicles_queued += 1
        
    def record_vehicle_departure(self) -> None:
        if self.vehicles_queued > 0:
            self.vehicles_queued -= 1
            
    def record_congestion_event(self) -> None:
        self.total_congestion_events += 1
        
    def record_capacity_exceeded(self) -> None:
        self.total_capacity_exceeded_events += 1

@dataclass
class StopMetrics:
    """Aggregated metrics for all stops in the system"""
    # Counts
    total_stops: int = 0
    active_stops: int = 0
    inactive_stops: int = 0
    
    # System-wide aggregates
    total_boardings: int = 0
    total_alightings: int = 0
    total_stop_visits: int = 0
    total_congestion_events: int = 0
    total_capacity_exceeded_events: int = 0
    
    # System-wide averages
    average_dwell_time: float = 0.0
    average_occupancy: float = 0.0
    average_waiting_passengers: float = 0.0
    average_vehicles_queued: float = 0.0
    
    # Current system-wide totals
    current_total_occupancy: int = 0
    current_total_waiting: int = 0
    current_total_queued: int = 0
    
    # Individual metrics storage
    individual_metrics: Dict[str, IndividualStopMetrics] = field(default_factory=dict)
    
    def update_aggregated_metrics(self) -> None:
        """Update all aggregated metrics based on individual metrics"""
        active_metrics = [m for m in self.individual_metrics.values()]
        n_active = len(active_metrics)
        
        if n_active == 0:
            return
            
        # Update totals
        self.total_boardings = sum(m.total_boardings for m in active_metrics)
        self.total_alightings = sum(m.total_alightings for m in active_metrics)
        self.total_stop_visits = sum(m.total_stop_visits for m in active_metrics)
        self.total_congestion_events = sum(m.total_congestion_events for m in active_metrics)
        self.total_capacity_exceeded_events = sum(m.total_capacity_exceeded_events for m in active_metrics)
        
        # Update current totals
        self.current_total_occupancy = sum(m.current_occupancy for m in active_metrics)
        self.current_total_waiting = sum(m.waiting_passengers for m in active_metrics)
        self.current_total_queued = sum(m.vehicles_queued for m in active_metrics)
        
        # Update averages
        total_dwell_time = sum(m.cumulative_dwell_time for m in active_metrics)
        self.average_dwell_time = total_dwell_time / self.total_stop_visits if self.total_stop_visits > 0 else 0.0
        self.average_occupancy = self.current_total_occupancy / n_active
        self.average_waiting_passengers = self.current_total_waiting / n_active
        self.average_vehicles_queued = self.current_total_queued / n_active

@dataclass
class SystemMetrics(ModelBase):
    timestamp: datetime
    operational: OperationalMetrics
    service: ServiceMetrics
    vehicle_metrics: VehicleMetrics
    stop_metrics: StopMetrics
    passenger_metrics: PassengerMetrics
    financial_metrics: Dict[str, float]