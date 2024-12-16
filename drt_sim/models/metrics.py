# drt_sim/models/metrics.py
from dataclasses import dataclass
from typing import Dict
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
class SystemMetrics(ModelBase):
    timestamp: datetime
    operational: OperationalMetrics
    service: ServiceMetrics
    vehicle_metrics: Dict[str, Dict[str, float]]
    stop_metrics: Dict[str, Dict[str, float]]
    passenger_metrics: Dict[str, Dict[str, float]]
    financial_metrics: Dict[str, float]