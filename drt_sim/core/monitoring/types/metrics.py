from enum import Enum
from typing import List, Dict, Optional, Any
from datetime import datetime
from pydantic import BaseModel, Field
import numpy as np

class MetricName(Enum):
    PASSENGER_WAIT_TIME = 'passenger.wait_time'
    PASSENGER_RIDE_TIME = 'passenger.ride_time'
    PASSENGER_WALK_TIME_TO_ORIGIN_STOP = 'passenger.walk_time_to_origin_stop'
    PASSENGER_WALK_TIME_FROM_DESTINATION_STOP = 'passenger.walk_time_from_destination_stop'
    PASSENGER_TOTAL_JOURNEY_TIME = 'passenger.total_journey_time'
    PASSENGER_NO_SHOW = 'passenger.no_show'

    REQUEST_RECEIVED = 'request.received'
    REQUEST_ASSIGNED = 'request.assigned'
    REQUEST_REJECTED = 'request.rejected'

    VEHICLE_UTILIZATION = 'vehicle.utilization'
    VEHICLE_WAIT_TIME = 'vehicle.wait_time'
    VEHICLE_OCCUPIED_DISTANCE = 'vehicle.occupied_distance'
    VEHICLE_EMPTY_DISTANCE = 'vehicle.empty_distance'
    VEHICLE_DWELL_TIME = 'vehicle.dwell_time'
    VEHICLE_STOPS_SERVED = 'vehicle.stops_served'
    VEHICLE_PASSENGERS_SERVED = 'vehicle.passengers_served'

    STOP_OCCUPANCY = 'stop.occupancy'

    SIMULATION_STEP_DURATION = 'simulation.step_duration'
    SIMULATION_TOTAL_STEPS = 'simulation.total_steps'
    SIMULATION_EVENT_PROCESSING_TIME = 'simulation.event_processing_time'
    SIMULATION_REPLICATION_TIME = 'simulation.replication_time'

    MATCHING_SUCCESS_RATE = 'matching.success_rate'
    MATCHING_FAILURE_REASON = 'matching.failure_reason'

    ROUTE_COMPLETION_TIME = 'route.completion_time'
    ROUTE_DEVIATION = 'route.deviation'

    SERVICE_VIOLATIONS = 'service.violations'
    SERVICE_ON_TIME_RATE = 'service.on_time_rate'
    SERVICE_CAPACITY_UTILIZATION = 'service.capacity_utilization'
    
    # User acceptance metrics
    USER_ACCEPTANCE_PROBABILITY = 'user.acceptance_probability'
    USER_ACCEPTANCE_RATE = 'user.acceptance_rate'
    USER_REJECTION_REASON = 'user.rejection_reason'
    USER_FEATURE_IMPORTANCE = 'user.feature_importance'

class MetricDefinition:
    def __init__(self, name: str, description: str, metric_type: str, unit: str,
                 required_context: List[str], aggregations: List[str],
                 visualizations: Optional[Dict[str, bool]] = None):
        self.name = name
        self.description = description
        self.metric_type = metric_type  # "event", "snapshot", "aggregate"
        self.unit = unit
        self.required_context = required_context
        self.aggregations = aggregations
        # Default visualization flags if none provided
        self.visualizations = visualizations or {
            'time_series': True,
            'distribution': True,
            'summary': True
        }

    def __repr__(self):
        return f"MetricDefinition(name={self.name})"

class MetricPoint(BaseModel):
    """Pydantic model for metric validation."""
    name: str
    value: float
    timestamp: datetime = datetime.now()
    tags: Dict[str, Any] = Field(default_factory=dict)
    schema_version: str = "1.0.0"  # Track schema version for future changes

    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }
        
    @property
    def metric_definition(self) -> Optional[MetricDefinition]:
        """Get the metric definition for this point."""
        from drt_sim.core.monitoring.metrics.registry import metric_registry
        return metric_registry.get(self.name)
        
    def validate_context(self) -> bool:
        """Validate that all required context fields are present in tags."""
        definition = self.metric_definition
        if not definition:
            return False
            
        missing_context = [
            key for key in definition.required_context
            if key not in self.tags
        ]
        return len(missing_context) == 0

    @property
    def should_plot_time_series(self) -> bool:
        """Check if this metric should have a time series plot."""
        return self.metric_definition.visualizations.get('time_series', True) if self.metric_definition else True

    @property
    def should_plot_distribution(self) -> bool:
        """Check if this metric should have a distribution plot."""
        return self.metric_definition.visualizations.get('distribution', True) if self.metric_definition else True

    @property
    def should_plot_summary(self) -> bool:
        """Check if this metric should have a summary plot."""
        return self.metric_definition.visualizations.get('summary', True) if self.metric_definition else True

class IncrementalStats:
    """Helper class for computing running statistics."""
    def __init__(self):
        self.count = 0
        self.mean = 0.0
        self._m2 = 0.0  # Used for computing variance/std
        self.min_val = float('inf')
        self.max_val = float('-inf')
        self.sum = 0.0
        
    def update(self, value: float):
        """Update running statistics with a new value."""
        self.count += 1
        delta = value - self.mean
        self.mean += delta / self.count
        delta2 = value - self.mean
        self._m2 += delta * delta2
        
        self.min_val = min(self.min_val, value)
        self.max_val = max(self.max_val, value)
        self.sum += value
        
    def merge(self, other: 'IncrementalStats'):
        """Merge another IncrementalStats object into this one."""
        if other.count == 0:
            return
            
        combined_count = self.count + other.count
        delta = other.mean - self.mean
        self.mean = (self.mean * self.count + other.mean * other.count) / combined_count
        self._m2 = (self._m2 + other._m2 + 
                   (delta * delta * self.count * other.count) / combined_count)
        
        self.min_val = min(self.min_val, other.min_val)
        self.max_val = max(self.max_val, other.max_val)
        self.sum += other.sum
        self.count = combined_count
        
    @property
    def std(self) -> float:
        """Compute standard deviation."""
        return np.sqrt(self._m2 / self.count) if self.count > 0 else 0.0
        
    def to_dict(self) -> Dict[str, float]:
        """Convert statistics to a dictionary."""
        return {
            'mean': self.mean,
            'min': self.min_val if self.count > 0 else None,
            'max': self.max_val if self.count > 0 else None,
            'count': self.count,
            'sum': self.sum,
            'std': self.std
        }