'''
Rejection models for the DRT system.

This module provides models for handling request rejections in the DRT system.
'''
from dataclasses import dataclass
from enum import Enum, auto
from typing import Dict, Any, Optional
from datetime import datetime

class RejectionReason(Enum):
    """
    Reasons for rejecting a transportation request.
    """
    UNKNOWN = "unknown"
    VALIDATION_FAILED = "validation_failed"
    NO_VEHICLE_AVAILABLE = "no_vehicle_available"
    CAPACITY_EXCEEDED = "capacity_exceeded"
    SERVICE_AREA_VIOLATION = "service_area_violation"
    TIME_CONSTRAINT_VIOLATION = "time_constraint_violation"
    SYSTEM_OVERLOAD = "system_overload"
    USER_REJECTED = "user_rejected"
    OPTIMIZATION_FAILED = "optimization_failed"
    TECHNICAL_ERROR = "technical_error"
    DUPLICATE_REQUEST = "duplicate_request"
    INVALID_PARAMETERS = "invalid_parameters"
    SERVICE_UNAVAILABLE = "service_unavailable"
    
    # Time-based constraints
    TIME_WINDOW_CONSTRAINT = "time_window_constraint"
    VEHICLE_ACCESS_TIME_CONSTRAINT = "vehicle_access_time_constraint"
    PASSENGER_WAIT_TIME_CONSTRAINT = "passenger_wait_time_constraint"
    RIDE_TIME_CONSTRAINT = "ride_time_constraint"
    DETOUR_CONSTRAINT = "detour_constraint"
    TOTAL_JOURNEY_TIME_CONSTRAINT = "total_journey_time_constraint"
    
    # Vehicle constraints
    CAPACITY_CONSTRAINT = "capacity_constraint"
    VEHICLE_RANGE_CONSTRAINT = "vehicle_range_constraint"
    VEHICLE_SHIFT_END_CONSTRAINT = "vehicle_shift_end_constraint"
    
    # Passenger constraints
    PASSENGER_WALK_TIME_CONSTRAINT = "passenger_walk_time_constraint"
    PASSENGER_ACCESSIBILITY_CONSTRAINT = "passenger_accessibility_constraint"
    
    # Cost constraints
    RESERVE_PRICE_CONSTRAINT = "reserve_price_constraint"
    OPERATIONAL_COST_CONSTRAINT = "operational_cost_constraint"
    
    # System constraints
    SYSTEM_CAPACITY_CONSTRAINT = "system_capacity_constraint"
    GEOGRAPHIC_CONSTRAINT = "geographic_constraint"

@dataclass
class RejectionMetadata:
    """
    Metadata for a rejected request.
    
    This class stores information about why a request was rejected,
    when it was rejected, and any additional details.
    """
    reason: RejectionReason
    details: str
    timestamp: datetime
    additional_data: Optional[Dict[str, Any]] = None
    
    def __post_init__(self):
        """Ensure reason is a RejectionReason enum."""
        if isinstance(self.reason, str):
            try:
                self.reason = RejectionReason(self.reason)
            except ValueError:
                self.reason = RejectionReason.UNKNOWN
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "reason": self.reason.value,
            "details": self.details,
            "timestamp": self.timestamp.isoformat(),
            "additional_data": self.additional_data
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'RejectionMetadata':
        """Create from dictionary representation."""
        timestamp = data.get("timestamp")
        if isinstance(timestamp, str):
            timestamp = datetime.fromisoformat(timestamp)
        
        return cls(
            reason=data.get("reason", RejectionReason.UNKNOWN),
            details=data.get("details", ""),
            timestamp=timestamp or datetime.now(),
            additional_data=data.get("additional_data")
        ) 