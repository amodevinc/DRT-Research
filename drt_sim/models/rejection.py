from dataclasses import dataclass
from enum import Enum
from typing import Dict, Any, Optional
from datetime import datetime

class RejectionReason(Enum):
    """Enum for request rejection reasons"""
    NO_VEHICLES_AVAILABLE = "no_vehicles_available"
    NO_FEASIBLE_INSERTION = "no_feasible_insertion"
    TECHNICAL_ERROR = "technical_error"
    
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
    """Metadata for request rejection"""
    reason: RejectionReason
    timestamp: str
    stage: str
    details: Dict[str, Any]
    
    def __post_init__(self):
        """Validate and process rejection metadata"""
        if not isinstance(self.reason, RejectionReason):
            raise ValueError("reason must be a RejectionReason enum")
        if not isinstance(self.timestamp, str):
            raise ValueError("timestamp must be a string")
        if not isinstance(self.stage, str):
            raise ValueError("stage must be a string")
        if not isinstance(self.details, dict):
            raise ValueError("details must be a dictionary")
            
        # Ensure timestamp is valid ISO format
        try:
            datetime.fromisoformat(self.timestamp)
        except ValueError:
            raise ValueError("timestamp must be in ISO format")
            
        # Add standard fields to details if not present
        if "evaluated_vehicles" not in self.details:
            self.details["evaluated_vehicles"] = 0
        if "rejection_counts" not in self.details:
            self.details["rejection_counts"] = {}
        if "constraint_violations" not in self.details:
            self.details["constraint_violations"] = {}
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert metadata to dictionary format"""
        return {
            "reason": self.reason.value,
            "timestamp": self.timestamp,
            "stage": self.stage,
            "details": self.details
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'RejectionMetadata':
        """Create metadata from dictionary"""
        return cls(
            reason=RejectionReason(data["reason"]),
            timestamp=data["timestamp"],
            stage=data["stage"],
            details=data["details"]
        ) 