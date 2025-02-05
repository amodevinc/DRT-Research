from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, Optional
from drt_sim.models.base import ModelBase
from drt_sim.models.location import Location
class UserPriorityLevel(Enum):
    """Priority levels for users"""
    STANDARD = "standard"
    PREMIUM = "premium"
    VIP = "vip"
    ACCESSIBILITY = "accessibility"  # For users with special needs
    CORPORATE = "corporate"         # For business accounts

class ServicePreference(Enum):
    """Service preferences that influence weights"""
    SPEED = "speed"                # Prefers faster service
    COMFORT = "comfort"            # Prefers more comfortable rides
    ECONOMY = "economy"            # Prefers lower cost
    RELIABILITY = "reliability"    # Prefers reliable arrival times
    DIRECT = "direct"              # Prefers minimal detours

@dataclass
class UserProfile:
    """User profile containing service preferences and historical data"""
    id: str
    priority_level: UserPriorityLevel = UserPriorityLevel.STANDARD
    service_preference: ServicePreference = ServicePreference.SPEED
    mobility_needs: bool = False
    frequent_locations: Dict[str, Location] = field(default_factory=dict)
    
    # Historical metrics
    total_trips: int = 0
    avg_rating: float = 5.0
    cancellation_rate: float = 0.0
    subscription_status: bool = False
    
    # Time-based preferences
    preferred_pickup_buffer: int = 300  # seconds
    preferred_dropoff_buffer: int = 300  # seconds

    
    def to_dict(self) -> Dict:
        return {
            'id': self.id,
            'priority_level': self.priority_level.value,
            'service_preference': self.service_preference.value,
            'mobility_needs': self.mobility_needs,
            'total_trips': self.total_trips,
            'avg_rating': self.avg_rating,
            'cancellation_rate': self.cancellation_rate,
            'subscription_status': self.subscription_status,
            'preferred_pickup_buffer': self.preferred_pickup_buffer,
            'preferred_dropoff_buffer': self.preferred_dropoff_buffer
        }
    def __init__(self, id: str):
        self.id = id

@dataclass
class WeightAdjuster:
    """Handles weight adjustments based on user profiles"""
    
    # Base weight multipliers for different priority levels
    priority_multipliers: Dict[UserPriorityLevel, float] = field(default_factory=lambda: {
        UserPriorityLevel.STANDARD: 1.0,
        UserPriorityLevel.PREMIUM: 1.2,
        UserPriorityLevel.VIP: 1.5,
        UserPriorityLevel.ACCESSIBILITY: 1.3,
        UserPriorityLevel.CORPORATE: 1.4
    })
    
    # Weight adjustments for different service preferences
    preference_adjustments: Dict[ServicePreference, Dict[str, float]] = field(default_factory=lambda: {
        ServicePreference.SPEED: {
            "waiting_time": 1.3,
            "detour_time": 1.2,
            "delay": 1.1,
            "distance": 0.8
        },
        ServicePreference.COMFORT: {
            "waiting_time": 0.9,
            "detour_time": 1.3,
            "delay": 1.1,
            "distance": 0.9
        },
        ServicePreference.ECONOMY: {
            "waiting_time": 0.8,
            "detour_time": 0.9,
            "delay": 0.9,
            "distance": 1.3
        },
        ServicePreference.RELIABILITY: {
            "waiting_time": 1.1,
            "detour_time": 1.0,
            "delay": 1.4,
            "distance": 0.8
        },
        ServicePreference.DIRECT: {
            "waiting_time": 0.9,
            "detour_time": 1.5,
            "delay": 1.1,
            "distance": 0.8
        }
    })
    
    def adjust_weights(
        self,
        base_weights: Dict[str, float],
        user_profile: UserProfile,
        request_context: Optional[Dict] = None
    ) -> Dict[str, float]:
        """
        Adjust weights based on user profile and request context
        
        Args:
            base_weights: Original weight configuration
            user_profile: User's profile with preferences
            request_context: Additional context about the request (optional)
            
        Returns:
            Dict[str, float]: Adjusted weights
        """
        adjusted_weights = base_weights.copy()
        
        # Apply priority level multiplier
        priority_mult = self.priority_multipliers[user_profile.priority_level]
        adjusted_weights = {k: v * priority_mult for k, v in adjusted_weights.items()}
        
        # Apply service preference adjustments
        pref_adjustments = self.preference_adjustments[user_profile.service_preference]
        for factor, adjustment in pref_adjustments.items():
            if factor in adjusted_weights:
                adjusted_weights[factor] *= adjustment
        
        # Special adjustments for mobility needs
        if user_profile.mobility_needs:
            adjusted_weights["waiting_time"] *= 1.2
            adjusted_weights["detour_time"] *= 0.8
        
        # Loyalty adjustments based on total trips
        if user_profile.total_trips > 100:
            loyalty_mult = min(1.3, 1 + (user_profile.total_trips / 1000))
            adjusted_weights = {k: v * loyalty_mult for k, v in adjusted_weights.items()}
        
        # Normalize weights to sum to 1
        weight_sum = sum(adjusted_weights.values())
        adjusted_weights = {k: v / weight_sum for k, v in adjusted_weights.items()}
        
        return adjusted_weights