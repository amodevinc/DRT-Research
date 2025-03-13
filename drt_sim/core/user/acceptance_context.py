"""
Context for user acceptance models.

This module provides a context class that encapsulates all information
needed by user acceptance models to make decisions.
"""
from typing import Dict, Any, Optional
from datetime import datetime, timedelta

from drt_sim.models.request import Request
from drt_sim.models.user import UserProfile

class AcceptanceContext:
    """
    Context for user acceptance decisions.
    
    This class encapsulates all information needed by user acceptance models
    to calculate acceptance probability and make decisions.
    """
    
    def __init__(
        self,
        features: Dict[str, Any],
        request: Optional[Request] = None,
        user_profile: Optional[UserProfile] = None
    ):
        """
        Initialize acceptance context.
        
        Args:
            features: Dictionary of features
            request: The transportation request
            user_profile: The user's profile
        """
        self.features = features
        self.request = request
        self.user_profile = user_profile
    
    @classmethod
    def from_assignment(
        cls,
        request: Request,
        walking_time_to_origin: float,
        waiting_time: float,
        in_vehicle_time: float,
        walking_time_from_destination: float,
        cost: Optional[float] = None,
        user_profile: Optional[UserProfile] = None,
        additional_attributes: Optional[Dict[str, Any]] = None
    ):
        """
        Create context from a potential assignment.
        
        Args:
            request: The transportation request
            walking_time_to_origin: Time to walk to the pickup point (minutes)
            waiting_time: Time to wait for the vehicle (minutes)
            in_vehicle_time: Time spent in the vehicle (minutes)
            walking_time_from_destination: Time to walk from drop-off to destination (minutes)
            cost: The cost of the service (if applicable)
            user_profile: The user's profile
            additional_attributes: Additional assignment attributes
            
        Returns:
            AcceptanceContext: Context for the assignment
        """
        # Create features dictionary with the main assignment features
        features = {
            "walking_time_to_origin": walking_time_to_origin,
            "waiting_time": waiting_time,
            "in_vehicle_time": in_vehicle_time,
            "walking_time_from_destination": walking_time_from_destination
        }
        
        # Add cost if available
        if cost is not None:
            features["cost"] = cost
        
        # Calculate total trip time
        features["total_trip_time"] = (
            walking_time_to_origin + waiting_time + in_vehicle_time + walking_time_from_destination
        )
        
        # Add additional attributes
        if additional_attributes:
            features.update(additional_attributes)
        
        # Create context
        return cls(
            features=features,
            request=request,
            user_profile=user_profile
        )
    
    def clone(self):
        """
        Create a copy of this context.
        
        Returns:
            AcceptanceContext: Copy of the context
        """
        return AcceptanceContext(
            features=self.features.copy(),
            request=self.request,
            user_profile=self.user_profile
        )
    
    def add_feature(self, name: str, value: Any) -> None:
        """
        Add a feature to the context.
        
        Args:
            name: Feature name
            value: Feature value
        """
        self.features[name] = value
    
    def add_features(self, features: Dict[str, Any]) -> None:
        """
        Add multiple features to the context.
        
        Args:
            features: Dictionary of features to add
        """
        self.features.update(features)
    
    def get_feature(self, name: str, default: Any = None) -> Any:
        """
        Get a feature from the context.
        
        Args:
            name: Feature name
            default: Default value if feature not found
            
        Returns:
            Any: Feature value
        """
        return self.features.get(name, default)
    
    def has_feature(self, name: str) -> bool:
        """
        Check if the context has a feature.
        
        Args:
            name: Feature name
            
        Returns:
            bool: True if feature exists, False otherwise
        """
        return name in self.features