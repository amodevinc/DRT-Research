"""
Context for user acceptance models.

This module provides a context class that encapsulates all information
needed by user acceptance models to make decisions.
"""
from typing import Dict, Any, Optional
import logging
from datetime import datetime, timedelta

from drt_sim.models.request import Request
from drt_sim.models.user import UserProfile

# Configure a logger for this module
logger = logging.getLogger(__name__)

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
        logger.debug(f"AcceptanceContext initialized with {len(features)} features")
        if request:
            logger.debug(f"Request ID: {request.id if hasattr(request, 'id') else 'unknown'}")
    
    @classmethod
    def from_assignment(
        cls,
        request: Request,
        service_attributes: Dict[str, Any],
        user_profile: Optional[UserProfile] = None
    ):
        """
        Create context from a potential assignment.
        
        Args:
            request: The transportation request
            service_attributes: Dictionary of service attributes
            user_profile: The user's profile
            
        Returns:
            AcceptanceContext: Context for the assignment
        """
        logger.info(f"Creating AcceptanceContext from assignment for request {request.id if hasattr(request, 'id') else 'unknown'}")
        
        # Create features dictionary with the main assignment features
        features = {
            "walking_time_to_origin": service_attributes.get("walking_time_to_origin", 0),
            "waiting_time": service_attributes.get("waiting_time", 0),
            "in_vehicle_time": service_attributes.get("in_vehicle_time", 0),
            "walking_time_from_destination": service_attributes.get("walking_time_from_destination", 0)
        }
        
        # Add cost if available
        if service_attributes.get("cost") is not None:
            features["cost"] = service_attributes.get("cost")
            logger.debug(f"Cost feature added: {features['cost']}")
        
        # Calculate total trip time
        features["total_trip_time"] = (
            service_attributes.get("walking_time_to_origin", 0) +
            service_attributes.get("waiting_time", 0) +
            service_attributes.get("in_vehicle_time", 0) +
            service_attributes.get("walking_time_from_destination", 0)
        )
        logger.debug(f"Total trip time calculated: {features['total_trip_time']}")
        
        # Add additional attributes
        if service_attributes:
            additional_attrs = {k: v for k, v in service_attributes.items() 
                               if k not in ["walking_time_to_origin", "waiting_time", 
                                           "in_vehicle_time", "walking_time_from_destination", "cost"]}
            features.update(additional_attrs)
            if additional_attrs:
                logger.debug(f"Added {len(additional_attrs)} additional service attributes")
        
        # Create context
        context = cls(
            features=features,
            request=request,
            user_profile=user_profile
        )
        logger.info(f"AcceptanceContext created successfully with {len(features)} features")
        return context
    
    def clone(self):
        """
        Create a copy of this context.
        
        Returns:
            AcceptanceContext: Copy of the context
        """
        logger.debug("Cloning AcceptanceContext")
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
        logger.debug(f"Feature added: {name} = {value}")
    
    def add_features(self, features: Dict[str, Any]) -> None:
        """
        Add multiple features to the context.
        
        Args:
            features: Dictionary of features to add
        """
        self.features.update(features)
        logger.debug(f"Added {len(features)} features: {', '.join(features.keys())}")
    
    def get_feature(self, name: str, default: Any = None) -> Any:
        """
        Get a feature from the context.
        
        Args:
            name: Feature name
            default: Default value if feature not found
            
        Returns:
            Any: Feature value
        """
        value = self.features.get(name, default)
        if name not in self.features:
            logger.debug(f"Feature '{name}' not found, returning default: {default}")
        return value
    
    def has_feature(self, name: str) -> bool:
        """
        Check if the context has a feature.
        
        Args:
            name: Feature name
            
        Returns:
            bool: True if feature exists, False otherwise
        """
        exists = name in self.features
        logger.debug(f"Feature check: '{name}' exists = {exists}")
        return exists
    
    def __str__(self) -> str:
        """
        String representation of the context.
        
        Returns:
            str: String representation
        """
        request_id = self.request.id if self.request and hasattr(self.request, 'id') else 'unknown'
        user_id = self.user_profile.id if self.user_profile and hasattr(self.user_profile, 'id') else 'unknown'
        return f"AcceptanceContext(request_id={request_id}, user_id={user_id}, features={len(self.features)})"
    
    def __repr__(self) -> str:
        """
        Developer representation of the context.
        
        Returns:
            str: Detailed representation
        """
        return self.__str__()