"""
Feature extraction utilities for user acceptance models.

This module provides common functionality for extracting and normalizing 
features used in user acceptance models.
"""
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
import numpy as np
import logging

from drt_sim.models.request import Request
from drt_sim.models.user import UserProfile
from drt_sim.config.config import Currency

logger = logging.getLogger(__name__)

class FeatureExtractor:
    """
    Feature extractor for user acceptance models.
    
    This class provides methods to extract and normalize features from various
    input formats, ensuring consistency across different acceptance models.
    """
    
    # Default feature registry with metadata
    DEFAULT_FEATURE_REGISTRY = {
        "waiting_time": {
            "description": "Time between request and pickup",
            "unit": "minutes",
            "normalization": 30.0,
            "importance": "high",
            "group": "time"
        },
        "in_vehicle_time": {
            "description": "Duration of the trip",
            "unit": "minutes",
            "normalization": 60.0,
            "importance": "high",
            "group": "time"
        },
        "walking_time_to_origin": {
            "description": "Time to walk to pickup location",
            "unit": "minutes",
            "normalization": 10.0,
            "importance": "high",
            "group": "time"
        },
        "walking_time_from_destination": {
            "description": "Time to walk from dropoff location",
            "unit": "minutes",
            "normalization": 10.0,
            "importance": "high",
            "group": "time"
        },
        "price": {
            "description": "Monetary price of the service",
            "unit": "currency",
            "normalization": 1.0,
            "importance": "high",
            "group": "price"
        },
        "time_of_day": {
            "description": "Hour of the day",
            "unit": "hour",
            "normalization": 24.0,
            "importance": "medium",
            "group": "time"
        },
        "day_of_week": {
            "description": "Day of the week (0-6)",
            "unit": "day",
            "normalization": 7.0,
            "importance": "medium",
            "group": "time"
        },
        "distance_to_pickup": {
            "description": "Distance to pickup location",
            "unit": "km",
            "normalization": 10.0,
            "importance": "medium",
            "group": "spatial"
        },
        "weather_condition": {
            "description": "Weather condition code",
            "unit": "code",
            "normalization": 10.0,
            "importance": "low",
            "group": "environmental"
        },
        "vehicle_capacity": {
            "description": "Available capacity in the vehicle",
            "unit": "seats",
            "normalization": 8.0,
            "importance": "low",
            "group": "quality"
        },
        "historical_acceptance_rate": {
            "description": "User's historical acceptance rate",
            "unit": "ratio",
            "normalization": 1.0,
            "importance": "high",
            "group": "user"
        }
    }
    
    def __init__(
        self,
        feature_registry=None,
        enabled_features=None,
        normalization_overrides=None
    ):
        """
        Initialize the feature extractor.
        
        Args:
            feature_registry: Registry containing feature metadata
            enabled_features: List of features to extract (None = all available)
            normalization_overrides: Custom normalization values
        """
        # Initialize feature registry
        self.feature_registry = self.DEFAULT_FEATURE_REGISTRY.copy()
        if feature_registry:
            self.feature_registry.update(feature_registry)
        
        # Set enabled features
        self.enabled_features = enabled_features or list(self.feature_registry.keys())
        
        # Apply normalization overrides
        if normalization_overrides:
            for feature, value in normalization_overrides.items():
                if feature in self.feature_registry:
                    self.feature_registry[feature]["normalization"] = value
    
    def get_feature_dim(self) -> int:
        """
        Get the dimensionality of the feature vector.
        
        Returns:
            int: Dimension of the feature vector
        """
        return len(self.enabled_features)
    
    def get_feature_names(self) -> List[str]:
        """
        Get the names of enabled features.
        
        Returns:
            List[str]: Names of enabled features
        """
        return self.enabled_features.copy()
    
    def extract_features_dict(
        self,
        features: Dict[str, Any],
        request: Optional[Request] = None,
        user_profile: Optional[UserProfile] = None
    ) -> Dict[str, float]:
        """
        Extract normalized features as a dictionary.
        
        Args:
            features: Dictionary of raw features
            request: The transportation request (optional)
            user_profile: The user's profile (optional)
            
        Returns:
            Dict[str, float]: Dictionary of normalized features
        """
        result = {}
        
        # Process all enabled features
        for feature_name in self.enabled_features:
            # Skip if feature is not in registry
            if feature_name not in self.feature_registry:
                continue
                
            # Get feature metadata
            metadata = self.feature_registry[feature_name]
            normalization = metadata["normalization"]
            
            # Extract waiting time
            if feature_name == "waiting_time":
                if "waiting_time" in features:
                    waiting_time = features["waiting_time"]
                    if isinstance(waiting_time, timedelta):
                        waiting_time = waiting_time.total_seconds() / 60
                    result[feature_name] = min(waiting_time / normalization, 1.0)
                elif "proposed_pickup_time" in features and "request_time" in features:
                    waiting_time = (features["proposed_pickup_time"] - features["request_time"]).total_seconds() / 60
                    result[feature_name] = min(waiting_time / normalization, 1.0)
            
            # Extract in-vehicle time
            elif feature_name == "in_vehicle_time":
                if "in_vehicle_time" in features:
                    in_vehicle_time = features["in_vehicle_time"]
                    if isinstance(in_vehicle_time, timedelta):
                        in_vehicle_time = in_vehicle_time.total_seconds() / 60
                    result[feature_name] = min(in_vehicle_time / normalization, 1.0)
                elif "travel_time" in features:
                    travel_time = features["travel_time"]
                    if isinstance(travel_time, timedelta):
                        travel_time = travel_time.total_seconds() / 60
                    result[feature_name] = min(travel_time / normalization, 1.0)
                elif "proposed_travel_time" in features:
                    travel_time = features["proposed_travel_time"]
                    if isinstance(travel_time, timedelta):
                        travel_time = travel_time.total_seconds() / 60
                    result[feature_name] = min(travel_time / normalization, 1.0)
            
            # Extract walking time to origin
            elif feature_name == "walking_time_to_origin":
                if "walking_time_to_origin" in features:
                    walking_time = features["walking_time_to_origin"]
                    if isinstance(walking_time, timedelta):
                        walking_time = walking_time.total_seconds() / 60
                    result[feature_name] = min(walking_time / normalization, 1.0)
                elif "walking_time_to_pickup" in features:
                    walking_time = features["walking_time_to_pickup"]
                    if isinstance(walking_time, timedelta):
                        walking_time = walking_time.total_seconds() / 60
                    result[feature_name] = min(walking_time / normalization, 1.0)
            
            # Extract walking time from destination
            elif feature_name == "walking_time_from_destination":
                if "walking_time_from_destination" in features:
                    walking_time = features["walking_time_from_destination"]
                    if isinstance(walking_time, timedelta):
                        walking_time = walking_time.total_seconds() / 60
                    result[feature_name] = min(walking_time / normalization, 1.0)
                elif "walking_time_from_dropoff" in features:
                    walking_time = features["walking_time_from_dropoff"]
                    if isinstance(walking_time, timedelta):
                        walking_time = walking_time.total_seconds() / 60
                    result[feature_name] = min(walking_time / normalization, 1.0)
            
            # Extract price
            elif feature_name == "price":
                if "price" in features:
                    # Handle currency-specific price normalization
                    if isinstance(normalization, dict) and request and hasattr(request, 'currency'):
                        try:
                            currency = Currency(request.currency)
                            norm_value = normalization.get(currency.value, normalization.get("USD", 1.0))
                        except Exception as e:
                            logger.error(f"Error getting currency-specific normalization: {e}")
                            norm_value = normalization.get("USD", 1.0)
                    else:
                        norm_value = normalization
                    
                    result[feature_name] = min(features["price"] / norm_value, 1.0)
            
            # Extract time of day
            elif feature_name == "time_of_day":
                if "time_of_day" in features:
                    result[feature_name] = features["time_of_day"] / normalization
                elif "proposed_pickup_time" in features:
                    pickup_time = features["proposed_pickup_time"]
                    hour = pickup_time.hour + pickup_time.minute / 60.0
                    result[feature_name] = hour / normalization
            
            # Extract day of week
            elif feature_name == "day_of_week":
                if "day_of_week" in features:
                    result[feature_name] = features["day_of_week"] / normalization
                elif "proposed_pickup_time" in features:
                    day = features["proposed_pickup_time"].weekday()
                    result[feature_name] = day / normalization
            
            # Extract distance to pickup
            elif feature_name == "distance_to_pickup":
                if "distance_to_pickup" in features:
                    result[feature_name] = min(features["distance_to_pickup"] / normalization, 1.0)
            
            # Extract weather condition
            elif feature_name == "weather_condition":
                if "weather_condition" in features:
                    result[feature_name] = features["weather_condition"] / normalization
            
            # Extract vehicle capacity
            elif feature_name == "vehicle_capacity":
                if "vehicle_capacity" in features:
                    result[feature_name] = min(features["vehicle_capacity"] / normalization, 1.0)
            
            # Extract historical acceptance rate
            elif feature_name == "historical_acceptance_rate":
                if "historical_acceptance_rate" in features:
                    result[feature_name] = features["historical_acceptance_rate"]
                elif user_profile and hasattr(user_profile, 'get_acceptance_rate') and callable(getattr(user_profile, 'get_acceptance_rate')):
                    result[feature_name] = user_profile.get_acceptance_rate()
            
            # For any other features, just use the raw value if available
            elif feature_name in features:
                # Try to normalize if normalization value is provided
                if normalization and normalization > 0:
                    result[feature_name] = min(features[feature_name] / normalization, 1.0)
                else:
                    result[feature_name] = features[feature_name]
        
        return result
    
    def extract_features_vector(
        self,
        features: Dict[str, Any],
        request: Optional[Request] = None,
        user_profile: Optional[UserProfile] = None
    ) -> np.ndarray:
        """
        Extract normalized features as a numpy vector.
        
        Args:
            features: Dictionary of raw features
            request: The transportation request (optional)
            user_profile: The user's profile (optional)
            
        Returns:
            np.ndarray: Vector of normalized features
        """
        # Get features as dictionary
        feature_dict = self.extract_features_dict(features, request, user_profile)
        
        # Convert to vector in the correct order
        feature_vector = np.zeros(len(self.enabled_features))
        for i, feature_name in enumerate(self.enabled_features):
            feature_vector[i] = feature_dict.get(feature_name, 0.0)
        
        return feature_vector
    
    def get_feature_groups(self) -> Dict[str, List[str]]:
        """
        Get features grouped by their categories.
        
        Returns:
            Dict[str, List[str]]: Dictionary mapping group names to feature lists
        """
        groups = {}
        
        for feature_name in self.enabled_features:
            if feature_name in self.feature_registry:
                metadata = self.feature_registry[feature_name]
                group = metadata.get("group", "other")
                
                if group not in groups:
                    groups[group] = []
                
                groups[group].append(feature_name)
        
        return groups
    
    def get_feature_metadata(self, feature_name: str) -> Dict[str, Any]:
        """
        Get metadata for a specific feature.
        
        Args:
            feature_name: Name of the feature
            
        Returns:
            Dict[str, Any]: Feature metadata
        """
        if feature_name in self.feature_registry:
            return self.feature_registry[feature_name].copy()
        else:
            return {}
    
    def add_custom_feature(
        self,
        name: str,
        description: str,
        unit: str,
        normalization: float,
        importance: str = "medium",
        group: str = "custom"
    ) -> None:
        """
        Add a custom feature to the registry.
        
        Args:
            name: Feature name
            description: Feature description
            unit: Feature unit
            normalization: Normalization value
            importance: Importance level (low, medium, high)
            group: Feature group
        """
        self.feature_registry[name] = {
            "description": description,
            "unit": unit,
            "normalization": normalization,
            "importance": importance,
            "group": group
        }
        
        # Add to enabled features if not already present
        if name not in self.enabled_features:
            self.enabled_features.append(name)
            
        logger.info(f"Added custom feature: {name}")