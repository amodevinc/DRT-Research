"""
Default user acceptance model.

This module provides a simple rule-based model for user acceptance decisions,
based on thresholds and user profiles.
"""
from typing import Dict, Any, Optional, List, Tuple
import random
import logging
import numpy as np

from drt_sim.algorithms.base_interfaces.user_acceptance_base import UserAcceptanceModel
from drt_sim.core.user.acceptance_context import AcceptanceContext
from drt_sim.core.user.feature_extractor import FeatureExtractor
from drt_sim.core.user.feature_provider import FeatureProviderRegistry
logger = logging.getLogger(__name__)

class DefaultModel(UserAcceptanceModel):
    """
    Default user acceptance model.
    
    This class implements a simple rule-based model for user acceptance,
    based on thresholds and user profiles.
    """
    
    def __init__(self, feature_extractor: Optional[FeatureExtractor] = None, feature_provider_registry: Optional[FeatureProviderRegistry] = None, **kwargs):
        """
        Initialize the default model.
        
        Args:
            feature_extractor: Feature extractor to use
            **kwargs: Additional parameters
        """
        super().__init__(feature_extractor, **kwargs)
        self.feature_provider_registry = feature_provider_registry
        
        # Default thresholds (these match the typical attributes in a user profile)
        self.default_thresholds = {
            "max_walking_time_to_origin": 10.0,     # minutes
            "max_waiting_time": 15.0,               # minutes
            "max_in_vehicle_time": 45.0,            # minutes
            "max_walking_time_from_destination": 10.0,  # minutes
            "max_cost": 50.0,                       # currency units
            "max_acceptable_delay": 15.0            # minutes
        }
        
        # Default weight importance (negative values indicate costs)
        self.default_weights = {
            "walking_time_to_origin": -0.2,
            "waiting_time": -0.3,
            "in_vehicle_time": -0.3,
            "walking_time_from_destination": -0.2,
            "cost": -0.3,
            "time_of_day": 0.0,
            "day_of_week": 0.0,
            "distance_to_pickup": -0.1
        }
        
        # Apply configuration if provided
        if 'config' in kwargs:
            self.configure(kwargs['config'])
    
    def calculate_acceptance_probability(self, context: AcceptanceContext) -> float:
        """
        Calculate probability of user accepting a proposed service.
        
        Args:
            context: Context containing request, features, and user profile
            
        Returns:
            float: Probability of acceptance (0.0 to 1.0)
        """
        # Extract features
        features = self.feature_extractor.extract_features_dict(
            context.features,
            context.request,
            context.user_profile
        )
        # Enrich features using the provider registry if available
        if self.feature_provider_registry is not None:
            # Create a context dict for the providers
            provider_context = {
                "features": features.copy(),
                "user_profile": context.user_profile
            }
            
            # Get additional features from providers
            additional_features = self.feature_provider_registry.get_features(
                context.request, 
                provider_context
            )
            
            # Update features with additional ones
            features.update(additional_features)
        
        # Get user profile
        user_profile = context.user_profile
        
        # Base acceptance probability
        base_probability = 0.5
        
        # Apply threshold-based adjustments
        thresholds = self._get_thresholds(user_profile)
        probability = self._apply_thresholds(features, thresholds, base_probability)
        
        # Apply preference-based adjustments
        probability = self._apply_preferences(features, user_profile, probability)
        
        # Apply historical behavior adjustment
        probability = self._apply_historical_behavior(user_profile, probability)
        
        # Clip probability to valid range
        return min(max(probability, 0.01), 0.99)
    
    def _get_thresholds(self, user_profile) -> Dict[str, float]:
        """
        Get thresholds for the user.
        
        Args:
            user_profile: User profile
            
        Returns:
            Dict[str, float]: Dictionary of thresholds
        """
        thresholds = self.default_thresholds.copy()
        
        if user_profile:
            # Override with user-specific thresholds from profile
            for threshold_name in self.default_thresholds.keys():
                if hasattr(user_profile, threshold_name):
                    thresholds[threshold_name] = getattr(user_profile, threshold_name)
        
        return thresholds
    
    def _apply_thresholds(
        self, 
        features: Dict[str, float], 
        thresholds: Dict[str, float],
        probability: float
    ) -> float:
        """
        Apply thresholds to adjust probability.
        
        Args:
            features: Feature dictionary
            thresholds: Threshold dictionary
            probability: Base probability
            
        Returns:
            float: Adjusted probability
        """
        # Map of feature names to their corresponding threshold names
        threshold_map = {
            "walking_time_to_origin": "max_walking_time_to_origin",
            "waiting_time": "max_waiting_time",
            "in_vehicle_time": "max_in_vehicle_time",
            "walking_time_from_destination": "max_walking_time_from_destination",
            "cost": "max_cost"
        }
        
        # Check each feature against its threshold
        for feature_name, threshold_name in threshold_map.items():
            if feature_name in features and threshold_name in thresholds:
                # Get unnormalized value if it was normalized by feature extractor
                if hasattr(self.feature_extractor, 'feature_registry') and feature_name in self.feature_extractor.feature_registry:
                    normalization = self.feature_extractor.feature_registry[feature_name]['normalization']
                    unnormalized = features[feature_name] * normalization
                else:
                    unnormalized = features[feature_name]
                
                # Check if value exceeds threshold
                threshold = thresholds[threshold_name]
                if unnormalized > threshold:
                    # Sharp decrease in probability if exceeds threshold
                    ratio = unnormalized / threshold
                    # More penalty for larger exceedance
                    factor = 1.0 - min((ratio - 1.0) * 0.5, 0.9)
                    probability *= factor
        
        # Check for total trip time vs. acceptable delay
        if "total_trip_time" in features and "max_acceptable_delay" in thresholds:
            direct_time = features.get("direct_time", features["total_trip_time"] * 0.7)  # Estimate direct time if not provided
            delay = features["total_trip_time"] - direct_time
            
            if delay > thresholds["max_acceptable_delay"]:
                # Decrease probability for excessive delay
                ratio = delay / thresholds["max_acceptable_delay"]
                factor = 1.0 - min((ratio - 1.0) * 0.5, 0.9)
                probability *= factor
        
        return probability
    
    def _apply_preferences(
        self,
        features: Dict[str, float],
        user_profile,
        probability: float
    ) -> float:
        """
        Apply user preferences to adjust probability.
        
        Args:
            features: Feature dictionary
            user_profile: User profile
            probability: Current probability
            
        Returns:
            float: Adjusted probability
        """
        if not user_profile:
            return probability
        
        # Get weights from user profile if available, otherwise use defaults
        weights = self.default_weights.copy()
        
        if hasattr(user_profile, 'weights') and isinstance(user_profile.weights, dict):
            weights.update(user_profile.weights)
        
        # Calculate utility score based on weights and features
        utility = 0.0
        
        for feature_name, value in features.items():
            if feature_name in weights:
                utility += weights[feature_name] * value
        
        # Apply service preference adjustments
        if hasattr(user_profile, 'service_preference'):
            preference = user_profile.service_preference
            
            if preference == 'speed':
                # Speed-focused users care more about time
                if "waiting_time" in features and features["waiting_time"] > 0.5:
                    probability *= 0.8  # Decrease for long waits
                
                if "in_vehicle_time" in features and features["in_vehicle_time"] > 0.6:
                    probability *= 0.7  # Decrease for long rides
                
                # But they might accept longer walks to save time
                if "walking_time_to_origin" in features and "waiting_time" in features:
                    # Accept more walking if it reduces waiting
                    if features["walking_time_to_origin"] < features["waiting_time"]:
                        probability *= 1.1
            
            elif preference == 'comfort':
                # Comfort-focused users care more about minimal walking and smoother rides
                if "walking_time_to_origin" in features and features["walking_time_to_origin"] > 0.4:
                    probability *= 0.8  # Decrease for long walks
                
                if "walking_time_from_destination" in features and features["walking_time_from_destination"] > 0.4:
                    probability *= 0.8  # Decrease for long walks
                
                # Comfort users might accept longer waits for better service
                if "waiting_time" in features and features["waiting_time"] < 0.3:
                    probability *= 1.1  # Increase for short waits
            
            elif preference == 'cost':
                # Cost-focused users care more about price
                if "cost" in features:
                    if features["cost"] < 0.3:
                        probability *= 1.3  # Increase for cheap services
                    elif features["cost"] > 0.7:
                        probability *= 0.7  # Decrease for expensive services
                
                # Cost users might accept longer trips if they're cheaper
                if "total_trip_time" in features and "cost" in features:
                    if features["total_trip_time"] > 0.7 and features["cost"] < 0.4:
                        probability *= 1.1  # Increase acceptance for long but cheap trips
        
        # Apply utility adjustment (logistic function)
        utility_factor = 1.0 / (1.0 + np.exp(-utility))
        
        # Blend base probability with utility factor
        probability = 0.3 * probability + 0.7 * utility_factor
        
        return probability
    
    def _apply_historical_behavior(self, user_profile, probability: float) -> float:
        """
        Apply historical behavior to adjust probability.
        
        Args:
            user_profile: User profile
            probability: Current probability
            
        Returns:
            float: Adjusted probability
        """
        if not user_profile:
            return probability
        
        # Historical acceptance rate
        if hasattr(user_profile, 'historical_acceptance_rate'):
            # Blend with historical acceptance rate (strong influence)
            historical_rate = user_profile.historical_acceptance_rate
            probability = 0.6 * probability + 0.4 * historical_rate
        
        # Experience adjustment
        if hasattr(user_profile, 'historical_trips'):
            trips = user_profile.historical_trips
            
            if trips < 5:
                # New users tend to accept more
                probability = probability * 0.8 + 0.2
            elif trips > 50:
                # Experienced users are more selective
                probability *= 0.9
        
        # Rating-based adjustment
        if hasattr(user_profile, 'historical_ratings') and user_profile.historical_ratings:
            avg_rating = sum(user_profile.historical_ratings) / len(user_profile.historical_ratings)
            
            # Users with high ratings may be more selective
            if avg_rating > 4.0:
                probability *= 0.95
            # Users with low ratings may be less selective
            elif avg_rating < 3.0:
                probability *= 1.05
        
        return probability
    
    def update_model(self, context: AcceptanceContext, accepted: bool) -> None:
        """
        Update model based on user decisions.
        
        Args:
            context: Context containing request, features, and user profile
            accepted: Whether the user accepted the service
        """
        # Extract user profile
        user_profile = context.user_profile
        
        # No history to update if there's no user profile
        if not user_profile:
            return
        
        # Update acceptance rate and trip count
        if hasattr(user_profile, 'historical_acceptance_rate') and hasattr(user_profile, 'historical_trips'):
            # Update with running average
            old_rate = user_profile.historical_acceptance_rate
            old_trips = user_profile.historical_trips
            
            # Update rate and count
            new_trips = old_trips + 1
            new_rate = (old_rate * old_trips + (1 if accepted else 0)) / new_trips
            
            # Store updated values
            user_profile.historical_acceptance_rate = new_rate
            user_profile.historical_trips = new_trips
            
            logger.debug(f"Updated user {user_profile.id} acceptance rate: {new_rate:.3f} ({new_trips} trips)")
    
    def get_feature_importance(self) -> Dict[str, float]:
        """
        Get the importance of each feature in the model.
        
        Returns:
            Dict[str, float]: Feature names mapped to their importance values
        """
        # Return absolute values of weights as importance
        importance = {}
        
        for name, weight in self.default_weights.items():
            importance[name] = abs(weight)
        
        return importance
    
    def get_required_features(self) -> List[str]:
        """
        Get the list of required features for this model.
        
        Returns:
            List[str]: List of feature names that are required by this model
        """
        return [
            "walking_time_to_origin",
            "waiting_time",
            "in_vehicle_time",
            "walking_time_from_destination"
        ]
    
    def get_optional_features(self) -> List[str]:
        """
        Get the list of optional features for this model.
        
        Returns:
            List[str]: List of feature names that are optional but can improve the model
        """
        return [
            "cost", 
            "total_trip_time", 
            "direct_time", 
            "time_of_day", 
            "day_of_week", 
            "distance_to_pickup"
        ]
    
    def configure(self, config: Dict[str, Any]) -> None:
        """
        Configure the model with the given configuration.
        
        Args:
            config: Dictionary containing configuration parameters
        """
        # Update thresholds
        if 'thresholds' in config:
            self.default_thresholds.update(config['thresholds'])
        
        # Update weights
        if 'weights' in config:
            self.default_weights.update(config['weights'])
        
        # Update config dictionary
        self.config.update(config)
        
        logger.info(f"Configured default model with: {config}")