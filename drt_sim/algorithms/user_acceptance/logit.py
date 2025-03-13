"""
Logit-based user acceptance model.

This module provides a logit-based model for user acceptance decisions,
modeling acceptance probability using logistic regression.
"""
from typing import Dict, Any, Optional, List, Tuple
import numpy as np
import logging
import pickle
import os
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from drt_sim.algorithms.base_interfaces.user_acceptance_base import UserAcceptanceModel
from drt_sim.core.user.acceptance_context import AcceptanceContext
from drt_sim.core.user.feature_extractor import FeatureExtractor
from drt_sim.core.user.feature_provider import FeatureProviderRegistry
logger = logging.getLogger(__name__)

class LogitModel(UserAcceptanceModel):
    """
    Logit-based user acceptance model.
    
    This class implements a user acceptance model based on logistic regression,
    which models the probability of acceptance as a function of features.
    """
    
    def __init__(self, feature_extractor: Optional[FeatureExtractor] = None, feature_provider_registry: Optional[FeatureProviderRegistry] = None, **kwargs):
        """
        Initialize the logit model.
        
        Args:
            feature_extractor: Feature extractor to use
            **kwargs: Additional parameters
        """
        super().__init__(feature_extractor, **kwargs)
        self.feature_provider_registry = feature_provider_registry
        
        # Initialize the logistic regression model
        self.model = LogisticRegression(
            solver='lbfgs',
            max_iter=1000,
            class_weight='balanced',
            random_state=42
        )
        
        # Feature scaling
        self.scaler = StandardScaler()
        
        # Model state
        self.is_trained = False
        self.feature_names = []
        self.training_data = []
        self.max_training_samples = kwargs.get('max_training_samples', 10000)
        
        # Default coefficients for primary features (negative values indicate costs)
        self.default_coefficients = {
            "walking_time_to_origin": -1.5,
            "waiting_time": -2.0,
            "in_vehicle_time": -1.5,
            "walking_time_from_destination": -1.5,
            "cost": -2.5,
            "total_trip_time": -1.0,
            "time_of_day": 0.0,
            "day_of_week": 0.0,
            "distance_to_pickup": -0.5,
            "weather_condition": -0.3,
            "vehicle_capacity": 0.1,
            "historical_acceptance_rate": 1.5
        }
        
        # Set required features
        self._required_features = [
            "walking_time_to_origin",
            "waiting_time",
            "in_vehicle_time",
            "walking_time_from_destination"
        ]
        
        # Set optional features
        self._optional_features = [
            "cost",
            "total_trip_time",
            "time_of_day",
            "day_of_week",
            "distance_to_pickup",
            "weather_condition",
            "vehicle_capacity",
            "historical_acceptance_rate"
        ]
        
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
        # Extract features as a dictionary
        features_dict = self.feature_extractor.extract_features_dict(
            context.features,
            context.request,
            context.user_profile
        )
        # Enrich features using the provider registry if available
        if self.feature_provider_registry is not None:
            # Create a context dict for the providers
            provider_context = {
                "features": features_dict.copy(),
                "user_profile": context.user_profile
            }
            
            # Get additional features from providers
            additional_features = self.feature_provider_registry.get_features(
                context.request, 
                provider_context
            )
            
            # Update features with additional ones
            features_dict.update(additional_features)
        
        # Convert to feature vector
        feature_vector = self._get_feature_vector(features_dict)
        
        # If model is not trained, use default logit function
        if not self.is_trained or len(self.feature_names) == 0:
            return self._calculate_default_probability(features_dict, context.user_profile)
        
        # Scale features
        if hasattr(self, 'scaler') and self.scaler is not None and hasattr(self.scaler, 'transform'):
            feature_vector = self.scaler.transform([feature_vector])[0]
        
        # Calculate probability
        try:
            # Reshape for single sample prediction
            X = feature_vector.reshape(1, -1)
            probability = self.model.predict_proba(X)[0, 1]
            return float(probability)
        except Exception as e:
            logger.error(f"Error calculating probability with model: {e}")
            return self._calculate_default_probability(features_dict, context.user_profile)
    
    def _calculate_default_probability(self, features: Dict[str, float], user_profile=None) -> float:
        """
        Calculate probability using default logit function.
        
        Args:
            features: Dictionary of normalized features
            user_profile: User profile (optional)
            
        Returns:
            float: Probability of acceptance (0.0 to 1.0)
        """
        # Apply user profile specific weights
        coefficients = self.default_coefficients.copy()
        
        if user_profile and hasattr(user_profile, 'weights') and isinstance(user_profile.weights, dict):
            # Blend default coefficients with user weights
            for feature, weight in user_profile.weights.items():
                if feature in coefficients:
                    # User weights are already negative for costs, so use as-is
                    coefficients[feature] = weight
        
        # Apply service preference factors
        if user_profile and hasattr(user_profile, 'service_preference'):
            preference = user_profile.service_preference
            
            if preference == 'speed':
                # Speed-focused users care more about time
                coefficients['waiting_time'] *= 1.5
                coefficients['in_vehicle_time'] *= 1.5
                if 'cost' in coefficients:
                    coefficients['cost'] *= 0.8  # Less sensitive to cost
                
            elif preference == 'comfort':
                # Comfort-focused users care more about minimal walking
                coefficients['walking_time_to_origin'] *= 1.5
                coefficients['walking_time_from_destination'] *= 1.5
                if 'vehicle_capacity' in coefficients:
                    coefficients['vehicle_capacity'] *= 2.0  # More sensitive to comfort
                
            elif preference == 'cost':
                # Cost-focused users care more about price
                if 'cost' in coefficients:
                    coefficients['cost'] *= 2.0  # More sensitive to cost
                coefficients['waiting_time'] *= 0.8  # Less sensitive to waiting
                coefficients['in_vehicle_time'] *= 0.8  # Less sensitive to travel time
        
        # Calculate utility using coefficients
        utility = 0.0
        for feature_name, value in features.items():
            if feature_name in coefficients:
                utility += coefficients[feature_name] * value
        
        # Add intercept (baseline utility)
        utility += 1.0
        
        # Convert utility to probability using logistic function
        probability = 1.0 / (1.0 + np.exp(-utility))
        
        return min(max(probability, 0.01), 0.99)  # Clip to avoid extreme values
    
    def update_model(self, context: AcceptanceContext, accepted: bool) -> None:
        """
        Update model based on user decisions.
        
        Args:
            context: Context containing request, features, and user profile
            accepted: Whether the user accepted the service
        """
        # Extract features
        features_dict = self.feature_extractor.extract_features_dict(
            context.features,
            context.request,
            context.user_profile
        )
        
        # Add to training data
        self.training_data.append({
            'features': features_dict.copy(),
            'accepted': accepted,
            'user_profile': context.user_profile
        })
        
        # Limit training data size
        if len(self.training_data) > self.max_training_samples:
            self.training_data = self.training_data[-self.max_training_samples:]
        
        # Retrain model if we have enough data
        if len(self.training_data) >= 50:
            self._retrain_model()
    
    def _retrain_model(self) -> None:
        """
        Retrain the logistic regression model.
        """
        if len(self.training_data) == 0:
            return
        
        try:
            # Prepare feature names
            sample_features = self.training_data[0]['features']
            self.feature_names = list(sample_features.keys())
            
            # Prepare training data
            X = []
            y = []
            
            for sample in self.training_data:
                features = sample['features']
                feature_vector = self._get_feature_vector(features)
                X.append(feature_vector)
                y.append(1 if sample['accepted'] else 0)
            
            X = np.array(X)
            y = np.array(y)
            
            # Scale features
            self.scaler.fit(X)
            X_scaled = self.scaler.transform(X)
            
            # Train the model
            self.model.fit(X_scaled, y)
            
            self.is_trained = True
            logger.info(f"Retrained logit model with {len(self.training_data)} samples")
        except Exception as e:
            logger.error(f"Error retraining logit model: {e}")
    
    def _get_feature_vector(self, features: Dict[str, float]) -> np.ndarray:
        """
        Convert feature dictionary to feature vector.
        
        Args:
            features: Dictionary of features
            
        Returns:
            np.ndarray: Feature vector
        """
        if not self.feature_names:
            # Initialize feature names from the first sample
            self.feature_names = list(features.keys())
        
        # Create vector from the feature names in the correct order
        feature_vector = np.zeros(len(self.feature_names))
        
        for i, name in enumerate(self.feature_names):
            if name in features:
                feature_vector[i] = features[name]
        
        return feature_vector
    
    def batch_update(self, training_data: List[Dict[str, Any]]) -> None:
        """
        Update model with batch training data.
        
        Args:
            training_data: List of training examples with features and outcomes
        """
        # Process training examples
        for example in training_data:
            if "features" in example and "accepted" in example:
                # Create context
                context = AcceptanceContext(
                    features=example["features"],
                    request=example.get("request"),
                    user_profile=example.get("user_profile")
                )
                
                # Update model
                self.update_model(context, example["accepted"])
    
    def get_feature_importance(self) -> Dict[str, float]:
        """
        Get the importance of each feature in the model.
        
        Returns:
            Dict[str, float]: Feature names mapped to their importance values
        """
        if not self.is_trained or not hasattr(self.model, 'coef_'):
            # Return default coefficients if model not trained
            return {name: abs(coef) for name, coef in self.default_coefficients.items()}
        
        # Get feature importance from model coefficients
        importance = {}
        coefficients = self.model.coef_[0]
        
        for i, name in enumerate(self.feature_names):
            if i < len(coefficients):
                importance[name] = abs(coefficients[i])
        
        return importance
    
    def get_required_features(self) -> List[str]:
        """
        Get the list of required features for this model.
        
        Returns:
            List[str]: List of feature names that are required by this model
        """
        return self._required_features
    
    def get_optional_features(self) -> List[str]:
        """
        Get the list of optional features for this model.
        
        Returns:
            List[str]: List of feature names that are optional but can improve the model
        """
        return self._optional_features
    
    def save_model(self, path: str) -> None:
        """
        Save the model to a file.
        
        Args:
            path: Path to save the model
        """
        try:
            model_data = {
                'logistic_model': self.model,
                'scaler': self.scaler,
                'feature_names': self.feature_names,
                'is_trained': self.is_trained,
                'default_coefficients': self.default_coefficients,
                'required_features': self._required_features,
                'optional_features': self._optional_features
            }
            
            with open(path, 'wb') as f:
                pickle.dump(model_data, f)
                
            logger.info(f"Saved logit model to {path}")
        except Exception as e:
            logger.error(f"Error saving logit model: {e}")
    
    def load_model(self, path: str) -> None:
        """
        Load the model from a file.
        
        Args:
            path: Path to load the model from
        """
        try:
            if not os.path.exists(path):
                logger.warning(f"Model file {path} does not exist")
                return
                
            with open(path, 'rb') as f:
                model_data = pickle.load(f)
            
            self.model = model_data.get('logistic_model')
            self.scaler = model_data.get('scaler')
            self.feature_names = model_data.get('feature_names', [])
            self.is_trained = model_data.get('is_trained', False)
            self.default_coefficients = model_data.get('default_coefficients', self.default_coefficients)
            self._required_features = model_data.get('required_features', self._required_features)
            self._optional_features = model_data.get('optional_features', self._optional_features)
            
            logger.info(f"Loaded logit model from {path}")
        except Exception as e:
            logger.error(f"Error loading logit model: {e}")
    
    def configure(self, config: Dict[str, Any]) -> None:
        """
        Configure the model with the given configuration.
        
        Args:
            config: Dictionary containing configuration parameters
        """
        # Update max training samples
        if 'max_training_samples' in config:
            self.max_training_samples = config['max_training_samples']
        
        # Update default coefficients
        if 'default_coefficients' in config:
            self.default_coefficients.update(config['default_coefficients'])
        
        # Update logistic regression parameters
        if 'logistic_params' in config:
            params = config['logistic_params']
            if 'solver' in params:
                self.model.solver = params['solver']
            if 'max_iter' in params:
                self.model.max_iter = params['max_iter']
            if 'class_weight' in params:
                self.model.class_weight = params['class_weight']
        
        # Update feature lists
        if 'required_features' in config:
            self._required_features = config['required_features']
        if 'optional_features' in config:
            self._optional_features = config['optional_features']
        
        # Update the config
        self.config.update(config)
        
        logger.info(f"Configured logit model with: {config}")