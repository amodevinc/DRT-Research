"""
Policy gradient-based user acceptance model.

This module provides a policy gradient-based model for user acceptance decisions,
learning from ongoing interactions with users using direct policy optimization.
"""
from typing import Dict, Any, Optional, List, Tuple
import numpy as np
import logging
import pickle
import os
import json
import random
from collections import deque

from drt_sim.algorithms.base_interfaces.user_acceptance_base import UserAcceptanceModel
from drt_sim.core.user.acceptance_context import AcceptanceContext
from drt_sim.core.user.feature_extractor import FeatureExtractor
from drt_sim.core.user.feature_provider import FeatureProviderRegistry
logger = logging.getLogger(__name__)

class PolicyGradientAgentModel(UserAcceptanceModel):
    """
    Policy gradient-based user acceptance model.
    
    This class implements a user acceptance model based on policy gradient,
    which directly optimizes the policy for predicting user acceptance.
    """
    
    def __init__(self, feature_extractor: Optional[FeatureExtractor] = None, 
                 feature_provider_registry: Optional[FeatureProviderRegistry] = None, 
                 **kwargs):
        """
        Initialize the policy gradient model.
        
        Args:
            feature_extractor: Feature extractor to use
            feature_provider_registry: Feature provider registry
            **kwargs: Additional parameters
        """
        super().__init__(feature_extractor, **kwargs)
        self.feature_provider_registry = feature_provider_registry
        
        # Policy gradient parameters
        self.alpha = kwargs.get('alpha', 0.01)  # Learning rate
        self.beta = kwargs.get('beta', 1.0)  # Softmax temperature/sensitivity
        self.gamma = kwargs.get('gamma', 0.95)  # Discount factor
        self.epsilon = kwargs.get('epsilon', 0.1)  # Exploration rate
        self.min_epsilon = kwargs.get('min_epsilon', 0.01)  # Minimum exploration rate
        self.epsilon_decay = kwargs.get('epsilon_decay', 0.995)  # Epsilon decay rate
        
        # Feature selection - these are the features used to define the state space
        self._feature_names = [
            "walking_time_to_origin",
            "waiting_time",
            "in_vehicle_time",
            "walking_time_from_destination",
            "price",
            "time_of_day",
            "day_of_week",
            "detour_factor",
            "vehicle_occupancy"
        ]
        
        # Initialize default weights - in standard policy gradient, we often start with small random values
        feature_dim = len(self._feature_names)
        self.default_weights = np.random.normal(0.0, 0.01, feature_dim)  # Small random values
        
        # Default feature coefficients - used for reference but not directly in weight initialization
        self.default_coefficients = {
            "walking_time_to_origin": -1.5,
            "waiting_time": -2.0,
            "in_vehicle_time": -1.5,
            "walking_time_from_destination": -1.5,
            "price": -2.5,
            "detour_factor": -1.0,
            "vehicle_occupancy": -0.5
        }
        
        # Experience buffer
        self.episode_history = []
        
        # Memory for experience replay
        self.memory = deque(maxlen=kwargs.get('memory_size', 1000))
        self.batch_size = kwargs.get('batch_size', 32)
        
        # Feature normalization parameters
        self.feature_means = None
        self.feature_stds = None
        # Apply configuration if provided
        if 'config' in kwargs:
            self.configure(kwargs['config'])
    
    def _normalize_features(self, features: Dict[str, float], update_stats=False) -> Dict[str, float]:
        """
        Normalize features to improve learning.
        
        Args:
            features: Dictionary of features
            update_stats: Whether to update normalization statistics
            
        Returns:
            Dict[str, float]: Normalized features
        """
        feature_vector = np.array([features.get(name, 0.0) for name in self._feature_names])
        
        if self.feature_means is None or self.feature_stds is None or update_stats:
            # Initialize or update normalization parameters
            if self.feature_means is None:
                self.feature_means = np.zeros(len(self._feature_names))
                self.feature_stds = np.ones(len(self._feature_names))
            
            # Update running statistics (simple moving average)
            if update_stats:
                alpha = 0.01  # Update rate
                self.feature_means = (1 - alpha) * self.feature_means + alpha * feature_vector
                self.feature_stds = (1 - alpha) * self.feature_stds + alpha * np.abs(feature_vector - self.feature_means)
                
                # Avoid division by zero
                self.feature_stds = np.maximum(self.feature_stds, 0.01)
        
        # Normalize
        normalized_vector = (feature_vector - self.feature_means) / self.feature_stds
        
        # Create normalized feature dictionary
        normalized_features = {}
        for i, name in enumerate(self._feature_names):
            normalized_features[name] = normalized_vector[i]
        
        return normalized_features
    
    def _get_feature_vector(self, features: Dict[str, float]) -> np.ndarray:
        """
        Convert feature dictionary to feature vector.
        
        Args:
            features: Dictionary of features
            
        Returns:
            np.ndarray: Feature vector
        """
        return np.array([features.get(name, 0.0) for name in self._feature_names])
    
    def calculate_acceptance_probability(self, context: AcceptanceContext) -> float:
        """
        Calculate probability of user accepting a proposed service.
        
        Args:
            context: Context containing request, features, and user profile
            
        Returns:
            float: Probability of acceptance (0.0 to 1.0)
        """
        # Extract features
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
        
        # Normalize features
        normalized_features = self._normalize_features(features_dict, update_stats=True)
        
        # Get weights by combining default weights with user profile weights
        weights = self._get_weights_from_profile(context.user_profile)
        
        # Extract feature vector
        feature_vector = self._get_feature_vector(normalized_features)
        
        # Calculate action value (weighted sum of features)
        action_value = np.dot(feature_vector, weights)
        
        # Convert to probability using sigmoid function
        probability = 1.0 / (1.0 + np.exp(-self.beta * action_value))
        
        # Clip to valid range
        return np.clip(probability, 0.01, 0.99)
    
    def _get_weights_from_profile(self, user_profile) -> np.ndarray:
        """
        Get weights from user profile or use default weights.
        
        Args:
            user_profile: User profile object
            
        Returns:
            np.ndarray: Weights vector
        """
        # Initialize with default weights
        weights = self.default_weights.copy()
        
        # Apply user profile specific weights if available
        if user_profile and hasattr(user_profile, 'weights') and isinstance(user_profile.weights, dict):
            for i, feature_name in enumerate(self._feature_names):
                if feature_name in user_profile.weights:
                    weights[i] = user_profile.weights[feature_name]
        
        return weights
    
    def _calculate_reward(self, accepted: bool, features: Dict[str, float], user_profile) -> float:
        """
        Calculate reward for reinforcement learning.
        
        Args:
            accepted: Whether the user accepted the service
            features: Feature dictionary
            user_profile: User profile
            
        Returns:
            float: Reward value
        """
        if accepted:
            # Base reward for acceptance
            reward = 1.0
            
            # Modified by service quality - using a simple formula based on feature values
            # This is more standard for policy gradient rather than using personalized adjustments
            quality_factor = 0.0
            count = 0
            
            # Calculate quality based on features with fixed weights
            weighted_features = {
                "walking_time_to_origin": -1.0,
                "waiting_time": -1.2,
                "in_vehicle_time": -1.0,
                "walking_time_from_destination": -1.0,
                "price": -1.5
            }
            
            # Apply quality factors
            for feature, weight in weighted_features.items():
                if feature in features:
                    quality_factor += weight * features[feature]
                    count += 1
            
            # Normalize quality factor
            if count > 0:
                quality_factor /= count
                
                # Rescale to 0.5-1.5 range
                quality_factor = 1.0 + quality_factor
                
                # Clip to reasonable range
                quality_factor = np.clip(quality_factor, 0.5, 1.5)
                
                # Apply to reward
                reward *= quality_factor
            
            return reward
        else:
            # Negative reward for rejection
            return -0.5
    
    def decide_acceptance(self, context: AcceptanceContext) -> Tuple[bool, float]:
        """
        Decide whether the user will accept the proposed service.
        
        Args:
            context: Context containing request, features, and user profile
            
        Returns:
            Tuple[bool, float]: (acceptance decision, acceptance probability)
        """
        # Calculate acceptance probability
        probability = self.calculate_acceptance_probability(context)
        
        # Exploration-exploitation tradeoff
        if random.random() < self.epsilon:
            # Explore: random decision
            accepted = random.random() < 0.5
        else:
            # Exploit: decision based on probability
            accepted = random.random() < probability
        
        return accepted, probability
    
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
        
        # Enrich features if provider registry available
        if self.feature_provider_registry is not None:
            provider_context = {
                "features": features_dict.copy(),
                "user_profile": context.user_profile
            }
            
            additional_features = self.feature_provider_registry.get_features(
                context.request, 
                provider_context
            )
            
            features_dict.update(additional_features)
        
        # Normalize features
        normalized_features = self._normalize_features(features_dict)
        
        # Get weights - use default weights as baseline for learning
        weights = self.default_weights.copy()
        
        # Extract feature vector
        feature_vector = self._get_feature_vector(normalized_features)
        
        # Calculate probability with current weights
        action_value = np.dot(feature_vector, weights)
        probability = 1.0 / (1.0 + np.exp(-self.beta * action_value))
        
        # Calculate reward
        reward = self._calculate_reward(accepted, normalized_features, context.user_profile)
        
        # Store in memory for experience replay
        self.memory.append((feature_vector, accepted, reward))
        
        # Policy gradient update
        # Gradient calculation
        if accepted:
            # For accepted services, gradient is (1 - P(accept)) * features
            gradient = (1 - probability) * feature_vector
        else:
            # For rejected services, gradient is -P(accept) * features
            gradient = -probability * feature_vector
        
        # Update weights using policy gradient
        weights += self.alpha * reward * gradient
        
        # Store updated weights in default weights for next prediction
        self.default_weights = weights
        
        # Store in episode history
        self.episode_history.append((feature_vector, accepted, reward, probability))
        
        # Perform experience replay
        self._experience_replay()
        
        # Decay exploration rate
        self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)
    
    def _experience_replay(self) -> None:
        """
        Perform experience replay using the memory buffer.
        """
        # Skip if not enough samples
        if len(self.memory) < self.batch_size:
            return
        
        # Sample batch from memory
        batch = random.sample(list(self.memory), self.batch_size)
        
        # Get default weights
        weights = self.default_weights.copy()
        
        # Process each experience
        for feature_vector, accepted, reward in batch:
            # Calculate current probability
            action_value = np.dot(feature_vector, weights)
            probability = 1.0 / (1.0 + np.exp(-self.beta * action_value))
            
            # Calculate gradient
            if accepted:
                gradient = (1 - probability) * feature_vector
            else:
                gradient = -probability * feature_vector
            
            # Small update from replay
            replay_alpha = self.alpha * 0.5  # Smaller learning rate for replay
            weights += replay_alpha * reward * gradient
        
        # Store updated default weights
        self.default_weights = weights
    
    def policy_evaluation(self) -> np.ndarray:
        """
        Evaluate policy by computing returns for each time step.
        
        Returns:
            np.ndarray: Array of returns for each time step
        """
        history = self.episode_history
        if not history:
            return np.array([])
        
        # Calculate returns using backwards-looking sum of discounted rewards
        returns = np.zeros(len(history))
        G = 0
        
        for t in reversed(range(len(history))):
            _, _, reward, _ = history[t]
            G = reward + self.gamma * G
            returns[t] = G
        
        return returns
    
    def policy_improvement(self, returns: np.ndarray) -> None:
        """
        Improve policy by updating weights based on returns.
        
        Args:
            returns: Array of returns for each time step
        """
        history = self.episode_history
        if not history or len(returns) == 0:
            return
        
        # Get default weights
        weights = self.default_weights.copy()
        
        # Process each step in the episode
        for t, (feature_vector, accepted, _, probability) in enumerate(history):
            # Calculate gradient
            if accepted:
                gradient = (1 - probability) * feature_vector
            else:
                gradient = -probability * feature_vector
            
            # Update weights using return (G) instead of immediate reward
            weights += self.alpha * returns[t] * gradient
        
        # Store updated weights
        self.default_weights = weights
        
        # Clear episode history
        self.episode_history = []
    
    def update_policy(self) -> None:
        """
        Update policy based on collected experience.
        """
        # Evaluate policy to get returns
        returns = self.policy_evaluation()
        
        # Improve policy using returns
        self.policy_improvement(returns)
    
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
        
        # Update policy after processing all examples
        self.update_policy()
    
    def get_feature_importance(self) -> Dict[str, float]:
        """
        Get the importance of each feature in the model.
        
        Returns:
            Dict[str, float]: Feature names mapped to their importance values
        """
        # Use absolute values of default weights as importance
        abs_weights = np.abs(self.default_weights)
        
        # Map to feature names
        importance = {}
        for i, name in enumerate(self._feature_names):
            if i < len(abs_weights):
                importance[name] = float(abs_weights[i])
        
        # Normalize to sum to 1
        total = sum(importance.values())
        if total > 0:
            for name in importance:
                importance[name] /= total
        
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
        return [name for name in self._feature_names if name not in self.get_required_features()]
    
    def save_model(self, filepath: str, metadata: Optional[Dict[str, Any]] = None) -> None:
        """
        Save the model to a file.
        
        Args:
            filepath: Path where the model should be saved
            metadata: Optional dictionary with additional metadata to save with the model
        
        Raises:
            IOError: If the model cannot be saved to the specified path
        """
        # Create custom metadata specific to PolicyGradientAgentModel
        pg_metadata = {
            'alpha': self.alpha,
            'beta': self.beta,
            'gamma': self.gamma,
            'epsilon': self.epsilon,
            'min_epsilon': self.min_epsilon,
            'epsilon_decay': self.epsilon_decay,
            'feature_names': self._feature_names,
            'default_coefficients': self.default_coefficients,
            'feature_means': self.feature_means.tolist() if self.feature_means is not None else None,
            'feature_stds': self.feature_stds.tolist() if self.feature_stds is not None else None
        }
        
        # Merge with user-provided metadata
        if metadata:
            pg_metadata.update(metadata)
        
        # Call the parent class implementation
        super().save_model(filepath, pg_metadata)
        
        try:
            # Save weights separately
            weights_path = f"{filepath}.weights"
            
            # Convert weights to serializable format
            serializable_data = {
                "weights": self.default_weights.tolist()
            }
            
            with open(weights_path, 'w') as f:
                json.dump(serializable_data, f)
                
            logger.info(f"Saved policy gradient weights to {weights_path}")
        except Exception as e:
            logger.error(f"Error saving policy gradient weights: {e}")
            raise IOError(f"Failed to save model weights: {str(e)}")
    
    @classmethod
    def load_model(cls, filepath: str) -> 'PolicyGradientAgentModel':
        """
        Load a model from a file.
        
        Args:
            filepath: Path to the saved model
            
        Returns:
            PolicyGradientAgentModel: The loaded model
            
        Raises:
            IOError: If the model cannot be loaded from the specified path
        """
        # First load the base model using the parent class method
        model = super().load_model(filepath)
        
        # Load additional model-specific components
        try:
            # Load weights
            weights_path = f"{filepath}.weights"
            if os.path.exists(weights_path):
                with open(weights_path, 'r') as f:
                    data = json.load(f)
                
                # Load weights
                if "weights" in data:
                    model.default_weights = np.array(data["weights"])
                elif "default_weights" in data:  # Backward compatibility
                    model.default_weights = np.array(data["default_weights"])
                
                logger.info(f"Loaded policy gradient weights from {weights_path}")
            
            # Load metadata to update model attributes
            metadata_path = f"{filepath}.meta.json"
            if os.path.exists(metadata_path):
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
                
                # Update model attributes from metadata
                if 'alpha' in metadata:
                    model.alpha = metadata['alpha']
                if 'beta' in metadata:
                    model.beta = metadata['beta']
                if 'gamma' in metadata:
                    model.gamma = metadata['gamma']
                if 'epsilon' in metadata:
                    model.epsilon = metadata['epsilon']
                if 'min_epsilon' in metadata:
                    model.min_epsilon = metadata['min_epsilon']
                if 'epsilon_decay' in metadata:
                    model.epsilon_decay = metadata['epsilon_decay']
                if 'feature_names' in metadata:
                    model._feature_names = metadata['feature_names']
                if 'default_coefficients' in metadata:
                    model.default_coefficients = metadata['default_coefficients']
                if 'feature_means' in metadata and metadata['feature_means'] is not None:
                    model.feature_means = np.array(metadata['feature_means'])
                if 'feature_stds' in metadata and metadata['feature_stds'] is not None:
                    model.feature_stds = np.array(metadata['feature_stds'])
            
            return model
        except Exception as e:
            logger.error(f"Error loading policy gradient model components: {e}")
            raise IOError(f"Failed to load model components: {str(e)}")
    
    def configure(self, config: Dict[str, Any]) -> None:
        """
        Configure the model with the given configuration.
        
        Args:
            config: Dictionary containing configuration parameters
        """
        # Update learning parameters
        if 'alpha' in config:
            self.alpha = config['alpha']
        if 'beta' in config:
            self.beta = config['beta']
        if 'gamma' in config:
            self.gamma = config['gamma']
        if 'epsilon' in config:
            self.epsilon = config['epsilon']
        if 'min_epsilon' in config:
            self.min_epsilon = config['min_epsilon']
        if 'epsilon_decay' in config:
            self.epsilon_decay = config['epsilon_decay']
        
        # Update default coefficients
        if 'default_coefficients' in config:
            self.default_coefficients.update(config['default_coefficients'])
        
        # Update feature names
        if 'feature_names' in config:
            old_feature_names = self._feature_names
            self._feature_names = config['feature_names']
            
            # Resize weights if feature dimension changed
            if len(old_feature_names) != len(self._feature_names):
                old_weights = self.default_weights
                new_weights = np.zeros(len(self._feature_names))
                
                # Copy values for features that exist in both
                for i, name in enumerate(self._feature_names):
                    if name in old_feature_names:
                        old_idx = old_feature_names.index(name)
                        if old_idx < len(old_weights):
                            new_weights[i] = old_weights[old_idx]
                
                self.default_weights = new_weights
                
                # Reset normalization parameters
                self.feature_means = None
                self.feature_stds = None
                
                # Clear memory and episode history
                self.memory.clear()
                self.episode_history = []
        
        # Update memory parameters
        if 'memory_size' in config:
            old_memory = list(self.memory)
            self.memory = deque(maxlen=config['memory_size'])
            for item in old_memory[-config['memory_size']:]:
                self.memory.append(item)
        
        if 'batch_size' in config:
            self.batch_size = config['batch_size']
        
        # Update config dictionary
        self.config.update(config)
        
        logger.info(f"Configured policy gradient model with: {config}")