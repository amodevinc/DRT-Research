"""
Reinforcement learning-based user acceptance model.

This module provides a reinforcement learning-based model for user
acceptance decisions, learning from ongoing interactions with users.
"""
from typing import Dict, Any, Optional, List, Tuple
import numpy as np
import logging
import pickle
import os
import random
from collections import deque

from drt_sim.algorithms.base_interfaces.user_acceptance_base import UserAcceptanceModel
from drt_sim.core.user.acceptance_context import AcceptanceContext
from drt_sim.core.user.feature_extractor import FeatureExtractor
from drt_sim.core.user.feature_provider import FeatureProviderRegistry
logger = logging.getLogger(__name__)

class RLAcceptanceModel(UserAcceptanceModel):
    """
    Reinforcement learning-based user acceptance model.
    
    This class implements a user acceptance model based on reinforcement learning,
    specifically using Q-learning to learn user preferences over time.
    """
    
    def __init__(self, feature_extractor: Optional[FeatureExtractor] = None, feature_provider_registry: Optional[FeatureProviderRegistry] = None, **kwargs):
        """
        Initialize the RL model.
        
        Args:
            feature_extractor: Feature extractor to use
            **kwargs: Additional parameters
        """
        super().__init__(feature_extractor, **kwargs)
        self.feature_provider_registry = feature_provider_registry
        # RL parameters
        self.alpha = kwargs.get('alpha', 0.1)  # Learning rate
        self.gamma = kwargs.get('gamma', 0.9)  # Discount factor
        self.epsilon = kwargs.get('epsilon', 0.2)  # Exploration rate
        self.min_epsilon = kwargs.get('min_epsilon', 0.05)  # Minimum exploration rate
        self.epsilon_decay = kwargs.get('epsilon_decay', 0.9999)  # Epsilon decay rate
        
        # State discretization parameters
        self.num_bins = kwargs.get('num_bins', 10)  # Number of bins for feature discretization
        
        # Q-table for storing state-action values
        self.q_tables = {}  # Dictionary mapping user types to their Q-tables
        
        # Memory buffer for experience replay
        self.memory = deque(maxlen=kwargs.get('memory_size', 1000))
        self.batch_size = kwargs.get('batch_size', 64)
        
        # Feature selection - these are the features used to define the state space
        self._feature_names = [
            "walking_time_to_origin",
            "waiting_time",
            "in_vehicle_time",
            "walking_time_from_destination",
            "cost",
            "time_of_day",
            "day_of_week"
        ]
        
        # Default model as fallback
        from drt_sim.algorithms.user_acceptance.default import DefaultModel
        self.default_model = DefaultModel(feature_extractor=feature_extractor)
        
        # User type mapping
        self.service_preference_map = {
            "speed": 0,
            "comfort": 1,
            "cost": 2,
            "unknown": 3
        }
        
        # Initialize Q-tables for each user type
        for user_type in self.service_preference_map.values():
            self._initialize_q_table(user_type)
        
        # Apply configuration if provided
        if 'config' in kwargs:
            self.configure(kwargs['config'])
    
    def _initialize_q_table(self, user_type: int) -> None:
        """
        Initialize Q-table for a user type.
        
        Args:
            user_type: User type identifier
        """
        # Create a default Q-table with zeros
        # Structure: {state_tuple: [q_value_for_reject, q_value_for_accept]}
        self.q_tables[user_type] = {}
    
    def _get_state(self, features: Dict[str, float]) -> Tuple:
        """
        Convert features to a discrete state.
        
        Args:
            features: Feature dictionary
            
        Returns:
            Tuple: Discretized state representation
        """
        state = []
        
        for feature_name in self._feature_names:
            if feature_name in features:
                # Discretize the feature value into bins [0, 1, 2, ..., num_bins-1]
                bin_value = min(int(features[feature_name] * self.num_bins), self.num_bins - 1)
                state.append(bin_value)
            else:
                state.append(0)  # Default bin if feature not available
        
        return tuple(state)
    
    def _get_user_type(self, user_profile) -> int:
        """
        Get the user type from the user profile.
        
        Args:
            user_profile: User profile
            
        Returns:
            int: User type identifier
        """
        if user_profile and hasattr(user_profile, 'service_preference'):
            preference = user_profile.service_preference
            return self.service_preference_map.get(preference, self.service_preference_map["unknown"])
        
        return self.service_preference_map["unknown"]
    
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
        
        # Get user type
        user_type = self._get_user_type(context.user_profile)
        
        # Get state representation
        state = self._get_state(features)
        
        # Get Q-values for this state
        q_table = self.q_tables.get(user_type, {})
        
        if state in q_table:
            q_values = q_table[state]
            # Convert Q-values to probability using softmax
            q_reject, q_accept = q_values
            
            # Apply temperature-based softmax
            temperature = 1.0
            exp_reject = np.exp(q_reject / temperature)
            exp_accept = np.exp(q_accept / temperature)
            
            # Calculate acceptance probability
            probability = exp_accept / (exp_reject + exp_accept)
            
            # Apply user profile preferences
            probability = self._apply_user_preferences(probability, features, context.user_profile)
            
            # Clip to valid range
            return min(max(probability, 0.01), 0.99)
        
        # If state not in Q-table, use default model as fallback
        return self.default_model.calculate_acceptance_probability(context)
    
    def _apply_user_preferences(self, probability: float, features: Dict[str, float], user_profile) -> float:
        """
        Apply user preferences to adjust probability.
        
        Args:
            probability: Base probability
            features: Feature dictionary
            user_profile: User profile
            
        Returns:
            float: Adjusted probability
        """
        if not user_profile:
            return probability
        
        # Apply service preference adjustments
        if hasattr(user_profile, 'service_preference'):
            preference = user_profile.service_preference
            
            if preference == 'speed':
                # Speed-focused users care more about time
                if "waiting_time" in features and "in_vehicle_time" in features:
                    total_time = features["waiting_time"] + features["in_vehicle_time"]
                    if total_time > 0.6:  # High normalized time
                        probability *= 0.9  # Decrease probability for slow services
                    elif total_time < 0.3:  # Low normalized time
                        probability *= 1.1  # Increase probability for fast services
            
            elif preference == 'comfort':
                # Comfort-focused users care more about minimal walking
                if "walking_time_to_origin" in features and "walking_time_from_destination" in features:
                    total_walking = features["walking_time_to_origin"] + features["walking_time_from_destination"]
                    if total_walking > 0.5:  # High normalized walking time
                        probability *= 0.9  # Decrease probability for long walks
                    elif total_walking < 0.2:  # Low normalized walking time
                        probability *= 1.1  # Increase probability for short walks
            
            elif preference == 'cost':
                # Cost-focused users care more about price
                if "cost" in features:
                    if features["cost"] > 0.6:  # High normalized cost
                        probability *= 0.8  # Decrease probability for expensive services
                    elif features["cost"] < 0.3:  # Low normalized cost
                        probability *= 1.2  # Increase probability for cheap services
        
        # Apply historical behavior adjustment
        if hasattr(user_profile, 'historical_acceptance_rate'):
            # Slightly bias toward historical behavior
            historical_rate = user_profile.historical_acceptance_rate
            probability = 0.9 * probability + 0.1 * historical_rate
        
        return probability
    
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
        features = self.feature_extractor.extract_features_dict(
            context.features,
            context.request,
            context.user_profile
        )
        
        # Get user type
        user_type = self._get_user_type(context.user_profile)
        
        # Get state representation
        state = self._get_state(features)
        
        # Get Q-table for this user type
        q_table = self.q_tables.get(user_type, {})
        
        # Initialize state if not seen before
        if state not in q_table:
            q_table[state] = [0.0, 0.0]  # [q_reject, q_accept]
        
        # Get current Q-values
        q_values = q_table[state]
        
        # Calculate reward based on user's decision and trip features
        reward = self._calculate_reward(accepted, features, context.user_profile)
        
        # Update Q-value for the taken action using Q-learning
        action_idx = 1 if accepted else 0  # 1 for accept, 0 for reject
        
        # Simple Q-learning update
        q_values[action_idx] = (1 - self.alpha) * q_values[action_idx] + self.alpha * reward
        
        # Store experience in replay buffer
        self.memory.append((state, action_idx, reward, user_type))
        
        # Perform experience replay if enough samples
        if len(self.memory) >= self.batch_size:
            self._experience_replay()
        
        # Decay exploration rate
        self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)
        
        # Store updated Q-table
        self.q_tables[user_type] = q_table
    
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
            
            # Modified by service quality
            quality_factor = 0.0
            count = 0
            
            # Sum feature quality factors
            if "walking_time_to_origin" in features:
                quality_factor -= features["walking_time_to_origin"]
                count += 1
                
            if "waiting_time" in features:
                quality_factor -= features["waiting_time"]
                count += 1
                
            if "in_vehicle_time" in features:
                quality_factor -= features["in_vehicle_time"]
                count += 1
                
            if "walking_time_from_destination" in features:
                quality_factor -= features["walking_time_from_destination"]
                count += 1
                
            if "cost" in features:
                quality_factor -= features["cost"]
                count += 1
            
            # Average quality factor and scale
            if count > 0:
                quality_factor = quality_factor / count
                quality_factor = quality_factor + 1.0  # Shift to [0, 1] range
                
                # Apply quality factor to reward
                reward *= quality_factor
            
            # Apply user preference boost
            if user_profile and hasattr(user_profile, 'service_preference'):
                preference = user_profile.service_preference
                
                if preference == 'speed' and "waiting_time" in features and "in_vehicle_time" in features:
                    # Boost for speed-focused users getting fast service
                    total_time = features["waiting_time"] + features["in_vehicle_time"]
                    if total_time < 0.3:
                        reward *= 1.2
                
                elif preference == 'comfort' and "walking_time_to_origin" in features and "walking_time_from_destination" in features:
                    # Boost for comfort-focused users getting comfortable service
                    total_walking = features["walking_time_to_origin"] + features["walking_time_from_destination"]
                    if total_walking < 0.2:
                        reward *= 1.2
                
                elif preference == 'cost' and "cost" in features:
                    # Boost for cost-focused users getting cheap service
                    if features["cost"] < 0.3:
                        reward *= 1.2
            
            return reward
        else:
            # Negative reward for rejection
            return -0.5
    
    def _experience_replay(self) -> None:
        """
        Perform experience replay to update Q-values from past experiences.
        """
        # Sample random batch from memory
        batch = random.sample(self.memory, min(self.batch_size, len(self.memory)))
        
        for state, action, reward, user_type in batch:
            # Get Q-table for this user type
            q_table = self.q_tables.get(user_type, {})
            
            # Skip if state not in table
            if state not in q_table:
                continue
            
            # Get current Q-values
            q_values = q_table[state]
            
            # Update Q-value for the action
            q_values[action] = (1 - self.alpha) * q_values[action] + self.alpha * reward
            
            # Store updated Q-values
            q_table[state] = q_values
            self.q_tables[user_type] = q_table
    
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
        # Analyze Q-values to determine feature importance
        importance = {name: 0.0 for name in self._feature_names}
        
        # Analyze all Q-tables
        for user_type, q_table in self.q_tables.items():
            # Skip if Q-table is empty
            if not q_table:
                continue
            
            # Analyze each feature's impact on Q-values
            for feature_idx, feature_name in enumerate(self._feature_names):
                # Calculate average absolute difference in Q-values when this feature changes
                diff_sum = 0.0
                count = 0
                
                for state, q_values in q_table.items():
                    # Try different values for this feature
                    for value in range(self.num_bins):
                        # Skip if same as current value
                        if state[feature_idx] == value:
                            continue
                        
                        # Create new state with different feature value
                        new_state = list(state)
                        new_state[feature_idx] = value
                        new_state = tuple(new_state)
                        
                        # Skip if new state not in Q-table
                        if new_state not in q_table:
                            continue
                        
                        # Calculate absolute difference in Q-values for acceptance
                        q_diff = abs(q_table[new_state][1] - q_values[1])
                        diff_sum += q_diff
                        count += 1
                
                # Calculate average difference
                if count > 0:
                    importance[feature_name] += diff_sum / count
        
        # Normalize importance values
        max_importance = max(importance.values(), default=1.0)
        if max_importance > 0:
            for name in importance:
                importance[name] /= max_importance
        
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
    
    def save_model(self, path: str) -> None:
        """
        Save the model to a file.
        
        Args:
            path: Path to save the model
        """
        try:
            model_data = {
                'q_tables': self.q_tables,
                'alpha': self.alpha,
                'gamma': self.gamma,
                'epsilon': self.epsilon,
                'min_epsilon': self.min_epsilon,
                'epsilon_decay': self.epsilon_decay,
                'num_bins': self.num_bins,
                'feature_names': self._feature_names,
                'service_preference_map': self.service_preference_map
            }
            
            with open(path, 'wb') as f:
                pickle.dump(model_data, f)
                
            logger.info(f"Saved RL model to {path}")
        except Exception as e:
            logger.error(f"Error saving RL model: {e}")
    
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
            
            self.q_tables = model_data.get('q_tables', {})
            self.alpha = model_data.get('alpha', self.alpha)
            self.gamma = model_data.get('gamma', self.gamma)
            self.epsilon = model_data.get('epsilon', self.epsilon)
            self.min_epsilon = model_data.get('min_epsilon', self.min_epsilon)
            self.epsilon_decay = model_data.get('epsilon_decay', self.epsilon_decay)
            self.num_bins = model_data.get('num_bins', self.num_bins)
            self._feature_names = model_data.get('feature_names', self._feature_names)
            self.service_preference_map = model_data.get('service_preference_map', self.service_preference_map)
            
            logger.info(f"Loaded RL model from {path}")
        except Exception as e:
            logger.error(f"Error loading RL model: {e}")
    
    def configure(self, config: Dict[str, Any]) -> None:
        """
        Configure the model with the given configuration.
        
        Args:
            config: Dictionary containing configuration parameters
        """
        # Update RL parameters
        if 'alpha' in config:
            self.alpha = config['alpha']
        if 'gamma' in config:
            self.gamma = config['gamma']
        if 'epsilon' in config:
            self.epsilon = config['epsilon']
        if 'min_epsilon' in config:
            self.min_epsilon = config['min_epsilon']
        if 'epsilon_decay' in config:
            self.epsilon_decay = config['epsilon_decay']
        
        # Update discretization parameters
        if 'num_bins' in config:
            self.num_bins = config['num_bins']
            
            # Reinitialize Q-tables with new discretization
            for user_type in self.q_tables.keys():
                self._initialize_q_table(user_type)
        
        # Update memory parameters
        if 'memory_size' in config:
            # Create new memory with specified size
            old_memory = list(self.memory)
            self.memory = deque(maxlen=config['memory_size'])
            self.memory.extend(old_memory[-config['memory_size']:])
        
        if 'batch_size' in config:
            self.batch_size = config['batch_size']
        
        # Update feature selection
        if 'feature_names' in config:
            self._feature_names = config['feature_names']
        
        # Update config dictionary
        self.config.update(config)
        
        logger.info(f"Configured RL model with: {config}")