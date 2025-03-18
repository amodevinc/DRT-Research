"""
Base interface for user acceptance models.

This module provides the base interface for user acceptance models
used in the DRT simulation platform.
"""
from typing import Dict, Any, Optional, List, Tuple
import abc
import logging
import os
import json
import pickle

from drt_sim.core.user.acceptance_context import AcceptanceContext
from drt_sim.core.user.feature_extractor import FeatureExtractor
from drt_sim.core.user.feature_provider import FeatureProviderRegistry
logger = logging.getLogger(__name__)

class UserAcceptanceModel(abc.ABC):
    """
    Abstract base class for user acceptance models.
    
    This class defines the interface that all user acceptance models
    must implement.
    """
    
    def __init__(self, feature_extractor: Optional[FeatureExtractor] = None, feature_provider_registry: Optional[FeatureProviderRegistry] = None, **kwargs):
        """
        Initialize the user acceptance model.
        
        Args:
            feature_extractor: Feature extractor to use
            **kwargs: Additional parameters
        """
        self.feature_extractor = feature_extractor or FeatureExtractor()
        self.config = {}
    
    @abc.abstractmethod
    def calculate_acceptance_probability(self, context: AcceptanceContext) -> float:
        """
        Calculate probability of user accepting a proposed service.
        
        Args:
            context: Context containing request, features, and user profile
            
        Returns:
            float: Probability of acceptance (0.0 to 1.0)
        """
        pass
    
    def decide_acceptance(self, context: AcceptanceContext) -> Tuple[bool, float]:
        """
        Decide whether the user will accept the proposed service.
        
        Args:
            context: Context containing request, features, and user profile
            
        Returns:
            Tuple[bool, float]: (acceptance decision, acceptance probability)
        """
        # Default implementation that can be overridden
        import random
        
        # Calculate acceptance probability
        probability = self.calculate_acceptance_probability(context)
        
        # Make decision based on probability
        accepted = random.random() < probability
        
        return accepted, probability
    
    @abc.abstractmethod
    def update_model(self, context: AcceptanceContext, accepted: bool) -> None:
        """
        Update model based on user decisions.
        
        Args:
            context: Context containing request, features, and user profile
            accepted: Whether the user accepted the service
        """
        pass
    
    def batch_update(self, training_data: List[Dict[str, Any]]) -> None:
        """
        Update model with batch training data.
        
        Args:
            training_data: List of training examples with features and outcomes
        """
        # Default implementation that can be overridden
        for example in training_data:
            if "request" in example and "features" in example and "accepted" in example:
                context = AcceptanceContext(
                    request=example["request"],
                    features=example["features"],
                    user_profile=example.get("user_profile")
                )
                self.update_model(context, example["accepted"])
    
    def get_feature_importance(self) -> Dict[str, float]:
        """
        Get the importance of each feature in the model.
        
        Returns:
            Dict[str, float]: Feature names mapped to their importance values
        """
        # Default implementation that should be overridden
        return {name: 0.0 for name in self.feature_extractor.get_feature_names()}
    
    def get_required_features(self) -> List[str]:
        """
        Get the list of required features for this model.
        
        Returns:
            List[str]: List of feature names that are required by this model
        """
        # Default implementation that should be overridden
        return []
    
    def get_optional_features(self) -> List[str]:
        """
        Get the list of optional features for this model.
        
        Returns:
            List[str]: List of feature names that are optional but can improve the model
        """
        # Default implementation that should be overridden
        return []
    
    def configure(self, config: Dict[str, Any]) -> None:
        """
        Configure the model with the given configuration.
        
        Args:
            config: Dictionary containing configuration parameters
        """
        # Update the config dictionary
        self.config.update(config)
        
        logger.info(f"{self.__class__.__name__} configured with: {config}")
        
    def save_model(self, filepath: str, metadata: Optional[Dict[str, Any]] = None) -> None:
        """
        Save the model to a file.
        
        This method provides a default implementation that saves the model using pickle.
        Subclasses may override this method to implement custom serialization.
        
        Args:
            filepath: Path where the model should be saved
            metadata: Optional dictionary with additional metadata to save with the model
        
        Raises:
            IOError: If the model cannot be saved to the specified path
        """
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
        
        # Prepare metadata
        save_metadata = {
            "model_type": self.__class__.__name__,
            "config": self.config,
            "feature_names": self.feature_extractor.get_feature_names(),
        }
        
        # Add user metadata if provided
        if metadata:
            save_metadata.update(metadata)
            
        try:
            # Default implementation uses pickle for the model and JSON for metadata
            # Save metadata as JSON
            metadata_path = f"{filepath}.meta.json"
            with open(metadata_path, 'w') as f:
                json.dump(save_metadata, f, indent=2)
                
            # Save model using pickle
            with open(filepath, 'wb') as f:
                pickle.dump(self, f)
                
            logger.info(f"Model saved to {filepath} with metadata at {metadata_path}")
        except (IOError, pickle.PickleError) as e:
            logger.error(f"Failed to save model to {filepath}: {str(e)}")
            raise IOError(f"Failed to save model: {str(e)}")
            
    @classmethod
    def load_model(cls, filepath: str) -> 'UserAcceptanceModel':
        """
        Load a model from a file.
        
        This method provides a default implementation that loads the model using pickle.
        Subclasses may override this method to implement custom deserialization.
        
        Args:
            filepath: Path to the saved model
            
        Returns:
            UserAcceptanceModel: The loaded model
            
        Raises:
            IOError: If the model cannot be loaded from the specified path
            ValueError: If the loaded model is not of the expected type
        """
        try:
            # Load model using pickle
            with open(filepath, 'rb') as f:
                model = pickle.load(f)
                
            # Check if the loaded model is of the expected type
            if not isinstance(model, cls):
                raise ValueError(f"Loaded model is of type {type(model).__name__}, expected {cls.__name__}")
                
            # Load metadata if available
            metadata_path = f"{filepath}.meta.json"
            if os.path.exists(metadata_path):
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
                    logger.info(f"Loaded model metadata from {metadata_path}")
                    
                    # Update config from metadata
                    if 'config' in metadata:
                        model.config.update(metadata['config'])
            
            logger.info(f"Model loaded from {filepath}")
            return model
        except (IOError, pickle.PickleError) as e:
            logger.error(f"Failed to load model from {filepath}: {str(e)}")
            raise IOError(f"Failed to load model: {str(e)}")
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            raise