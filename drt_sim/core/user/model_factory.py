"""
Factory for creating user acceptance models.

This module provides a factory for creating various user acceptance models,
making it easy to switch between different models.
"""
from typing import Dict, Any, Optional, List, Type
import logging
import importlib
import os

from drt_sim.algorithms.base_interfaces.user_acceptance_base import UserAcceptanceModel
from drt_sim.core.user.feature_extractor import FeatureExtractor
from drt_sim.core.user.feature_provider import FeatureProviderRegistry

logger = logging.getLogger(__name__)

class ModelFactory:
    """
    Factory for creating user acceptance models.
    
    This class provides methods for creating and configuring
    various user acceptance models.
    """
    
    # Registry of available model types
    _model_registry = {}
    
    @classmethod
    def register_model_type(cls, name: str, model_class: Type[UserAcceptanceModel]) -> None:
        """
        Register a new model type.
        
        Args:
            name: Model type name
            model_class: Model class
        """
        cls._model_registry[name] = model_class
        logger.info(f"Registered user acceptance model type: {name}")
    
    @classmethod
    def create_model(
        cls,
        model_type: str,
        config: Optional[Dict[str, Any]] = None,
        feature_extractor: Optional[FeatureExtractor] = None,
        feature_provider_registry: Optional[FeatureProviderRegistry] = None,
        **kwargs
    ) -> UserAcceptanceModel:
        """
        Create a user acceptance model.
        
        Args:
            model_type: Type of model to create
            config: Configuration parameters
            feature_extractor: Feature extractor to use
            feature_provider_registry: Feature provider registry to use
            **kwargs: Additional parameters
            
        Returns:
            UserAcceptanceModel: The created model
            
        Raises:
            ValueError: If the model type is not supported
        """
        # Get configuration
        config = config or {}
        
        # Create feature extractor if not provided
        if not feature_extractor:
            feature_registry = config.get("feature_registry", None)
            enabled_features = config.get("enabled_features", None)
            normalization_overrides = config.get("normalization_overrides", None)
            feature_extractor = FeatureExtractor(
                feature_registry=feature_registry,
                enabled_features=enabled_features,
                normalization_overrides=normalization_overrides
            )
        
        # Try to get model class from registry
        model_class = cls._model_registry.get(model_type)
        
        # If not in registry, try to load dynamically
        if not model_class:
            try:
                # Import the model class dynamically
                module_path = f"drt_sim.algorithms.user_acceptance.{model_type}"
                module = importlib.import_module(module_path)
                
                # Get the model class (assuming it follows naming convention)
                class_name = "".join(word.capitalize() for word in model_type.split("_")) + "Model"
                model_class = getattr(module, class_name)
                
                # Register the model type for future use
                cls.register_model_type(model_type, model_class)
            except (ImportError, AttributeError) as e:
                logger.error(f"Failed to load model type {model_type}: {str(e)}")
                raise ValueError(f"Unsupported model type: {model_type}")
        
        # Create model instance with feature extractor
        model = model_class(
            feature_extractor=feature_extractor,
            feature_provider_registry=feature_provider_registry,  # Pass the registry
            **kwargs
        )
        
        # Configure model if configuration is provided
        if config:
            model.configure(config)
        
        # Load pre-trained model if specified
        if "model_path" in config:
            model_path = config["model_path"]
            if hasattr(model, "load_model") and os.path.exists(model_path):
                model.load_model(model_path)
                logger.info(f"Loaded user acceptance model from {model_path}")
            else:
                logger.warning(f"Model path {model_path} specified but could not load model")
        
        logger.info(f"Created user acceptance model of type: {model_type}")
        return model
    
    @classmethod
    def get_available_model_types(cls) -> List[str]:
        """
        Get the list of available model types.
        
        Returns:
            List[str]: List of available model types
        """
        return list(cls._model_registry.keys())
    
    @classmethod
    def get_model_class(cls, model_type: str) -> Type[UserAcceptanceModel]:
        """
        Get the class for a model type.
        
        Args:
            model_type: Model type name
            
        Returns:
            Type[UserAcceptanceModel]: Model class
            
        Raises:
            ValueError: If the model type is not supported
        """
        if model_type in cls._model_registry:
            return cls._model_registry[model_type]
        
        # Try to load dynamically
        try:
            # Import the model class dynamically
            module_path = f"drt_sim.algorithms.user_acceptance.{model_type}"
            module = importlib.import_module(module_path)
            
            # Get the model class (assuming it follows naming convention)
            class_name = "".join(word.capitalize() for word in model_type.split("_")) + "Model"
            model_class = getattr(module, class_name)
            
            # Register the model type for future use
            cls.register_model_type(model_type, model_class)
            
            return model_class
        except (ImportError, AttributeError) as e:
            logger.error(f"Failed to load model type {model_type}: {str(e)}")
            raise ValueError(f"Unsupported model type: {model_type}")
    
    @classmethod
    def create_default_models(cls) -> Dict[str, UserAcceptanceModel]:
        """
        Create default instances of all available models.
        
        Returns:
            Dict[str, UserAcceptanceModel]: Dictionary of model type to model instance
        """
        default_models = {}
        
        # Try to register some common models
        try:
            from drt_sim.algorithms.user_acceptance.default import DefaultModel
            cls.register_model_type("default", DefaultModel)
        except ImportError:
            pass
        
        try:
            from drt_sim.algorithms.user_acceptance.logit import LogitModel
            cls.register_model_type("logit", LogitModel)
        except ImportError:
            pass
        
        try:
            from drt_sim.algorithms.user_acceptance.rl import RLAcceptanceModel
            cls.register_model_type("rl", RLAcceptanceModel)
        except ImportError:
            pass
        
        # Create instances of all registered models
        for model_type in cls.get_available_model_types():
            try:
                default_models[model_type] = cls.create_model(model_type)
            except Exception as e:
                logger.warning(f"Failed to create default model of type {model_type}: {str(e)}")
        
        return default_models