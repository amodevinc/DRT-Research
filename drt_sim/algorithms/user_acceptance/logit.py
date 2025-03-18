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
import json
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import traceback
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
        
        # Keep track of feature types
        self.categorical_features = set()
        self.numeric_features = set()
        
        # Initialize the logistic regression model
        self.logistic_model = LogisticRegression(
            solver='lbfgs',
            max_iter=1000,
            class_weight='balanced',
            random_state=42
        )
        
        # Feature processors
        self.feature_processor = None
        self.scaler = StandardScaler()
        
        # The complete model pipeline
        self.model = None
        
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
            "price": -2.5,
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
            "price",
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
    
    def _detect_feature_types(self, features_dict):
        """
        Detect which features are categorical and which are numeric.
        
        Args:
            features_dict: Dictionary of features
        """
        for name, value in features_dict.items():
            if name not in self.numeric_features and name not in self.categorical_features:
                if isinstance(value, (int, float)):
                    self.numeric_features.add(name)
                elif isinstance(value, str):
                    self.categorical_features.add(name)
    
    def _setup_feature_processing(self):
        """
        Set up the feature processing pipeline based on detected feature types.
        """
        if not self.feature_names:
            return
            
        transformers = []
        
        # Add categorical feature processing if needed
        categorical_features = [feature for feature in self.categorical_features if feature in self.feature_names]
        if categorical_features:
            cat_transformer = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
            transformers.append(('cat', cat_transformer, categorical_features))
        
        # Add numeric feature processing
        numeric_features = [feature for feature in self.numeric_features if feature in self.feature_names]
        if numeric_features:
            num_transformer = StandardScaler()
            transformers.append(('num', num_transformer, numeric_features))
        
        # Create column transformer
        self.feature_processor = ColumnTransformer(
            transformers=transformers,
            remainder='drop'  # Drop any columns not explicitly specified
        )
        
        # Create the full pipeline
        self.model = Pipeline([
            ('preprocessor', self.feature_processor),
            ('classifier', self.logistic_model)
        ])
    
    def _prepare_features_df(self, features_dict):
        """
        Convert features dictionary to pandas DataFrame for processing.
        
        Args:
            features_dict: Dictionary of features
            
        Returns:
            pd.DataFrame: DataFrame with features
        """
        # Create a single-row DataFrame
        df = pd.DataFrame([features_dict])
        
        # Ensure all expected columns are present
        missing_features = []
        for feature in self.feature_names:
            if feature not in df.columns:
                df[feature] = 0.0
                missing_features.append(feature)
        
        if missing_features:
            logger.warning(f"Missing features filled with 0.0: {missing_features}")
            
        # Validate categorical and numeric features
        for feature in self.categorical_features:
            if feature not in df.columns:
                logger.warning(f"Categorical feature '{feature}' not found in input features")
                
        for feature in self.numeric_features:
            if feature not in df.columns:
                logger.warning(f"Numeric feature '{feature}' not found in input features")
        
        return df
    
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
        
        # Detect feature types
        self._detect_feature_types(features_dict)
        
        # If model is not trained, use default logit function
        if not self.is_trained or not self.model:
            return self._calculate_default_probability(features_dict, context.user_profile)
            
        # Convert to DataFrame for processing
        features_df = self._prepare_features_df(features_dict)
        
        # Calculate probability using the pipeline
        try:
            probability = self.model.predict_proba(features_df)[0, 1]
            return float(probability)
        except Exception as e:
            logger.error(f"Error calculating probability with model: {e}")
            logger.exception(e)
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
            for feature, weight in user_profile.weights.items():
                if feature in coefficients:
                    coefficients[feature] = weight
        
        # Calculate utility using coefficients
        utility = 0.0
        for feature_name, value in features.items():
            if feature_name in coefficients and isinstance(value, (int, float)):
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
            # Extract features and labels
            features_list = []
            labels = []
            
            for sample in self.training_data:
                features_list.append(sample['features'])
                labels.append(1 if sample['accepted'] else 0)
            
            # Create DataFrame for training
            features_df = pd.DataFrame(features_list)
            
            if features_df.empty:
                logger.warning("Empty features DataFrame, cannot retrain model")
                return
                
            # Log feature information for debugging
            logger.debug(f"Training with features: {features_df.columns.tolist()}")
            
            # Update feature names
            self.feature_names = list(features_df.columns)
            
            # Detect feature types for all training data
            for features_dict in features_list:
                self._detect_feature_types(features_dict)
            
            # Log detected feature types
            logger.debug(f"Categorical features: {self.categorical_features}")
            logger.debug(f"Numeric features: {self.numeric_features}")
            
            # Set up feature processing pipeline
            self._setup_feature_processing()
            
            # Train the model
            self.model.fit(features_df, labels)
            
            self.is_trained = True
            logger.info(f"Retrained logit model with {len(self.training_data)} samples")
            
        except KeyError as e:
            logger.error(f"Feature not found in training data: {e}")
            logger.debug(f"Available features: {features_df.columns.tolist() if 'features_df' in locals() else 'No DataFrame created'}")
            logger.error(f"Error retraining logit model: {traceback.format_exc()}")
        except ValueError as e:
            logger.error(f"Value error in model training: {e}")
            logger.error(f"Error retraining logit model: {traceback.format_exc()}")
        except Exception as e:
            logger.error(f"Unexpected error retraining logit model: {e}")
            logger.error(f"Error retraining logit model: {traceback.format_exc()}")
    
    def _get_feature_vector(self, features: Dict[str, Any]) -> np.ndarray:
        """
        Convert feature dictionary to feature vector.
        
        This method is maintained for backward compatibility but is no longer the primary
        feature processing method. It is used as a fallback or for specific operations.
        
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
                # Handle different feature types
                if isinstance(features[name], (int, float)):
                    # Numeric features
                    feature_vector[i] = features[name]
                elif isinstance(features[name], str):
                    # Skip categorical features for legacy method
                    logger.debug(f"Skipping categorical feature '{name}' with value '{features[name]}'")
                else:
                    # Other types
                    logger.debug(f"Skipping feature '{name}' with unsupported type: {type(features[name])}")
        
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
    
    def save_model(self, filepath: str, metadata: Optional[Dict[str, Any]] = None) -> None:
        """
        Save the model to a file.
        
        Args:
            filepath: Path where the model should be saved
            metadata: Optional dictionary with additional metadata to save with the model
        
        Raises:
            IOError: If the model cannot be saved to the specified path
        """
        # Create custom metadata specific to LogitModel
        logit_metadata = {
            'is_trained': self.is_trained,
            'feature_names': self.feature_names,
            'categorical_features': list(self.categorical_features),
            'numeric_features': list(self.numeric_features),
            'default_coefficients': self.default_coefficients,
            'required_features': self._required_features,
            'optional_features': self._optional_features
        }
        
        # Merge with user-provided metadata
        if metadata:
            logit_metadata.update(metadata)
        
        # Call the parent class implementation
        super().save_model(filepath, logit_metadata)
        
        try:
            # Save additional model-specific components that aren't handled by pickle
            # For LogitModel, we need to save the scikit-learn model and scaler separately
            model_components_path = f"{filepath}.components"
            model_components = {
                'model': self.model,
                'logistic_model': self.logistic_model,
                'feature_processor': self.feature_processor,
                'scaler': self.scaler
            }
            
            with open(model_components_path, 'wb') as f:
                pickle.dump(model_components, f)
                
            logger.info(f"Saved logit model components to {model_components_path}")
        except Exception as e:
            logger.error(f"Error saving logit model components: {e}")
            raise IOError(f"Failed to save model components: {str(e)}")
    
    @classmethod
    def load_model(cls, filepath: str) -> 'LogitModel':
        """
        Load a model from a file.
        
        Args:
            filepath: Path to the saved model
            
        Returns:
            LogitModel: The loaded model
            
        Raises:
            IOError: If the model cannot be loaded from the specified path
        """
        # First load the base model using the parent class method
        model = super().load_model(filepath)
        
        # Load additional model-specific components
        try:
            model_components_path = f"{filepath}.components"
            if os.path.exists(model_components_path):
                with open(model_components_path, 'rb') as f:
                    components = pickle.load(f)
                    
                # Update model with components
                model.model = components.get('model')
                model.logistic_model = components.get('logistic_model')
                model.feature_processor = components.get('feature_processor') 
                model.scaler = components.get('scaler')
                
                logger.info(f"Loaded logit model components from {model_components_path}")
            
            # Load metadata to update model attributes
            metadata_path = f"{filepath}.meta.json"
            if os.path.exists(metadata_path):
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
                
                # Update model attributes from metadata
                if 'feature_names' in metadata:
                    model.feature_names = metadata['feature_names']
                if 'categorical_features' in metadata:
                    model.categorical_features = set(metadata['categorical_features'])
                if 'numeric_features' in metadata:
                    model.numeric_features = set(metadata['numeric_features'])
                if 'is_trained' in metadata:
                    model.is_trained = metadata['is_trained']
                if 'default_coefficients' in metadata:
                    model.default_coefficients = metadata['default_coefficients']
                if 'required_features' in metadata:
                    model._required_features = metadata['required_features']
                if 'optional_features' in metadata:
                    model._optional_features = metadata['optional_features']
            
            return model
        except Exception as e:
            logger.error(f"Error loading logit model components: {e}")
            raise IOError(f"Failed to load model components: {str(e)}")
    
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
                self.logistic_model.solver = params['solver']
            if 'max_iter' in params:
                self.logistic_model.max_iter = params['max_iter']
            if 'class_weight' in params:
                self.logistic_model.class_weight = params['class_weight']
        
        # Update feature lists
        if 'required_features' in config:
            self._required_features = config['required_features']
        if 'optional_features' in config:
            self._optional_features = config['optional_features']
        
        # Update the config
        self.config.update(config)
        
        logger.info(f"Configured logit model with: {config}")