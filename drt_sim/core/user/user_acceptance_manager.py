"""
User acceptance management for the DRT system.

This module provides functionality for managing user acceptance models
and coordinating user acceptance decisions in the DRT system.
"""
from typing import Dict, Any, Optional, List, Tuple
from datetime import datetime, timedelta
import logging
import os
import traceback

from drt_sim.models.request import Request
from drt_sim.algorithms.base_interfaces.user_acceptance_base import UserAcceptanceModel
from drt_sim.core.user.user_profile_manager import UserProfileManager
from drt_sim.core.user.feature_extractor import FeatureExtractor
from drt_sim.core.user.feature_provider import FeatureProviderRegistry
from drt_sim.core.user.model_factory import ModelFactory
from drt_sim.core.user.acceptance_context import AcceptanceContext
from drt_sim.config.config import UserAcceptanceConfig

logger = logging.getLogger(__name__)

class UserAcceptanceManager:
    """
    Manages user acceptance models and decisions in the DRT system.
    
    This class is responsible for loading and managing user acceptance models,
    making acceptance decisions, and tracking acceptance metrics.
    """
    
    def __init__(
        self,
        config: UserAcceptanceConfig,
        user_profile_manager: UserProfileManager
    ):
        """
        Initialize the user acceptance manager.
        
        Args:
            config: Configuration for user acceptance
            user_profile_manager: Manager for user profiles
        """
        logger.info("Initializing UserAcceptanceManager")
        self.config = config
        self.user_profile_manager = user_profile_manager
        
        logger.debug("Initializing feature providers")
        self.feature_provider_registry = self._initialize_feature_providers()
        logger.debug(f"Initialized {len(self.feature_provider_registry.providers)} feature providers")
        
        logger.debug("Initializing feature extractor")
        self.feature_extractor = self._initialize_feature_extractor()
        
        logger.debug("Initializing acceptance model")
        self.model = self._initialize_model()
        logger.info(f"Using acceptance model: {self.model.__class__.__name__}")
        
        self.acceptance_history: List[Dict[str, Any]] = []
        self.acceptance_metrics: Dict[str, Any] = {
            "total_requests": 0,
            "accepted_requests": 0,
            "rejected_requests": 0,
            "acceptance_rate": 0.0,
            "by_user_type": {},
            "by_time_of_day": {},
            "by_waiting_time": {}
        }
        logger.info("UserAcceptanceManager initialization complete")
    
    def _initialize_feature_extractor(self) -> FeatureExtractor:
        """
        Initialize the feature extractor from configuration.
        
        Returns:
            FeatureExtractor: The initialized feature extractor
        """
        feature_extractor_config = self.config.feature_extractor_config
        
        # Extract configuration parameters
        feature_registry = feature_extractor_config.get("feature_registry", None)
        enabled_features = feature_extractor_config.get("enabled_features", None)
        normalization_overrides = feature_extractor_config.get("normalization_overrides", None)
        
        if enabled_features:
            logger.debug(f"Enabled features: {', '.join(enabled_features)}")
        
        # Create the feature extractor
        extractor = FeatureExtractor(
            feature_registry=feature_registry,
            enabled_features=enabled_features,
            normalization_overrides=normalization_overrides
        )
        logger.debug("Feature extractor initialized successfully")
        return extractor
    
    def _initialize_feature_providers(self) -> FeatureProviderRegistry:
        """
        Initialize feature providers from configuration.
        
        Returns:
            FeatureProviderRegistry: The initialized feature provider registry
        """
        from drt_sim.core.user.feature_provider import (
            TimeBasedFeatureProvider,
            UserHistoryFeatureProvider,
            WeatherFeatureProvider
        )
        
        # Create the registry
        registry = FeatureProviderRegistry()
        
        # Get feature provider configurations
        provider_config = self.config.feature_provider_config
        
        # Register time-based feature provider if enabled
        if provider_config.get("time_based", {}).get("enabled", True):
            time_config = provider_config.get("time_based", {})
            holidays = time_config.get("holidays", [])
            logger.debug(f"Registering time-based feature provider with {len(holidays)} holidays")
            registry.register_provider(
                "time",
                TimeBasedFeatureProvider(holidays=holidays)
            )
        
        # Register user history feature provider if enabled
        if provider_config.get("user_history", {}).get("enabled", True):
            logger.debug("Registering user history feature provider")
            registry.register_provider(
                "user_history",
                UserHistoryFeatureProvider()
            )
        
        # Register weather feature provider if enabled
        if provider_config.get("weather", {}).get("enabled", False):
            weather_config = provider_config.get("weather", {})
            weather_service = weather_config.get("service", None)
            if weather_service:
                logger.debug(f"Registering weather feature provider with service: {weather_service}")
                registry.register_provider(
                    "weather",
                    WeatherFeatureProvider(weather_service=weather_service)
                )

        # Register custom providers if specified
        custom_providers = provider_config.get("custom_providers", [])
        for provider_info in custom_providers:
            if "name" in provider_info and "class" in provider_info:
                try:
                    # Dynamically import and instantiate the provider
                    provider_class = self._import_class(provider_info["class"])
                    provider_args = provider_info.get("args", {})
                    logger.debug(f"Registering custom provider '{provider_info['name']}' with class {provider_info['class']}")
                    provider = provider_class(**provider_args)
                    registry.register_provider(provider_info["name"], provider)
                except Exception as e:
                    logger.error(f"Error registering custom provider {provider_info['name']}: {e}")
                    logger.debug(f"Provider registration stack trace: {traceback.format_exc()}")
        
        return registry
    
    def _import_class(self, class_path: str) -> type:
        """
        Import a class from a module path string.
        
        Args:
            class_path: String in format "module.submodule.ClassName"
            
        Returns:
            type: The imported class
        """
        logger.debug(f"Importing class from path: {class_path}")
        module_path, class_name = class_path.rsplit(".", 1)
        module = __import__(module_path, fromlist=[class_name])
        return getattr(module, class_name)
    
    def _initialize_model(self) -> UserAcceptanceModel:
        """
        Initialize the user acceptance model based on configuration.
        
        Returns:
            UserAcceptanceModel: The initialized user acceptance model
        """
        model_config = self.config.model
        model_type = model_config.get("type", "default")
        model_params = model_config.get("parameters", {})
        
        logger.info(f"Initializing acceptance model of type '{model_type}'")
        logger.debug(f"Model parameters: {model_params}")
        
        try:
            # Use ModelFactory to create the model
            model = ModelFactory.create_model(
                model_type=model_type,
                config=model_params,
                feature_extractor=self.feature_extractor,
                feature_provider_registry=self.feature_provider_registry
            )
            
            # Load pre-trained model if specified
            if "model_path" in model_config:
                model_path = model_config["model_path"]
                if os.path.exists(model_path) and hasattr(model, "load_model"):
                    logger.info(f"Loading pre-trained model from {model_path}")
                    model.load_model(model_path)
                    logger.info(f"Successfully loaded user acceptance model from {model_path}")
                else:
                    logger.warning(f"Model path {model_path} does not exist or model doesn't support loading, using untrained model")
            
            return model
        
        except Exception as e:
            logger.error(f"Error initializing user acceptance model of type '{model_type}': {str(e)}")
            logger.error(f"Stack trace: {traceback.format_exc()}")
    def calculate_acceptance_probability(
        self,
        request: Request,
        service_attributes: Dict[str, Any]
    ) -> float:
        """
        Calculate the probability of a user accepting a proposed service.
        
        Args:
            request: The transportation request
            service_attributes: Service attributes
            
        Returns:
            float: Probability of acceptance (0.0 to 1.0)
        """
        request_id = getattr(request, "id", "unknown")
        logger.info(f"Calculating acceptance probability for request {request_id}")
        
        try:
            # Get user profile if available
            user_profile = None
            if hasattr(request, "user_id") and request.user_id:
                user_id = request.user_id
                user_profile = self.user_profile_manager.get_profile(user_id)
            
            # Create the acceptance context
            context = AcceptanceContext.from_assignment(
                request=request,
                service_attributes=service_attributes,
                user_profile=user_profile
            )
            # Calculate acceptance probability
            logger.debug("Calling model to calculate acceptance probability")
            probability = self.model.calculate_acceptance_probability(context)
            
            logger.info(f"Calculated acceptance probability: {probability:.4f} for request {request_id}")
            return probability
        
        except Exception as e:
            logger.error(f"Error calculating acceptance probability for request {request_id}: {str(e)}")
            logger.error(f"Stack trace: {traceback.format_exc()}")
            # Default to high probability in case of error
            logger.info("Using default probability of 0.9 due to error")
            return 0.9
    
    def decide_acceptance(
        self,
        request: Request,
        service_attributes: Dict[str, Any]
    ) -> Tuple[bool, float]:
        """
        Decide whether a user will accept a proposed service.
        
        Args:
            request: The transportation request
            service_attributes: Service attributes
            
        Returns:
            Tuple[bool, float]: (acceptance decision, acceptance probability)
        """
        request_id = getattr(request, "id", "unknown")
        logger.info(f"Making acceptance decision for request {request_id}")
        
        try:
            # Get user profile if available
            user_profile = None
            user_id = None
            if request.user_id:
                user_id = request.user_id
                user_profile = self.user_profile_manager.get_profile(user_id)
            
            # Create the acceptance context
            logger.debug(f"Creating AcceptanceContext with {len(service_attributes)} service attributes")
            context = AcceptanceContext.from_assignment(
                request=request,
                service_attributes=service_attributes,
                user_profile=user_profile
            )
            
            # Make acceptance decision
            logger.debug("Calling model to decide acceptance")
            accepted, probability = self.model.decide_acceptance(context)
            
            if accepted:
                logger.info(f"Acceptance decision: ACCEPTED with probability {probability:.4f} for request {request_id}")
            else:
                logger.info(f"Acceptance decision: REJECTED with probability {probability:.4f} for request {request_id}")
            
            # Record decision in history
            logger.debug("Recording decision in history")
            self._record_decision(
                request=request,
                accepted=accepted,
                probability=probability,
                service_attributes=service_attributes
            )
            
            return accepted, probability
        
        except Exception as e:
            logger.error(f"Error deciding acceptance for request {request_id}: {str(e)}")
            logger.error(f"Stack trace: {traceback.format_exc()}")
            # Default to acceptance in case of error
            logger.info("Using default decision (ACCEPTED with probability 0.9) due to error")
            return True, 0.9
    
    def update_model(
        self,
        request: Request,
        accepted: bool,
        service_attributes: Dict[str, Any]
    ) -> None:
        """
        Update the model based on user decisions.
        
        Args:
            request: The transportation request
            accepted: Whether the user accepted the service
            service_attributes: Service attributes
        """
        request_id = getattr(request, "id", "unknown")
        logger.info(f"Updating model with {'acceptance' if accepted else 'rejection'} for request {request_id}")
        
        try:
            # Get user profile if available
            user_profile = None
            user_id = None
            if hasattr(request, "user_id") and request.user_id:
                user_id = request.user_id
                logger.debug(f"Getting user profile for user {user_id}")
                user_profile = self.user_profile_manager.get_profile(user_id)
                
                # Update user profile acceptance rate if available
                if user_profile and hasattr(user_profile, "update_acceptance_rate"):
                    logger.debug(f"Updating acceptance rate in user profile for user {user_id}")
                    user_profile.update_acceptance_rate(accepted)
            
            # Create the acceptance context
            logger.debug(f"Creating AcceptanceContext for model update with {len(service_attributes)} service attributes")
            context = AcceptanceContext.from_assignment(
                request=request,
                service_attributes=service_attributes,
                user_profile=user_profile
            )
            
            # Update the model
            logger.debug("Calling model update method")
            self.model.update_model(context, accepted)
            logger.debug("Model update completed")
            
            # Update metrics
            logger.debug("Updating metrics")
            self._update_metrics(
                request, 
                accepted, 
                service_attributes
            )
            
            logger.info(f"Successfully updated model and metrics for request {request_id}")
        
        except Exception as e:
            logger.error(f"Error updating user acceptance model for request {request_id}: {str(e)}")
            logger.error(f"Stack trace: {traceback.format_exc()}")
    
    def batch_update(self, training_data: List[Dict[str, Any]]) -> None:
        """
        Update the model with batch training data.
        
        Args:
            training_data: List of training examples with features and outcomes
        """
        logger.info(f"Performing batch update with {len(training_data)} examples")
        
        try:
            # Convert training data to contexts if necessary
            processed_data = []
            skipped_examples = 0
            
            for i, example in enumerate(training_data):
                try:
                    if "context" in example:
                        # Already has context
                        logger.debug(f"Example {i} already has context")
                        processed_data.append(example)
                    elif "request" in example and "features" in example and "accepted" in example:
                        # Create context from features
                        logger.debug(f"Creating context for example {i} with {len(example['features'])} features")
                        context = AcceptanceContext(
                            features=example["features"],
                            request=example["request"],
                            user_profile=example.get("user_profile")
                        )
                        processed_data.append({
                            "context": context,
                            "accepted": example["accepted"]
                        })
                    else:
                        logger.warning(f"Skipping training example {i} due to missing data: {example.keys()}")
                        skipped_examples += 1
                except Exception as e:
                    logger.error(f"Error processing training example {i}: {str(e)}")
                    skipped_examples += 1
            
            if skipped_examples > 0:
                logger.warning(f"Skipped {skipped_examples} training examples due to errors or missing data")
            
            # Update the model
            logger.info(f"Updating model with {len(processed_data)} examples")
            self.model.batch_update(processed_data)
            logger.info(f"Successfully updated user acceptance model with {len(processed_data)} examples")
        except Exception as e:
            logger.error(f"Error batch updating user acceptance model: {str(e)}")
            logger.error(f"Stack trace: {traceback.format_exc()}")
    
    def save_model(self, path: str) -> None:
        """
        Save the model to a file.
        
        Args:
            path: Path to save the model
        """
        logger.info(f"Saving user acceptance model to {path}")
        
        try:
            if hasattr(self.model, "save_model"):
                self.model.save_model(path)
                logger.info(f"Successfully saved user acceptance model to {path}")
            else:
                logger.warning(f"Model {self.model.__class__.__name__} does not support saving")
        except Exception as e:
            logger.error(f"Error saving user acceptance model to {path}: {str(e)}")
            logger.error(f"Stack trace: {traceback.format_exc()}")
    
    def get_feature_importance(self) -> Dict[str, float]:
        """
        Get the importance of each feature in the model.
        
        Returns:
            Dict[str, float]: Feature names mapped to their importance values
        """
        logger.info("Getting feature importance from model")
        
        try:
            if hasattr(self.model, "get_feature_importance"):
                importance = self.model.get_feature_importance()
                logger.debug(f"Retrieved importance for {len(importance)} features")
                
                # Log top features by importance
                if importance:
                    top_features = sorted(importance.items(), key=lambda x: x[1], reverse=True)[:5]
                    logger.info(f"Top features by importance: {', '.join([f'{f}:{v:.4f}' for f, v in top_features])}")
                
                return importance
            else:
                logger.warning(f"Model {self.model.__class__.__name__} does not support feature importance")
                return {}
        except Exception as e:
            logger.error(f"Error getting feature importance: {str(e)}")
            logger.error(f"Stack trace: {traceback.format_exc()}")
            return {}
    
    def get_metrics(self) -> Dict[str, Any]:
        """
        Get acceptance metrics.
        
        Returns:
            Dict[str, Any]: Acceptance metrics
        """
        logger.debug("Retrieving acceptance metrics")
        
        # Log summary metrics
        total = self.acceptance_metrics["total_requests"]
        if total > 0:
            acceptance_rate = self.acceptance_metrics["acceptance_rate"] * 100
            logger.info(f"Acceptance metrics - Total: {total}, Rate: {acceptance_rate:.2f}%")
        
        return self.acceptance_metrics
    
    def _record_decision(
        self,
        request: Request,
        accepted: bool,
        probability: float,
        service_attributes: Dict[str, Any]
    ) -> None:
        """
        Record an acceptance decision in history.
        
        Args:
            request: The transportation request
            accepted: Whether the service was accepted
            probability: The calculated acceptance probability
            service_attributes: Service attributes
        """
        request_id = getattr(request, "id", "unknown")
        user_id = getattr(request, "user_id", None)
        
        logger.debug(f"Recording {'acceptance' if accepted else 'rejection'} decision for request {request_id}")
        
        # Create record
        record = {
            "request_id": request_id,
            "user_id": user_id,
            "timestamp": datetime.now().isoformat(),
            "accepted": accepted,
            "probability": probability,
            **service_attributes
        }
        
        # Add to history
        self.acceptance_history.append(record)
        
        # Limit history size
        max_history = self.config.max_history_size
        if len(self.acceptance_history) > max_history:
            logger.debug(f"Trimming acceptance history to {max_history} entries")
            self.acceptance_history = self.acceptance_history[-max_history:]
    
    def _update_metrics(
        self,
        request: Request,
        accepted: bool,
        service_attributes: Dict[str, Any]
    ) -> None:
        """
        Update acceptance metrics.
        
        Args:
            request: The transportation request
            accepted: Whether the service was accepted
            service_attributes: Service attributes that were offered
        """
        request_id = getattr(request, "id", "unknown")
        logger.debug(f"Updating metrics for request {request_id} (accepted: {accepted})")
        
        # Update basic metrics
        self.acceptance_metrics["total_requests"] += 1
        if accepted:
            self.acceptance_metrics["accepted_requests"] += 1
        else:
            self.acceptance_metrics["rejected_requests"] += 1
        
        # Update acceptance rate
        total = self.acceptance_metrics["total_requests"]
        accepted_count = self.acceptance_metrics["accepted_requests"]
        self.acceptance_metrics["acceptance_rate"] = accepted_count / total if total > 0 else 0
        
        # Update by user type
        user_type = "unknown"
        if hasattr(request, "user_id") and request.user_id:
            user_profile = self.user_profile_manager.get_profile(request.user_id)
            if user_profile and hasattr(user_profile, "service_preference"):
                user_type = user_profile.service_preference.value if hasattr(user_profile.service_preference, 'value') else str(user_profile.service_preference)
        
        logger.debug(f"Updating metrics for user type: {user_type}")
        
        if user_type not in self.acceptance_metrics["by_user_type"]:
            self.acceptance_metrics["by_user_type"][user_type] = {
                "total": 0,
                "accepted": 0,
                "rate": 0.0
            }
        
        self.acceptance_metrics["by_user_type"][user_type]["total"] += 1
        if accepted:
            self.acceptance_metrics["by_user_type"][user_type]["accepted"] += 1
        
        user_type_total = self.acceptance_metrics["by_user_type"][user_type]["total"]
        user_type_accepted = self.acceptance_metrics["by_user_type"][user_type]["accepted"]
        self.acceptance_metrics["by_user_type"][user_type]["rate"] = (
            user_type_accepted / user_type_total if user_type_total > 0 else 0
        )
        
        # Update by time of day
        hour = datetime.now().hour
        hour_range = f"{hour:02d}:00-{(hour+1)%24:02d}:00"
        
        logger.debug(f"Updating metrics for time range: {hour_range}")
        
        if hour_range not in self.acceptance_metrics["by_time_of_day"]:
            self.acceptance_metrics["by_time_of_day"][hour_range] = {
                "total": 0,
                "accepted": 0,
                "rate": 0.0
            }
        
        self.acceptance_metrics["by_time_of_day"][hour_range]["total"] += 1
        if accepted:
            self.acceptance_metrics["by_time_of_day"][hour_range]["accepted"] += 1
        
        time_total = self.acceptance_metrics["by_time_of_day"][hour_range]["total"]
        time_accepted = self.acceptance_metrics["by_time_of_day"][hour_range]["accepted"]
        self.acceptance_metrics["by_time_of_day"][hour_range]["rate"] = (
            time_accepted / time_total if time_total > 0 else 0
        )
        
        # Update by waiting time
        waiting_time = service_attributes.get("waiting_time", 0)
        
        # Categorize waiting time
        if waiting_time <= 5:
            wait_category = "0-5 min"
        elif waiting_time <= 10:
            wait_category = "5-10 min"
        elif waiting_time <= 15:
            wait_category = "10-15 min"
        elif waiting_time <= 20:
            wait_category = "15-20 min"
        else:
            wait_category = "20+ min"
        
        logger.debug(f"Updating metrics for waiting time category: {wait_category}")
        
        if wait_category not in self.acceptance_metrics["by_waiting_time"]:
            self.acceptance_metrics["by_waiting_time"][wait_category] = {
                "total": 0,
                "accepted": 0,
                "rate": 0.0
            }
        
        self.acceptance_metrics["by_waiting_time"][wait_category]["total"] += 1
        if accepted:
            self.acceptance_metrics["by_waiting_time"][wait_category]["accepted"] += 1
        
        wait_total = self.acceptance_metrics["by_waiting_time"][wait_category]["total"]
        wait_accepted = self.acceptance_metrics["by_waiting_time"][wait_category]["accepted"]
        self.acceptance_metrics["by_waiting_time"][wait_category]["rate"] = (
            wait_accepted / wait_total if wait_total > 0 else 0
        )