"""
Feature providers for user acceptance models.

This module provides various feature providers that can enrich the
feature set used by user acceptance models.
"""
from typing import Dict, Any, Optional, List
from datetime import datetime
import logging
import abc

from drt_sim.models.request import Request
from drt_sim.models.user import UserProfile

logger = logging.getLogger(__name__)

class FeatureProvider(abc.ABC):
    """
    Abstract base class for feature providers.
    
    Feature providers are responsible for extracting or computing
    specific features that can be used by user acceptance models.
    """
    
    @abc.abstractmethod
    def get_features(
        self,
        request: Optional[Request],
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Get features from this provider.
        
        Args:
            request: The transportation request
            context: Dictionary containing context information
            
        Returns:
            Dict[str, Any]: Dictionary of features
        """
        pass
    
    def get_feature_names(self) -> List[str]:
        """
        Get the names of features provided by this provider.
        
        Returns:
            List[str]: List of feature names
        """
        pass

class TimeBasedFeatureProvider(FeatureProvider):
    """
    Provider for time-based features.
    
    This provider extracts features related to time, such as
    time of day, day of week, is_weekend, is_holiday, etc.
    """
    
    def __init__(self, holidays=None):
        """
        Initialize the time-based feature provider.
        
        Args:
            holidays: List of holiday dates
        """
        self.holidays = holidays or []
    
    def get_features(
        self,
        request: Optional[Request],
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Get time-based features.
        
        Args:
            request: The transportation request
            context: Dictionary containing context information
            
        Returns:
            Dict[str, Any]: Dictionary of time-based features
        """
        features = {}
        
        # Get reference time
        reference_time = None
        if "proposed_pickup_time" in context:
            reference_time = context["proposed_pickup_time"]
        elif request and hasattr(request, "request_time"):
            reference_time = request.request_time
        elif "current_time" in context:
            reference_time = context["current_time"]
        
        if reference_time:
            # Extract time of day (hour)
            features["time_of_day"] = reference_time.hour + reference_time.minute / 60.0
            
            # Extract day of week (0 = Monday, 6 = Sunday)
            features["day_of_week"] = reference_time.weekday()
            
            # Is weekend
            features["is_weekend"] = 1.0 if features["day_of_week"] >= 5 else 0.0
            
            # Time periods
            hour = reference_time.hour
            if 6 <= hour < 10:
                features["time_period"] = "morning_peak"
            elif 10 <= hour < 16:
                features["time_period"] = "midday"
            elif 16 <= hour < 19:
                features["time_period"] = "evening_peak"
            else:
                features["time_period"] = "night"
            
            # Is holiday
            features["is_holiday"] = 0.0
            if self.holidays:
                reference_date = reference_time.date()
                if reference_date in self.holidays:
                    features["is_holiday"] = 1.0
        
        return features
    
    def get_feature_names(self) -> List[str]:
        """
        Get the names of features provided by this provider.
        
        Returns:
            List[str]: List of feature names
        """
        return [
            "time_of_day",
            "day_of_week",
            "is_weekend",
            "time_period",
            "is_holiday"
        ]

class SpatialFeatureProvider(FeatureProvider):
    """
    Provider for spatial features.
    
    This provider extracts features related to location, such as
    distance to pickup, distance to destination, is_urban, etc.
    """
    
    def __init__(self, urban_areas=None, region_info=None):
        """
        Initialize the spatial feature provider.
        
        Args:
            urban_areas: Dictionary mapping area IDs to urban classification
            region_info: Dictionary containing region-specific information
        """
        self.urban_areas = urban_areas or {}
        self.region_info = region_info or {}
    
    def get_features(
        self,
        request: Optional[Request],
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Get spatial features.
        
        Args:
            request: The transportation request
            context: Dictionary containing context information
            
        Returns:
            Dict[str, Any]: Dictionary of spatial features
        """
        features = {}
        
        # Direct distance (if provided)
        if "direct_distance" in context:
            features["direct_distance"] = context["direct_distance"]
        
        # Calculate distance to pickup
        if "distance_to_pickup" in context:
            features["distance_to_pickup"] = context["distance_to_pickup"]
        
        # Urban classification
        if request and hasattr(request, "origin_area_id") and request.origin_area_id in self.urban_areas:
            features["is_urban_origin"] = 1.0 if self.urban_areas[request.origin_area_id] == "urban" else 0.0
        
        if request and hasattr(request, "destination_area_id") and request.destination_area_id in self.urban_areas:
            features["is_urban_destination"] = 1.0 if self.urban_areas[request.destination_area_id] == "urban" else 0.0
        
        # Region density
        if request and hasattr(request, "origin_area_id") and request.origin_area_id in self.region_info:
            region = self.region_info[request.origin_area_id]
            if "population_density" in region:
                features["origin_population_density"] = region["population_density"]
        
        return features
    
    def get_feature_names(self) -> List[str]:
        """
        Get the names of features provided by this provider.
        
        Returns:
            List[str]: List of feature names
        """
        return [
            "direct_distance",
            "distance_to_pickup",
            "is_urban_origin",
            "is_urban_destination",
            "origin_population_density"
        ]

class UserHistoryFeatureProvider(FeatureProvider):
    """
    Provider for user history features.
    
    This provider extracts features related to user history, such as
    historical acceptance rate, number of trips, etc.
    """
    
    def get_features(
        self,
        request: Optional[Request],
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Get user history features.
        
        Args:
            request: The transportation request
            context: Dictionary containing context information
            
        Returns:
            Dict[str, Any]: Dictionary of user history features
        """
        features = {}
        
        # Get user profile from context
        user_profile = context.get("user_profile")
        
        if user_profile:
            # Historical acceptance rate (if available)
            if hasattr(user_profile, "get_acceptance_rate") and callable(getattr(user_profile, "get_acceptance_rate")):
                features["historical_acceptance_rate"] = user_profile.get_acceptance_rate()
            
            # Number of past trips (if available)
            if hasattr(user_profile, "get_trip_count") and callable(getattr(user_profile, "get_trip_count")):
                features["trip_count"] = user_profile.get_trip_count()
                # Normalize by assuming 100 trips is "experienced"
                features["user_experience"] = min(features["trip_count"] / 100.0, 1.0)
            
            # Average trip distance (if available)
            if hasattr(user_profile, "get_average_trip_distance") and callable(getattr(user_profile, "get_average_trip_distance")):
                features["average_trip_distance"] = user_profile.get_average_trip_distance()
            
            # User type (if available)
            if hasattr(user_profile, "user_type"):
                features["user_type"] = user_profile.user_type
        
        return features
    
    def get_feature_names(self) -> List[str]:
        """
        Get the names of features provided by this provider.
        
        Returns:
            List[str]: List of feature names
        """
        return [
            "historical_acceptance_rate",
            "trip_count",
            "user_experience",
            "average_trip_distance",
            "user_type"
        ]

class WeatherFeatureProvider(FeatureProvider):
    """
    Provider for weather-related features.
    
    This provider extracts features related to weather conditions.
    """
    
    def __init__(self, weather_service=None):
        """
        Initialize the weather feature provider.
        
        Args:
            weather_service: Service to fetch weather information
        """
        self.weather_service = weather_service
    
    def get_features(
        self,
        request: Optional[Request],
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Get weather features.
        
        Args:
            request: The transportation request
            context: Dictionary containing context information
            
        Returns:
            Dict[str, Any]: Dictionary of weather features
        """
        features = {}
        
        # If weather information is already in context, use it
        if "weather" in context:
            weather = context["weather"]
            features.update(weather)
            return features
        
        # If weather service is available, fetch weather information
        if self.weather_service and request and hasattr(request, "origin_location"):
            try:
                # Get reference time
                reference_time = None
                if "proposed_pickup_time" in context:
                    reference_time = context["proposed_pickup_time"]
                elif hasattr(request, "request_time"):
                    reference_time = request.request_time
                
                # Fetch weather information
                weather = self.weather_service.get_weather(
                    location=request.origin_location,
                    time=reference_time
                )
                
                # Add weather features
                if weather:
                    features["temperature"] = weather.get("temperature", 0) / 40.0  # Normalize to approx. [-1, 1]
                    features["precipitation"] = min(weather.get("precipitation", 0) / 10.0, 1.0)  # Normalize to [0, 1]
                    features["wind_speed"] = min(weather.get("wind_speed", 0) / 30.0, 1.0)  # Normalize to [0, 1]
                    
                    # Weather condition code
                    condition = weather.get("condition", "clear")
                    condition_map = {
                        "clear": 0,
                        "partly_cloudy": 1,
                        "cloudy": 2,
                        "light_rain": 3,
                        "rain": 4,
                        "heavy_rain": 5,
                        "snow": 6,
                        "fog": 7,
                        "storm": 8
                    }
                    features["weather_condition"] = condition_map.get(condition, 0)
                    
                    # Bad weather flag
                    bad_weather_conditions = ["heavy_rain", "snow", "storm"]
                    features["is_bad_weather"] = 1.0 if condition in bad_weather_conditions else 0.0
            
            except Exception as e:
                logger.warning(f"Failed to fetch weather information: {e}")
        
        return features
    
    def get_feature_names(self) -> List[str]:
        """
        Get the names of features provided by this provider.
        
        Returns:
            List[str]: List of feature names
        """
        return [
            "temperature",
            "precipitation",
            "wind_speed",
            "weather_condition",
            "is_bad_weather"
        ]

class ServiceQualityFeatureProvider(FeatureProvider):
    """
    Provider for service quality features.
    
    This provider extracts features related to service quality, such as
    vehicle comfort, driver rating, etc.
    """
    
    def get_features(
        self,
        request: Optional[Request],
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Get service quality features.
        
        Args:
            request: The transportation request
            context: Dictionary containing context information
            
        Returns:
            Dict[str, Any]: Dictionary of service quality features
        """
        features = {}
        
        # Extract vehicle information
        if "vehicle" in context:
            vehicle = context["vehicle"]
            
            if hasattr(vehicle, "comfort_level"):
                # Normalize to [0, 1] assuming 5-star scale
                features["vehicle_comfort"] = vehicle.comfort_level / 5.0
            
            if hasattr(vehicle, "capacity"):
                features["vehicle_capacity"] = vehicle.capacity
                
            if hasattr(vehicle, "vehicle_type"):
                features["vehicle_type"] = vehicle.vehicle_type
        
        # Extract driver information
        if "driver" in context:
            driver = context["driver"]
            
            if hasattr(driver, "rating"):
                # Normalize to [0, 1] assuming 5-star scale
                features["driver_rating"] = driver.rating / 5.0
            
            if hasattr(driver, "experience"):
                # Normalize to [0, 1] assuming 10 years is maximum
                features["driver_experience"] = min(driver.experience / 10.0, 1.0)
        
        # Detour ratio (if not already extracted)
        if "detour_ratio" not in features and "direct_distance" in context and "actual_distance" in context:
            direct_distance = context["direct_distance"]
            actual_distance = context["actual_distance"]
            
            if direct_distance > 0:
                features["detour_ratio"] = (actual_distance / direct_distance) - 1.0
        
        # Extract occupancy
        if "occupancy" in context:
            features["vehicle_occupancy"] = context["occupancy"]
        elif "current_occupancy" in context and "max_occupancy" in context:
            features["vehicle_occupancy"] = context["current_occupancy"] / context["max_occupancy"]
        
        return features
    
    def get_feature_names(self) -> List[str]:
        """
        Get the names of features provided by this provider.
        
        Returns:
            List[str]: List of feature names
        """
        return [
            "vehicle_comfort",
            "vehicle_capacity",
            "vehicle_type",
            "driver_rating",
            "driver_experience",
            "detour_ratio",
            "vehicle_occupancy"
        ]

class FeatureProviderRegistry:
    """
    Registry for feature providers.
    
    This class manages and coordinates multiple feature providers.
    """
    
    def __init__(self):
        """
        Initialize the feature provider registry.
        """
        self.providers = {}
    
    def register_provider(self, name: str, provider: FeatureProvider) -> None:
        """
        Register a feature provider.
        
        Args:
            name: Provider name
            provider: Feature provider instance
        """
        self.providers[name] = provider
        logger.info(f"Registered feature provider: {name}")
    
    def get_features(
        self,
        request: Optional[Request],
        context: Dict[str, Any],
        provider_names: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Get features from specified providers.
        
        Args:
            request: The transportation request
            context: Dictionary containing context information
            provider_names: Names of providers to use (None = all)
            
        Returns:
            Dict[str, Any]: Dictionary of features
        """
        features = {}
        
        # Determine which providers to use
        providers_to_use = provider_names or self.providers.keys()
        
        # Get features from each provider
        for name in providers_to_use:
            if name in self.providers:
                provider_features = self.providers[name].get_features(request, context)
                features.update(provider_features)
        
        return features
    
    def get_all_feature_names(self) -> List[str]:
        """
        Get names of all available features.
        
        Returns:
            List[str]: List of all feature names
        """
        all_names = set()
        
        for provider in self.providers.values():
            names = provider.get_feature_names()
            all_names.update(names)
        
        return sorted(list(all_names))
    
    def get_provider_names(self) -> List[str]:
        """
        Get names of registered providers.
        
        Returns:
            List[str]: List of provider names
        """
        return list(self.providers.keys())