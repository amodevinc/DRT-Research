"""
User profile module for DRT simulation.

This module provides a redesigned user profile class that integrates well
with the acceptance modeling framework and focuses on essential attributes.
"""
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, Any, Optional, List
import logging
import json
import os

logger = logging.getLogger(__name__)

class ServicePreference(Enum):
    """Service preferences that influence acceptance weights"""
    SPEED = "speed"                # Prefers faster service
    RELIABILITY = "reliability"    # Prefers reliable arrival times

@dataclass
class UserProfile:
    """
    User profile containing essential attributes for acceptance modeling.
    
    This class represents a user profile with preferences and historical data
    that are relevant to acceptance modeling in DRT systems.
    """
    id: str
    
    # Acceptance preferences
    max_walking_time_to_origin: float = 3.0  # minutes
    max_walking_time_from_destination: float = 3.0  # minutes
    max_waiting_time: float = 10.0  # minutes
    max_in_vehicle_time: float = 25.0   # minutes
    max_cost: float = 30.0          # currency units
    max_acceptable_delay: float = 7.0 # minutes
    
    # User preferences
    service_preference: ServicePreference = ServicePreference.SPEED
    
    # Feature weights for acceptance decisions
    weights: Dict[str, float] = field(default_factory=lambda: {
        "walking_time_to_origin": 0.4,
        "wait_time": 0.3,
        "in_vehicle_time": 0.2,
        "walking_time_from_destination": 0.1,
        "time_of_day": 0.0,
        "day_of_week": 0.0,
        "distance_to_pickup": 0.0
    })
    
    # Historical data
    historical_trips: int = 0
    historical_acceptance_rate: float = 0.0
    historical_ratings: List[float] = field(default_factory=list)
    
    # Manager reference (set by UserProfileManager)
    _manager: Any = None
    
    def __post_init__(self):
        """Initialize after instance creation."""
        # Ensure weights are properly normalized
        weight_sum = sum(self.weights.values())
        if weight_sum > 0:
            self.weights = {k: v / weight_sum for k, v in self.weights.items()}
    
    def get_acceptance_rate(self) -> float:
        """
        Get the historical acceptance rate for this user.
        
        Returns:
            float: Acceptance rate (0.0 to 1.0)
        """
        return self.historical_acceptance_rate
    
    def get_trip_count(self) -> int:
        """
        Get the number of trips taken by this user.
        
        Returns:
            int: Number of trips
        """
        return self.historical_trips
    
    def get_average_rating(self) -> float:
        """
        Get the average rating given by this user.
        
        Returns:
            float: Average rating
        """
        if not self.historical_ratings:
            return 5.0
        return sum(self.historical_ratings) / len(self.historical_ratings)
    
    def update_acceptance_rate(self, accepted: bool) -> None:
        """
        Update the historical acceptance rate with a new decision.
        
        Args:
            accepted: Whether the user accepted the service
        """
        # Simple running average
        if self.historical_trips == 0:
            self.historical_acceptance_rate = 1.0 if accepted else 0.0
        else:
            new_rate = ((self.historical_acceptance_rate * self.historical_trips) + 
                        (1.0 if accepted else 0.0)) / (self.historical_trips + 1)
            self.historical_acceptance_rate = new_rate
        
        self.historical_trips += 1
        
        # Notify manager of update if available
        if self._manager:
            self._manager.save_user_profile(self)
    
    def add_rating(self, rating: float) -> None:
        """
        Add a new rating given by this user.
        
        Args:
            rating: Rating value (typically 1.0 to 5.0)
        """
        self.historical_ratings.append(rating)
        
        # Notify manager of update if available
        if self._manager:
            self._manager.save_user_profile(self)
    
    def get_weights(self) -> Dict[str, float]:
        """
        Get weights for acceptance model features.
        
        Returns:
            Dict[str, float]: Feature weights
        """
        return self.weights.copy()
    
    def update_weights(self, new_weights: Dict[str, float]) -> None:
        """
        Update feature weights.
        
        Args:
            new_weights: New weight values to set
        """
        self.weights.update(new_weights)
        
        # Renormalize weights
        weight_sum = sum(self.weights.values())
        if weight_sum > 0:
            self.weights = {k: v / weight_sum for k, v in self.weights.items()}
        
        # Notify manager of update if available
        if self._manager:
            self._manager.save_user_profile(self)
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert to dictionary for serialization.
        
        Returns:
            Dict[str, Any]: Dictionary representation
        """
        return {
            "id": self.id,
            "max_walking_time_to_origin": self.max_walking_time_to_origin,
            "max_walking_time_from_destination": self.max_walking_time_from_destination,
            "max_waiting_time": self.max_waiting_time,
            "max_in_vehicle_time": self.max_in_vehicle_time,
            "max_cost": self.max_cost,
            "max_acceptable_delay": self.max_acceptable_delay,
            "service_preference": self.service_preference.value,
            "weights": self.weights,
            "historical_trips": self.historical_trips,
            "historical_acceptance_rate": self.historical_acceptance_rate,
            "historical_ratings": self.historical_ratings
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'UserProfile':
        """
        Create from dictionary representation.
        
        Args:
            data: Dictionary data
            
        Returns:
            UserProfile: New user profile instance
        """
        # Create a copy to avoid modifying the original
        data_copy = data.copy()
        
        # Parse service preference enum
        if "service_preference" in data_copy and isinstance(data_copy["service_preference"], str):
            try:
                data_copy["service_preference"] = ServicePreference(data_copy["service_preference"])
            except ValueError:
                data_copy["service_preference"] = ServicePreference.SPEED
        
        return cls(**data_copy)

class UserProfileManager:
    """
    Manager for user profiles.
    
    This class manages user profiles, including loading, saving, and accessing
    user-specific data such as weights.
    """
    
    def __init__(self, profiles_dir: str = "user_profiles"):
        """
        Initialize the user profile manager.
        
        Args:
            profiles_dir: Directory for storing user profiles
        """
        self.profiles_dir = profiles_dir
        self.profiles = {}
        
        # Create directory if it doesn't exist
        os.makedirs(profiles_dir, exist_ok=True)
        
        # Load existing profiles
        self._load_profiles()
    
    def _load_profiles(self) -> None:
        """Load profiles from the profiles directory."""
        try:
            if not os.path.exists(self.profiles_dir):
                return
            
            for filename in os.listdir(self.profiles_dir):
                if filename.endswith(".json"):
                    try:
                        with open(os.path.join(self.profiles_dir, filename), 'r') as f:
                            data = json.load(f)
                        
                        profile = UserProfile.from_dict(data)
                        profile._manager = self
                        self.profiles[profile.id] = profile
                    except Exception as e:
                        logger.error(f"Error loading profile from {filename}: {e}")
            
            logger.info(f"Loaded {len(self.profiles)} user profiles")
        except Exception as e:
            logger.error(f"Error loading user profiles: {e}")
    
    def get_user_profile(self, user_id: str) -> Optional[UserProfile]:
        """
        Get a user profile by ID.
        
        Args:
            user_id: User ID
            
        Returns:
            Optional[UserProfile]: User profile or None if not found
        """
        return self.profiles.get(user_id)
    
    def get_or_create_user_profile(self, user_id: str) -> UserProfile:
        """
        Get a user profile or create a new one if it doesn't exist.
        
        Args:
            user_id: User ID
            
        Returns:
            UserProfile: User profile
        """
        if user_id in self.profiles:
            return self.profiles[user_id]
        
        # Create a new profile
        profile = UserProfile(id=user_id)
        profile._manager = self
        self.profiles[user_id] = profile
        
        # Save the new profile
        self.save_user_profile(profile)
        
        return profile
    
    def save_user_profile(self, profile: UserProfile) -> None:
        """
        Save a user profile.
        
        Args:
            profile: User profile to save
        """
        try:
            # Create directory if it doesn't exist
            os.makedirs(self.profiles_dir, exist_ok=True)
            
            # Save profile
            file_path = os.path.join(self.profiles_dir, f"{profile.id}.json")
            with open(file_path, 'w') as f:
                json.dump(profile.to_dict(), f, indent=2)
            
            # Add to profiles dictionary
            self.profiles[profile.id] = profile
            
            logger.debug(f"Saved user profile: {profile.id}")
        except Exception as e:
            logger.error(f"Error saving user profile {profile.id}: {e}")
    
    def get_user_weights(self, user_id: str) -> Dict[str, float]:
        """
        Get weights for a specific user.
        
        Args:
            user_id: User ID
            
        Returns:
            Dict[str, float]: User-specific weights
        """
        profile = self.get_user_profile(user_id)
        if profile:
            return profile.get_weights()
        
        # Return default weights if no profile found
        return {
            "waiting_time": 0.4,
            "travel_time": 0.3,
            "cost": 0.2,
            "detour_ratio": 0.1
        }
    
    def update_user_weights(self, user_id: str, new_weights: Dict[str, float]) -> None:
        """
        Update weights for a specific user.
        
        Args:
            user_id: User ID
            new_weights: New weight values
        """
        profile = self.get_or_create_user_profile(user_id)
        profile.update_weights(new_weights)
    
    def get_all_user_ids(self) -> List[str]:
        """
        Get all user IDs.
        
        Returns:
            List[str]: List of user IDs
        """
        return list(self.profiles.keys())
    
    def delete_user_profile(self, user_id: str) -> bool:
        """
        Delete a user profile.
        
        Args:
            user_id: User ID
            
        Returns:
            bool: True if deleted, False if not found
        """
        if user_id not in self.profiles:
            return False
        
        # Remove from dictionary
        del self.profiles[user_id]
        
        # Remove file
        file_path = os.path.join(self.profiles_dir, f"{user_id}.json")
        if os.path.exists(file_path):
            os.remove(file_path)
        
        return True