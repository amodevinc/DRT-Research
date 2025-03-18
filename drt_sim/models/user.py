"""
User model for DRT simulation.

This module defines the user profile model and related data structures
for use in acceptance decision modeling, with support for tracking weight
changes over time across different simulation studies.
"""
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional
from weakref import ref, ReferenceType
from datetime import datetime
import logging
import copy
import json
import os
logger = logging.getLogger(__name__)

# Define a protocol for profile saving to avoid circular imports
class ProfileSaver:
    """Protocol for profile saving functionality."""
    
    def save_user_profile(self, profile: 'UserProfile') -> None:
        """
        Save a user profile.
        
        Args:
            profile: User profile to save
        """
        pass

@dataclass
class WeightChangeRecord:
    """Record of a weight change event."""
    timestamp: str
    weights: Dict[str, float]
    study_id: str
    simulation_id: str
    reason: str = "Weight update"

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
    max_price: float = 30.0          # currency units
    max_acceptable_delay: float = 7.0 # minutes
    
    # Feature weights for acceptance decisions
    base_weights: Dict[str, float] = field(default_factory=lambda: {
        "walking_time_to_origin": 0.4,
        "wait_time": 0.3,
        "in_vehicle_time": 0.2,
        "walking_time_from_destination": 0.1,
        "time_of_day": 0.0,
        "day_of_week": 0.0,
        "distance_to_pickup": 0.0
    })
    
    weights: Dict[str, float] = field(default_factory=dict)
    
    # Weight history
    weight_history: List[Dict[str, Any]] = field(default_factory=list)
    
    # Historical data
    historical_trips: int = 0
    historical_acceptance_rate: float = 0.0
    historical_ratings: List[float] = field(default_factory=list)
    
    # Timestamps
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    last_updated: str = field(default_factory=lambda: datetime.now().isoformat())
    
    # Manager reference (set by UserProfileManager) - using weak reference to avoid circular references
    _manager_ref: Optional[ReferenceType[ProfileSaver]] = None
    
    # Current study and simulation context
    _current_study_id: str = "default_study"
    _current_simulation_id: str = "default_simulation"
    
    def __post_init__(self):
        """Initialize after instance creation."""
        # If weights not provided, use base_weights
        if not self.weights:
            self.weights = copy.deepcopy(self.base_weights)
        
        # Ensure weights are properly normalized
        self._normalize_weights()
        
        # Initialize weight history if empty
        if not self.weight_history:
            self.weight_history = [{
                "timestamp": datetime.now().isoformat(),
                "weights": copy.deepcopy(self.weights),
                "study_id": self._current_study_id,
                "simulation_id": self._current_simulation_id,
                "reason": "Initial profile creation"
            }]
    
    def _normalize_weights(self):
        """
        Legacy method for weight normalization.
        
        No longer normalizes weights since we're using logit model coefficients
        that should preserve their exact values.
        """
        # No normalization for logit coefficients
        pass
    
    @property
    def _manager(self) -> Optional[ProfileSaver]:
        """Get the manager if it exists."""
        if self._manager_ref is not None:
            return self._manager_ref()
        return None
    
    @_manager.setter
    def _manager(self, manager: ProfileSaver) -> None:
        """Set the manager reference."""
        if manager is not None:
            self._manager_ref = ref(manager)
        else:
            self._manager_ref = None
    
    def set_manager(self, manager: ProfileSaver) -> None:
        """
        Set the manager for this profile.
        
        Args:
            manager: The manager to use for saving this profile
        """
        self._manager = manager
    
    def set_study_context(self, study_id: str, simulation_id: str) -> None:
        """
        Set the current study and simulation context.
        
        Args:
            study_id: Current study ID
            simulation_id: Current simulation ID
        """
        self._current_study_id = study_id
        self._current_simulation_id = simulation_id
    
    def get_study_context(self) -> Dict[str, str]:
        """
        Get the current study and simulation context.
        
        Returns:
            Dict containing study_id and simulation_id
        """
        return {
            "study_id": self._current_study_id,
            "simulation_id": self._current_simulation_id
        }
    
    def _notify_update(self) -> None:
        """Notify the manager of an update if available."""
        self.last_updated = datetime.now().isoformat()
        manager = self._manager
        if manager is not None:
            try:
                manager.save_user_profile(self)
            except Exception as e:
                logger.error(f"Failed to save profile {self.id}: {e}")
    
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
    
    def add_trip(self, accepted: bool, rating: Optional[float] = None, reason: str = None) -> None:
        """
        Add a trip to the user's history.
        
        Args:
            accepted: Whether the trip was accepted
            rating: Optional rating given by the user
            reason: Optional reason for the decision
        """
        # Update trip count
        self.historical_trips += 1
        
        # Update acceptance rate
        if self.historical_trips > 1:
            old_accepted_count = self.historical_acceptance_rate * (self.historical_trips - 1)
            new_accepted_count = old_accepted_count + (1 if accepted else 0)
            self.historical_acceptance_rate = new_accepted_count / self.historical_trips
        else:
            self.historical_acceptance_rate = 1.0 if accepted else 0.0
        
        # Add rating if provided
        if rating is not None:
            self.historical_ratings.append(rating)
        
        # Notify manager
        self._notify_update()
    
    def update_weights(self, new_weights: Dict[str, float], reason: str = "Weight update") -> None:
        """
        Update the user's weights.
        
        Args:
            new_weights: New weights to apply (partial update supported)
            reason: Reason for the weight update
        """
        # Create a record of the old weights before updating
        old_weights = copy.deepcopy(self.weights)
        
        # Update weights - preserving exact values as logit model coefficients
        for key, value in new_weights.items():
            self.weights[key] = value
        
        # Call normalize method (no longer performs normalization for logit coefficients)
        self._normalize_weights()
        
        # Record the weight change in history
        self.weight_history.append({
            "timestamp": datetime.now().isoformat(),
            "weights": copy.deepcopy(self.weights),
            "study_id": self._current_study_id,
            "simulation_id": self._current_simulation_id,
            "reason": reason,
            "prev_weights": old_weights
        })
        
        # Notify manager
        self._notify_update()
    
    def reset_to_base_weights(self, reason: str = "Reset to base weights") -> None:
        """
        Reset weights to the original base weights.
        
        Args:
            reason: Reason for resetting weights
        """
        old_weights = copy.deepcopy(self.weights)
        self.weights = copy.deepcopy(self.base_weights)
        
        # Record the weight change in history
        self.weight_history.append({
            "timestamp": datetime.now().isoformat(),
            "weights": copy.deepcopy(self.weights),
            "study_id": self._current_study_id,
            "simulation_id": self._current_simulation_id,
            "reason": reason,
            "prev_weights": old_weights
        })
        
        # Notify manager
        self._notify_update()
    
    def reset_to_simulation_point(self, study_id: str, simulation_id: str, reason: str = None) -> bool:
        """
        Reset weights to a specific simulation point.
        
        Args:
            study_id: Study ID to reset to
            simulation_id: Simulation ID to reset to
            reason: Reason for resetting weights
            
        Returns:
            bool: True if reset was successful, False otherwise
        """
        # Find the latest weight update for the specified study and simulation
        for entry in reversed(self.weight_history):
            if entry["study_id"] == study_id and entry["simulation_id"] == simulation_id:
                old_weights = copy.deepcopy(self.weights)
                self.weights = copy.deepcopy(entry["weights"])
                
                reset_reason = reason or f"Reset to weights from {study_id}/{simulation_id}"
                
                # Record the weight change in history
                self.weight_history.append({
                    "timestamp": datetime.now().isoformat(),
                    "weights": copy.deepcopy(self.weights),
                    "study_id": self._current_study_id,
                    "simulation_id": self._current_simulation_id,
                    "reason": reset_reason,
                    "prev_weights": old_weights,
                    "reset_to": {"study_id": study_id, "simulation_id": simulation_id}
                })
                
                # Notify manager
                self._notify_update()
                return True
        
        logger.warning(f"No weight history found for {study_id}/{simulation_id}")
        return False
    
    def get_weight_evolution(self, weight_name: str = None, study_id: str = None) -> List[Dict]:
        """
        Get the evolution of weights over time.
        
        Args:
            weight_name: Optional specific weight to track
            study_id: Optional study ID to filter by
            
        Returns:
            List of weight change records
        """
        if not self.weight_history:
            return []
        
        filtered_history = self.weight_history
        
        # Filter by study ID if specified
        if study_id:
            filtered_history = [entry for entry in filtered_history 
                               if entry["study_id"] == study_id]
        
        # Extract just the specific weight if specified
        if weight_name:
            result = []
            for entry in filtered_history:
                if weight_name in entry["weights"]:
                    result.append({
                        "timestamp": entry["timestamp"],
                        "value": entry["weights"][weight_name],
                        "study_id": entry["study_id"],
                        "simulation_id": entry["simulation_id"],
                        "reason": entry.get("reason", "Weight update")
                    })
            return result
        
        return filtered_history
    
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
            "max_price": self.max_price,
            "max_acceptable_delay": self.max_acceptable_delay,
            "base_weights": self.base_weights,
            "weights": self.weights,
            "weight_history": self.weight_history,
            "historical_trips": self.historical_trips,
            "historical_acceptance_rate": self.historical_acceptance_rate,
            "historical_ratings": self.historical_ratings,
            "created_at": self.created_at,
            "last_updated": self.last_updated
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
        
        # Remove service_preference if present (for backward compatibility)
        if "service_preference" in data_copy:
            del data_copy["service_preference"]
        
        # Handle missing base_weights by using current weights
        if "base_weights" not in data_copy and "weights" in data_copy:
            data_copy["base_weights"] = copy.deepcopy(data_copy["weights"])
        
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
                        profile.set_manager(self)
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
        profile.set_manager(self)
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