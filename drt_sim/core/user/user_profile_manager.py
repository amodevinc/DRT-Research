"""
User profile manager for DRT simulation.

This module provides functionality to manage user profiles, including
loading and saving profiles, and managing user-specific weights.
"""
from typing import Dict, Any, Optional, List
from datetime import datetime
import os
import json
import logging
import pandas as pd

from drt_sim.models.user import UserProfile, ServicePreference
from drt_sim.config.config import UserAcceptanceConfig
logger = logging.getLogger(__name__)

class UserProfileManager:
    """
    Manager for user profiles.
    
    This class manages user profiles, including loading, saving, and accessing
    user-specific data such as weights.
    """
    
    def __init__(
        self, 
        cfg: UserAcceptanceConfig
    ):
        """
        Initialize the user profile manager.
        
        Args:
            cfg: User acceptance configuration
        """
        self.profiles_dir = cfg.user_profiles_dir_path
        self.profiles = {}
        self.last_updated = {}
        self.profile_analytics = {}
        
        # Create directory if it doesn't exist
        os.makedirs(self.profiles_dir, exist_ok=True)
        
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
                        file_path = os.path.join(self.profiles_dir, filename)
                        with open(file_path, 'r') as f:
                            data = json.load(f)
                        
                        profile = UserProfile.from_dict(data)
                        profile._manager = self
                        self.profiles[profile.id] = profile
                        self.last_updated[profile.id] = datetime.now()
                    except Exception as e:
                        logger.error(f"Error loading profile from {filename}: {e}")
            
            logger.info(f"Loaded {len(self.profiles)} user profiles")
        except Exception as e:
            logger.error(f"Error loading user profiles: {e}")
    
    def get_profile(self, user_id: str) -> Optional[UserProfile]:
        """
        Get a user profile by ID.
        
        Args:
            user_id: User ID
            
        Returns:
            Optional[UserProfile]: User profile or None if not found
        """
        return self.profiles.get(user_id)
    
    def get_or_create_profile(self, user_id: str) -> UserProfile:
        """
        Get a user profile or create a new one if it doesn't exist.
        
        Args:
            user_id: User ID
            
        Returns:
            UserProfile: User profile
        """
        if user_id in self.profiles:
            return self.profiles[user_id]
        
        # Create a new profile with default values
        profile = UserProfile(id=user_id)
        profile._manager = self
        
        # Save the new profile
        self.update_profile(profile)
        
        return profile
    
    def save_profile(self, profile: UserProfile) -> None:
        """
        Save a user profile.
        
        Args:
            profile: User profile to save
        """
        try:
            # Create directory if it doesn't exist
            os.makedirs(self.profiles_dir, exist_ok=True)
            
            # Save profile to JSON
            file_path = os.path.join(self.profiles_dir, f"{profile.id}.json")
            with open(file_path, 'w') as f:
                json.dump(profile.to_dict(), f, indent=2)
            
            # Store in memory
            self.profiles[profile.id] = profile
            self.last_updated[profile.id] = datetime.now()
            
            # Update analytics
            if profile.id not in self.profile_analytics:
                self.profile_analytics[profile.id] = {
                    "created_at": datetime.now(),
                    "update_count": 0,
                    "preference_history": []
                }
            
            self.profile_analytics[profile.id]["update_count"] += 1
            self.profile_analytics[profile.id]["preference_history"].append({
                "timestamp": datetime.now().isoformat(),
                "service_preference": profile.service_preference.value if hasattr(profile.service_preference, 'value') else str(profile.service_preference),
                "trips": profile.historical_trips,
                "acceptance_rate": profile.historical_acceptance_rate
            })
            
            logger.debug(f"Saved user profile: {profile.id}")
        except Exception as e:
            logger.error(f"Error saving user profile {profile.id}: {e}")
    
    def update_profile(self, profile: UserProfile) -> None:
        """
        Update a user profile.
        
        Args:
            profile: User profile to update
        """
        profile._manager = self
        self.save_profile(profile)
    
    def get_user_weights(self, user_id: str) -> Dict[str, float]:
        """
        Get weights for a specific user.
        
        Args:
            user_id: User ID
            
        Returns:
            Dict[str, float]: User-specific weights
        """
        profile = self.get_profile(user_id)
        if profile:
            return profile.get_weights()
        
        # Return default weights if profile not found
        return {
            "walking_time_to_origin": 0.4,
            "wait_time": 0.3,
            "in_vehicle_time": 0.2,
            "walking_time_from_destination": 0.1,
            "time_of_day": 0.0,
            "day_of_week": 0.0,
            "distance_to_pickup": 0.0
        }
    
    def update_user_weights(self, user_id: str, new_weights: Dict[str, float]) -> None:
        """
        Update weights for a specific user.
        
        Args:
            user_id: User ID
            new_weights: New weight values
        """
        # Get or create profile and update weights
        profile = self.get_or_create_profile(user_id)
        profile.update_weights(new_weights)
    
    def get_all_user_ids(self) -> List[str]:
        """
        Get all user IDs.
        
        Returns:
            List[str]: List of user IDs
        """
        return list(self.profiles.keys())
    
    def delete_profile(self, user_id: str) -> bool:
        """
        Delete a user profile.
        
        Args:
            user_id: User ID
            
        Returns:
            bool: True if deleted, False if not found
        """
        if user_id not in self.profiles:
            return False
        
        # Remove from memory
        del self.profiles[user_id]
        if user_id in self.last_updated:
            del self.last_updated[user_id]
        if user_id in self.profile_analytics:
            del self.profile_analytics[user_id]
        
        # Remove file
        file_path = os.path.join(self.profiles_dir, f"{user_id}.json")
        if os.path.exists(file_path):
            os.remove(file_path)
        
        return True
    
    def get_inactive_users(self, days: int = 30) -> List[str]:
        """
        Get list of user IDs who haven't been updated in specified days.
        
        Args:
            days: Number of days of inactivity
            
        Returns:
            List[str]: List of inactive user IDs
        """
        threshold = datetime.now().timestamp() - (days * 24 * 60 * 60)
        return [
            user_id for user_id, last_update in self.last_updated.items()
            if last_update.timestamp() < threshold
        ]
    
    def get_user_analytics(self, user_id: str) -> Optional[Dict]:
        """
        Get analytics for a specific user.
        
        Args:
            user_id: User ID
            
        Returns:
            Optional[Dict]: User analytics or None if not found
        """
        return self.profile_analytics.get(user_id)
    
    def get_users_by_preference(self, service_preference: ServicePreference) -> List[str]:
        """
        Get all users with a specific service preference.
        
        Args:
            service_preference: Service preference to match
            
        Returns:
            List[str]: List of matching user IDs
        """
        return [
            user_id for user_id, profile in self.profiles.items()
            if profile.service_preference == service_preference
        ]
    
    def bulk_update_profiles(self, profiles: List[UserProfile]) -> None:
        """
        Bulk update multiple user profiles.
        
        Args:
            profiles: List of profiles to update
        """
        for profile in profiles:
            self.update_profile(profile)
    
    def export_profiles_to_csv(self, file_path: str) -> bool:
        """
        Export all profiles to a CSV file.
        
        Args:
            file_path: Path to export the CSV file
            
        Returns:
            bool: True if export was successful, False otherwise
        """
        try:
            data = []
            for user_id, profile in self.profiles.items():
                row = profile.to_dict()
                # Flatten nested structures
                if 'weights' in row and isinstance(row['weights'], dict):
                    for key, value in row['weights'].items():
                        row[f"weight_{key}"] = value
                    del row['weights']
                
                data.append(row)
            
            if not data:
                logger.warning("No profiles to export")
                return False
            
            df = pd.DataFrame(data)
            df.to_csv(file_path, index=False)
            
            logger.info(f"Exported {len(data)} profiles to {file_path}")
            return True
        
        except Exception as e:
            logger.error(f"Error exporting profiles to CSV: {e}")
            return False
    
    def import_profiles_from_csv(self, file_path: str) -> int:
        """
        Import profiles from a CSV file.
        
        Args:
            file_path: Path to the CSV file
            
        Returns:
            int: Number of profiles imported
        """
        try:
            if not os.path.exists(file_path):
                logger.error(f"File not found: {file_path}")
                return 0
            
            df = pd.read_csv(file_path)
            
            if 'id' not in df.columns:
                logger.error("CSV file must contain an 'id' column")
                return 0
            
            count = 0
            for _, row in df.iterrows():
                try:
                    # Extract basic profile data
                    profile_data = {col: row[col] for col in df.columns if not col.startswith('weight_') and pd.notna(row[col])}
                    
                    # Extract weights
                    weights = {}
                    for col in df.columns:
                        if col.startswith('weight_') and pd.notna(row[col]):
                            weight_name = col[7:]  # Remove 'weight_' prefix
                            weights[weight_name] = row[col]
                    
                    # Add weights to profile data
                    if weights:
                        profile_data['weights'] = weights
                    
                    # Create and save profile
                    profile = UserProfile.from_dict(profile_data)
                    self.update_profile(profile)
                    count += 1
                
                except Exception as e:
                    logger.error(f"Error importing profile from row {row['id']}: {e}")
            
            logger.info(f"Imported {count} profiles from {file_path}")
            return count
        
        except Exception as e:
            logger.error(f"Error importing profiles from CSV: {e}")
            return 0