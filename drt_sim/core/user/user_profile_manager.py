"""
User profile manager for DRT simulation.

This module provides functionality to manage user profiles, including
loading and saving profiles, and managing user-specific weights with
support for tracking changes over time across different simulation studies.
"""
from typing import Dict, Any, Optional, List, Tuple
from datetime import datetime
import os
import json
import logging
import pandas as pd
import copy

from drt_sim.models.user import UserProfile
from drt_sim.config.config import UserAcceptanceConfig
logger = logging.getLogger(__name__)

class UserProfileManager:
    """
    Manager for user profiles.
    
    This class manages user profiles, including loading, saving, and accessing
    user-specific data such as weights, with support for tracking changes over time.
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
        self.profiles: Dict[str, UserProfile] = {}
        self.last_updated = {}
        self.profile_analytics = {}
        
        # Directory for change logs
        self.change_logs_dir = os.path.join(os.path.dirname(self.profiles_dir), 'change_logs')
        os.makedirs(self.change_logs_dir, exist_ok=True)
        
        # Set default study and simulation context
        self.current_study_id = "default_study"
        self.current_simulation_id = "default_simulation"
        
        # Create directory if it doesn't exist
        os.makedirs(self.profiles_dir, exist_ok=True)
        
        # Load existing profiles
        self._load_profiles()
    
    def set_study_context(self, study_id: str, simulation_id: str) -> None:
        """
        Set the current study and simulation context.
        
        Args:
            study_id: Current study ID
            simulation_id: Current simulation ID
        """
        self.current_study_id = study_id
        self.current_simulation_id = simulation_id
        
        # Create directories for this study/simulation if they don't exist
        study_dir = os.path.join(self.change_logs_dir, study_id)
        os.makedirs(study_dir, exist_ok=True)
        
        sim_dir = os.path.join(study_dir, simulation_id)
        os.makedirs(sim_dir, exist_ok=True)
        
        # Update context for all profiles
        for profile in self.profiles.values():
            profile.set_study_context(study_id, simulation_id)
    
    def get_study_context(self) -> Dict[str, str]:
        """
        Get the current study and simulation context.
        
        Returns:
            Dict containing study_id and simulation_id
        """
        return {
            "study_id": self.current_study_id,
            "simulation_id": self.current_simulation_id
        }
    
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
                        profile.set_study_context(self.current_study_id, self.current_simulation_id)
                        self.profiles[profile.id] = profile
                        self.last_updated[profile.id] = datetime.now()
                    except Exception as e:
                        logger.error(f"Error loading profile from {filename}: {e}")
            
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
        profile.set_study_context(self.current_study_id, self.current_simulation_id)
        
        # Save the new profile
        self.update_profile(profile)
        
        # Log creation
        self._log_profile_change(user_id, "creation", "Profile created", profile.weights)
        
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
                    "weight_update_history": []
                }
            
            self.profile_analytics[profile.id]["update_count"] += 1
            
            logger.debug(f"Saved profile: {profile.id}")
        except Exception as e:
            logger.error(f"Error saving profile {profile.id}: {e}")
    
    def update_profile(self, profile: UserProfile) -> None:
        """
        Update a user profile.
        
        Args:
            profile: User profile to update
        """
        old_profile = self.get_profile(profile.id)
        if old_profile and old_profile.weights != profile.weights:
            self._log_profile_change(profile.id, "weight_update", "Weights updated", profile.weights)
        
        self.save_profile(profile)
    
    def get_user_weights(self, user_id: str) -> Dict[str, float]:
        """
        Get weights for a specific user.
        
        Args:
            user_id: User ID
            
        Returns:
            Dict[str, float]: User-specific feature weights or empty dict if user not found
        """
        profile = self.get_profile(user_id)
        if not profile:
            logger.warning(f"No profile found for user {user_id}")
            return {}
        
        return copy.deepcopy(profile.weights)
    
    def get_base_weights(self, user_id: str) -> Dict[str, float]:
        """
        Get original base weights for a specific user.
        
        Args:
            user_id: User ID
            
        Returns:
            Dict[str, float]: Original base weights or empty dict if user not found
        """
        profile = self.get_profile(user_id)
        if not profile:
            logger.warning(f"No profile found for user {user_id}")
            return {}
        
        return copy.deepcopy(profile.base_weights)
    
    def update_user_weights(self, user_id: str, new_weights: Dict[str, float], reason: str = None) -> None:
        """
        Update weights for a specific user.
        
        Args:
            user_id: User ID
            new_weights: New feature weights
            reason: Optional reason for the weight update
        """
        profile = self.get_profile(user_id)
        if profile:
            update_reason = reason or f"Weight update in {self.current_study_id}/{self.current_simulation_id}"
            profile.update_weights(new_weights, update_reason)
            
            # Log the weight change specifically
            self._log_profile_change(
                user_id, 
                "weight_update", 
                update_reason, 
                new_weights
            )
        else:
            logger.warning(f"Cannot update weights for non-existent user {user_id}")
    
    def reset_user_weights_to_base(self, user_id: str, reason: str = None) -> bool:
        """
        Reset user weights to their original base values.
        
        Args:
            user_id: User ID
            reason: Optional reason for the reset
            
        Returns:
            bool: True if reset was successful, False otherwise
        """
        profile = self.get_profile(user_id)
        if not profile:
            logger.warning(f"No profile found for user {user_id}")
            return False
        
        reset_reason = reason or f"Reset to base weights in {self.current_study_id}/{self.current_simulation_id}"
        profile.reset_to_base_weights(reset_reason)
        
        # Log the reset
        self._log_profile_change(
            user_id,
            "weight_reset",
            reset_reason,
            profile.weights
        )
        
        return True
    
    def reset_user_weights_to_simulation(self, user_id: str, study_id: str, simulation_id: str, reason: str = None) -> bool:
        """
        Reset user weights to a specific simulation point.
        
        Args:
            user_id: User ID
            study_id: Study ID to reset to
            simulation_id: Simulation ID to reset to
            reason: Optional reason for the reset
            
        Returns:
            bool: True if reset was successful, False otherwise
        """
        profile = self.get_profile(user_id)
        if not profile:
            logger.warning(f"No profile found for user {user_id}")
            return False
        
        reset_reason = reason or f"Reset to weights from {study_id}/{simulation_id}"
        result = profile.reset_to_simulation_point(study_id, simulation_id, reset_reason)
        
        if result:
            # Log the reset
            self._log_profile_change(
                user_id,
                "weight_reset_to_simulation",
                reset_reason,
                profile.weights
            )
        
        return result
    
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
        
        # Log deletion
        self._log_profile_change(user_id, "deletion", "Profile deleted")
        
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
    
    def get_weight_change_history(self, user_id: str, weight_name: str = None, study_id: str = None) -> List[Dict]:
        """
        Get history of weight changes for a specific user.
        
        Args:
            user_id: User ID
            weight_name: Optional specific weight to track
            study_id: Optional study ID to filter by
            
        Returns:
            List[Dict]: List of weight change events
        """
        profile = self.get_profile(user_id)
        if not profile:
            logger.warning(f"No profile found for user {user_id}")
            return []
        
        return profile.get_weight_evolution(weight_name, study_id)
    
    def bulk_update_profiles(self, profiles: List[UserProfile]) -> None:
        """
        Bulk update multiple user profiles.
        
        Args:
            profiles: List of profiles to update
        """
        for profile in profiles:
            self.update_profile(profile)
    
    def bulk_reset_to_base(self, user_ids: List[str] = None, reason: str = None) -> int:
        """
        Reset weights to base values for multiple users.
        
        Args:
            user_ids: List of user IDs to reset (None for all users)
            reason: Optional reason for the reset
            
        Returns:
            int: Number of profiles successfully reset
        """
        if user_ids is None:
            user_ids = self.get_all_user_ids()
        
        count = 0
        for user_id in user_ids:
            if self.reset_user_weights_to_base(user_id, reason):
                count += 1
        
        return count
    
    def compare_weights(self, user_id: str, study1: str, sim1: str, study2: str, sim2: str) -> Dict:
        """
        Compare weights between two simulation points.
        
        Args:
            user_id: User ID
            study1: First study ID
            sim1: First simulation ID
            study2: Second study ID
            sim2: Second simulation ID
            
        Returns:
            Dict: Comparison result with differences
        """
        profile = self.get_profile(user_id)
        if not profile:
            logger.warning(f"No profile found for user {user_id}")
            return {}
        
        # Find weights from the two points
        history = profile.weight_history
        weights1 = None
        weights2 = None
        
        for entry in history:
            if entry["study_id"] == study1 and entry["simulation_id"] == sim1:
                weights1 = entry["weights"]
            if entry["study_id"] == study2 and entry["simulation_id"] == sim2:
                weights2 = entry["weights"]
        
        if not weights1 or not weights2:
            return {"error": "Could not find weights for one or both simulation points"}
        
        # Calculate differences
        diff = {}
        all_keys = set(weights1.keys()) | set(weights2.keys())
        
        for key in all_keys:
            val1 = weights1.get(key, 0)
            val2 = weights2.get(key, 0)
            diff[key] = {
                "from": val1,
                "to": val2,
                "change": val2 - val1,
                "percent_change": (val2 - val1) / val1 * 100 if val1 != 0 else float('inf')
            }
        
        return {
            "point1": f"{study1}/{sim1}",
            "point2": f"{study2}/{sim2}",
            "differences": diff
        }
    
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
                
                if 'base_weights' in row and isinstance(row['base_weights'], dict):
                    for key, value in row['base_weights'].items():
                        row[f"base_weight_{key}"] = value
                    del row['base_weights']
                
                # Remove complex structures that don't fit well in CSV
                if 'weight_history' in row:
                    row['weight_history_count'] = len(row['weight_history'])
                    del row['weight_history']
                
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
            df = pd.read_csv(file_path)
            count = 0
            
            for _, row in df.iterrows():
                data = row.to_dict()
                user_id = data.get('id')
                
                if not user_id:
                    logger.warning(f"Skipping row without id: {data}")
                    continue
                
                # Reconstruct weights dictionary
                weights = {}
                base_weights = {}
                
                for key, value in list(data.items()):
                    if key.startswith('weight_'):
                        weight_name = key[7:]  # Remove 'weight_' prefix
                        weights[weight_name] = value
                        del data[key]
                    elif key.startswith('base_weight_'):
                        weight_name = key[12:]  # Remove 'base_weight_' prefix
                        base_weights[weight_name] = value
                        del data[key]
                
                if weights:
                    data['weights'] = weights
                if base_weights:
                    data['base_weights'] = base_weights
                
                # Create or update profile
                try:
                    profile = UserProfile.from_dict(data)
                    profile.set_study_context(self.current_study_id, self.current_simulation_id)
                    self.update_profile(profile)
                    count += 1
                except Exception as e:
                    logger.error(f"Error importing profile {user_id}: {e}")
            
            logger.info(f"Imported {count} profiles from {file_path}")
            return count
        
        except Exception as e:
            logger.error(f"Error importing profiles from CSV: {e}")
            return 0
    
    def export_weight_history(self, user_id: str, file_path: str) -> bool:
        """
        Export weight history for a specific user to a JSON file.
        
        Args:
            user_id: User ID
            file_path: Path to export the JSON file
            
        Returns:
            bool: True if export was successful, False otherwise
        """
        profile = self.get_profile(user_id)
        if not profile:
            logger.warning(f"No profile found for user {user_id}")
            return False
        
        try:
            weight_history = profile.weight_history
            with open(file_path, 'w') as f:
                json.dump(weight_history, f, indent=2)
            
            logger.info(f"Exported weight history for {user_id} to {file_path}")
            return True
        except Exception as e:
            logger.error(f"Error exporting weight history: {e}")
            return False
    
    def _log_profile_change(self, user_id: str, change_type: str, description: str, weights=None) -> None:
        """
        Log changes to a user profile.
        
        Args:
            user_id: ID of the user whose profile was changed
            change_type: Type of change (e.g., 'weight_update', 'preference_change')
            description: Description of the change
            weights: New weights if applicable
        """
        # Create study and simulation directories if they don't exist
        study_dir = os.path.join(self.change_logs_dir, self.current_study_id)
        os.makedirs(study_dir, exist_ok=True)
        
        sim_dir = os.path.join(study_dir, self.current_simulation_id)
        os.makedirs(sim_dir, exist_ok=True)
        
        change_log_path = os.path.join(sim_dir, f"{user_id}_changes.json")
        
        # Load existing log if it exists
        if os.path.exists(change_log_path):
            try:
                with open(change_log_path, 'r') as f:
                    change_log = json.load(f)
            except Exception as e:
                logger.error(f"Error loading change log for {user_id}: {e}")
                change_log = []
        else:
            change_log = []
        
        # Create new log entry
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "type": change_type,
            "study_id": self.current_study_id,
            "simulation_id": self.current_simulation_id,
            "description": description
        }
        
        # Add weights if provided
        if weights:
            log_entry["weights"] = copy.deepcopy(weights)
        
        # Add to log and save
        change_log.append(log_entry)
        try:
            with open(change_log_path, 'w') as f:
                json.dump(change_log, f, indent=2)
            logger.debug(f"Updated change log for {user_id}")
        except Exception as e:
            logger.error(f"Error saving change log for {user_id}: {e}")