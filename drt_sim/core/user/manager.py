# drt_sim/core/user/manager.py
from dataclasses import dataclass, field
from typing import Dict, Optional, List
from datetime import datetime

from drt_sim.models.user import UserProfile, WeightAdjuster
from drt_sim.models.location import Location

@dataclass
class UserProfileManager:
    """Manages user profiles and their weight adjustments"""
    weight_adjuster: WeightAdjuster = field(default_factory=WeightAdjuster)
    profiles: Dict[str, UserProfile] = field(default_factory=dict)
    
    # Track profile updates and analytics
    last_updated: Dict[str, datetime] = field(default_factory=dict)
    profile_analytics: Dict[str, Dict] = field(default_factory=dict)
    
    def get_adjusted_weights(
        self,
        user_id: str,
        base_weights: Dict[str, float],
        request_context: Optional[Dict] = None
    ) -> Dict[str, float]:
        """Get adjusted weights for a specific user"""
        profile = self.get_profile(user_id)
        
        return self.weight_adjuster.adjust_weights(
            base_weights,
            profile,
            request_context
        )
    
    def update_profile(self, profile: UserProfile) -> None:
        """Update or create a user profile with timestamp tracking"""
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
            "timestamp": datetime.now(),
            "service_preference": profile.service_preference.value,
            "priority_level": profile.priority_level.value
        })
    
    def get_profile(self, user_id: str) -> UserProfile:
        """Retrieve a user profile, creating a default one if none exists"""
        if user_id not in self.profiles:
            self.profiles[user_id] = UserProfile(id=user_id)
            self.last_updated[user_id] = datetime.now()
        return self.profiles[user_id]
    
    def add_frequent_location(self, user_id: str, label: str, location: Location) -> None:
        """Add or update a frequent location for a user"""
        profile = self.get_profile(user_id)
        profile.frequent_locations[label] = location
        self.update_profile(profile)
    
    def remove_frequent_location(self, user_id: str, label: str) -> bool:
        """Remove a frequent location for a user"""
        profile = self.get_profile(user_id)
        if label in profile.frequent_locations:
            del profile.frequent_locations[label]
            self.update_profile(profile)
            return True
        return False
    
    def get_inactive_users(self, days: int = 30) -> List[str]:
        """Get list of user IDs who haven't been updated in specified days"""
        threshold = datetime.now().timestamp() - (days * 24 * 60 * 60)
        return [
            user_id for user_id, last_update in self.last_updated.items()
            if last_update.timestamp() < threshold
        ]
    
    def get_user_analytics(self, user_id: str) -> Optional[Dict]:
        """Get analytics for a specific user"""
        return self.profile_analytics.get(user_id)
    
    def bulk_update_profiles(self, profiles: List[UserProfile]) -> None:
        """Bulk update multiple user profiles"""
        for profile in profiles:
            self.update_profile(profile)
    
    def get_users_by_preference(self, service_preference) -> List[str]:
        """Get all users with a specific service preference"""
        return [
            user_id for user_id, profile in self.profiles.items()
            if profile.service_preference == service_preference
        ]
