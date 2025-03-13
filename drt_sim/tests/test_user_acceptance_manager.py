"""
Unit tests for UserAcceptanceManager.

This module contains unit tests for the UserAcceptanceManager class,
verifying its functionality for managing user acceptance models and decisions.
"""
import unittest
from unittest.mock import Mock, patch, MagicMock
import datetime
import os
import tempfile
import json
from pathlib import Path

from drt_sim.models.request import Request
from drt_sim.core.user.user_profile_manager import UserProfileManager
from drt_sim.models.user import UserProfile, ServicePreference
from drt_sim.core.user.user_acceptance_manager import UserAcceptanceManager
from drt_sim.config.config import UserAcceptanceConfig


class TestUserAcceptanceManager(unittest.TestCase):
    """Test cases for UserAcceptanceManager."""

    def setUp(self):
        """Set up test fixtures."""
        # Create a temporary directory for test files
        self.temp_dir = tempfile.mkdtemp()
        
        # Mock config
        self.mock_config = {
            "model": {
                "type": "default",
                "parameters": {}
            },
            "max_history_size": 100,
            "profiles_dir": os.path.join(self.temp_dir, "profiles")
        }
        
        # Create test profiles directory
        os.makedirs(self.mock_config["profiles_dir"], exist_ok=True)
        
        # Create sample user profiles
        self.user_profiles = {
            "U1": {
                "id": "U1",
                "max_walking_time_to_origin": 3.0,
                "max_walking_time_from_destination": 3.0,
                "max_waiting_time": 10.0,
                "max_in_vehicle_time": 25.0,
                "max_cost": 30.0,
                "max_acceptable_delay": 7.0,
                "service_preference": "speed",
                "weights": {
                    "walking_time_to_origin": 0.4,
                    "wait_time": 0.3,
                    "in_vehicle_time": 0.2,
                    "walking_time_from_destination": -0.1,
                    "time_of_day": 0.0,
                    "day_of_week": 0.0,
                    "distance_to_pickup": 0.0
                },
                "historical_trips": 25,
                "historical_acceptance_rate": 0.75,
                "historical_ratings": [4.5, 3.8, 4.2, 4.0]
            },
            "U2": {
                "id": "U2",
                "max_walking_time_to_origin": 4.0,
                "max_walking_time_from_destination": 4.0,
                "max_waiting_time": 15.0,
                "max_in_vehicle_time": 30.0,
                "max_cost": 25.0,
                "max_acceptable_delay": 5.0,
                "service_preference": "reliability",
                "weights": {
                    "walking_time_to_origin": 0.1,
                    "wait_time": 0.5,
                    "in_vehicle_time": 0.3,
                    "walking_time_from_destination": 0.1,
                    "time_of_day": 0.0,
                    "day_of_week": 0.0,
                    "distance_to_pickup": 0.0
                },
                "historical_trips": 10,
                "historical_acceptance_rate": 0.8,
                "historical_ratings": [4.0, 4.5]
            }
        }
        
        # Save sample profiles to files
        for user_id, profile_data in self.user_profiles.items():
            profile_path = os.path.join(self.mock_config["profiles_dir"], f"{user_id}.json")
            with open(profile_path, 'w') as f:
                json.dump(profile_data, f)
        
        # Create a mock request
        self.mock_request = Mock(spec=Request)
        self.mock_request.id = "request-123"
        self.mock_request.user_id = "U1"
        self.mock_request.origin = (40.7128, -74.0060)  # NYC
        self.mock_request.destination = (40.7484, -73.9857)  # Empire State
        self.mock_request.request_time = datetime.datetime.now()
        
        # Initialize profile manager
        with patch('drt_sim.models.user.UserProfile.from_dict', side_effect=self._mock_from_dict):
            self.profile_manager = UserProfileManager(UserAcceptanceConfig(self.mock_config))
        
        # Initialize user acceptance manager with a mock model
        with patch('drt_sim.core.user.user_acceptance_manager.importlib.import_module') as mock_import:
            mock_model_class = MagicMock()
            mock_model = MagicMock()
            mock_model.calculate_acceptance_probability.return_value = 0.85
            mock_model.decide_acceptance.return_value = (True, 0.85)
            mock_model_class.return_value = mock_model
            mock_import.return_value = MagicMock()
            mock_import.return_value.__name__ = "mock_module"
            setattr(mock_import.return_value, "DefaultModel", mock_model_class)
            
            self.manager = UserAcceptanceManager(UserAcceptanceConfig(self.mock_config), self.profile_manager)
            self.mock_model = mock_model

    def _mock_from_dict(self, data):
        """Create a mock UserProfile from dictionary."""
        profile = MagicMock(spec=UserProfile)
        for key, value in data.items():
            setattr(profile, key, value)
        
        # Handle service preference
        if 'service_preference' in data:
            if data['service_preference'] == 'speed':
                profile.service_preference = ServicePreference.SPEED
            else:
                profile.service_preference = ServicePreference.RELIABILITY
        
        # Setup methods
        profile.get_weights.return_value = data.get('weights', {})
        profile.to_dict.return_value = data
        
        return profile

    def tearDown(self):
        """Clean up after tests."""
        # Clean up temporary directory
        for root, dirs, files in os.walk(self.temp_dir, topdown=False):
            for file in files:
                os.remove(os.path.join(root, file))
            for dir in dirs:
                os.rmdir(os.path.join(root, dir))
        os.rmdir(self.temp_dir)

    def test_initialize_model(self):
        """Test model initialization."""
        self.assertIsNotNone(self.manager.model)
        
    def test_calculate_acceptance_probability(self):
        """Test calculating acceptance probability."""
        # Setup
        now = datetime.datetime.now()
        proposed_pickup_time = now + datetime.timedelta(minutes=5)
        proposed_travel_time = datetime.timedelta(minutes=15)
        cost = 25.0
        
        # Execute
        probability = self.manager.calculate_acceptance_probability(
            self.mock_request,
            proposed_pickup_time,
            proposed_travel_time,
            cost
        )
        
        # Verify
        self.assertEqual(probability, 0.85)
        self.mock_model.calculate_acceptance_probability.assert_called_once()
        
    def test_decide_acceptance(self):
        """Test deciding acceptance."""
        # Setup
        now = datetime.datetime.now()
        proposed_pickup_time = now + datetime.timedelta(minutes=5)
        proposed_travel_time = datetime.timedelta(minutes=15)
        cost = 25.0
        
        # Execute
        accepted, probability = self.manager.decide_acceptance(
            self.mock_request,
            proposed_pickup_time,
            proposed_travel_time,
            cost
        )
        
        # Verify
        self.assertTrue(accepted)
        self.assertEqual(probability, 0.85)
        self.mock_model.decide_acceptance.assert_called_once()
        self.assertEqual(len(self.manager.acceptance_history), 1)
        
    def test_update_model(self):
        """Test updating the model based on user decisions."""
        # Setup
        service_attributes = {
            "waiting_time": 5.0,
            "travel_time": 15.0,
            "cost": 25.0
        }
        
        # Mock user profile manager to track calls
        profile = self.profile_manager.get_profile("U1")
        
        # Execute
        self.manager.update_model(
            self.mock_request,
            True,
            service_attributes
        )
        
        # Verify
        self.mock_model.update_model.assert_called_once()
        self.assertEqual(self.manager.acceptance_metrics["total_requests"], 1)
        self.assertEqual(self.manager.acceptance_metrics["accepted_requests"], 1)
        
    def test_batch_update(self):
        """Test batch updating the model."""
        # Setup
        training_data = [
            {"features": [5.0, 15.0, 25.0], "outcome": True},
            {"features": [10.0, 20.0, 30.0], "outcome": False}
        ]
        
        # Execute
        self.manager.batch_update(training_data)
        
        # Verify
        self.mock_model.batch_update.assert_called_once_with(training_data)
        
    def test_save_model(self):
        """Test saving the model."""
        # Setup
        save_path = os.path.join(self.temp_dir, "model.pkl")
        
        # Execute
        self.manager.save_model(save_path)
        
        # Verify
        self.mock_model.save_model.assert_called_once_with(save_path)
        
    def test_get_feature_importance(self):
        """Test getting feature importance."""
        # Setup
        expected_importance = {
            "walking_time_to_origin": 0.2,
            "wait_time": 0.3,
            "in_vehicle_time": 0.4,
            "walking_time_from_destination": 0.1
        }
        self.mock_model.get_feature_importance.return_value = expected_importance
        
        # Execute
        importance = self.manager.get_feature_importance()
        
        # Verify
        self.assertEqual(importance, expected_importance)
        self.mock_model.get_feature_importance.assert_called_once()
        
    def test_get_metrics(self):
        """Test getting acceptance metrics."""
        # Setup
        now = datetime.datetime.now()
        proposed_pickup_time = now + datetime.timedelta(minutes=5)
        proposed_travel_time = datetime.timedelta(minutes=15)
        cost = 25.0
        service_attributes = {
            "waiting_time": 5.0,
            "request_time": now,
            "proposed_pickup_time": proposed_pickup_time
        }
        
        # Make a few decisions and update the model with outcomes
        accepted, _ = self.manager.decide_acceptance(
            self.mock_request,
            proposed_pickup_time,
            proposed_travel_time,
            cost
        )
        self.manager.update_model(self.mock_request, accepted, service_attributes)
        
        # Change request user and make another decision
        self.mock_request.user_id = "U2"
        accepted, _ = self.manager.decide_acceptance(
            self.mock_request,
            proposed_pickup_time,
            proposed_travel_time,
            cost
        )
        self.manager.update_model(self.mock_request, accepted, service_attributes)
        
        # Execute
        metrics = self.manager.get_metrics()
        
        # Verify
        self.assertEqual(metrics["total_requests"], 2)
        self.assertEqual(metrics["accepted_requests"], 2)
        self.assertEqual(metrics["acceptance_rate"], 1.0)
        self.assertIn("speed", metrics["by_user_type"])
        self.assertIn("reliability", metrics["by_user_type"])
        
    def test_handle_error_in_decide_acceptance(self):
        """Test error handling in decide_acceptance."""
        # Setup
        now = datetime.datetime.now()
        proposed_pickup_time = now + datetime.timedelta(minutes=5)
        proposed_travel_time = datetime.timedelta(minutes=15)
        cost = 25.0
        
        # Make the model raise an exception
        self.mock_model.decide_acceptance.side_effect = Exception("Test error")
        
        # Execute
        accepted, probability = self.manager.decide_acceptance(
            self.mock_request,
            proposed_pickup_time,
            proposed_travel_time,
            cost
        )
        
        # Verify - should default to acceptance
        self.assertTrue(accepted)
        self.assertEqual(probability, 0.9)
        
    def test_history_size_limit(self):
        """Test that history size is limited."""
        # Setup
        now = datetime.datetime.now()
        proposed_pickup_time = now + datetime.timedelta(minutes=5)
        proposed_travel_time = datetime.timedelta(minutes=15)
        cost = 25.0
        
        # Set a small history size
        self.manager.config.max_history_size = 3
        
        # Make more decisions than the history size
        for i in range(5):
            self.manager.decide_acceptance(
                self.mock_request,
                proposed_pickup_time,
                proposed_travel_time,
                cost
            )
        
        # Verify
        self.assertEqual(len(self.manager.acceptance_history), 3)
        
    def test_update_metrics_by_waiting_time(self):
        """Test updating metrics by waiting time."""
        # Setup
        now = datetime.datetime.now()
        service_attributes = {
            "waiting_time": 12.0,  # Should fall in 10-15 min category
            "request_time": now,
            "proposed_pickup_time": now + datetime.timedelta(minutes=12)
        }
        
        # Execute
        self.manager._update_metrics(
            self.mock_request,
            True,
            service_attributes
        )
        
        # Verify
        self.assertIn("10-15 min", self.manager.acceptance_metrics["by_waiting_time"])
        self.assertEqual(self.manager.acceptance_metrics["by_waiting_time"]["10-15 min"]["total"], 1)
        self.assertEqual(self.manager.acceptance_metrics["by_waiting_time"]["10-15 min"]["accepted"], 1)


if __name__ == '__main__':
    unittest.main()