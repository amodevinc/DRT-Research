"""
Example script showing how to update user weights in a simulation run.

This script demonstrates how to:
1. Set up a study and simulation context
2. Load user profiles with logit model coefficients
3. Update coefficients based on acceptance decisions
4. Reset coefficients to base values or specific simulation points
5. Compare coefficients between different simulation points

The weights in this system are logit model coefficients where:
- Negative coefficients reduce the utility (make acceptance less likely)
- Positive coefficients increase the utility (make acceptance more likely)
- The magnitude of the coefficient represents the effect size
- Coefficients are NOT normalized to sum to 1.0 as they represent direct effects
"""
import os
import json
import argparse
import logging
import random
from pathlib import Path
from datetime import datetime

# Import the necessary modules
from drt_sim.core.user.user_profile_manager import UserProfileManager
from drt_sim.config.config import UserAcceptanceConfig

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def simulate_acceptance_decisions(profile_manager, user_ids, num_iterations=5):
    """
    Simulate user acceptance decisions and update weights accordingly.
    
    Args:
        profile_manager: UserProfileManager instance
        user_ids: List of user IDs to simulate
        num_iterations: Number of iterations to simulate
    """
    logger.info(f"Simulating {num_iterations} iterations of acceptance decisions")
    
    # Weights to adjust - these are logit model coefficients
    # Negative values mean the attribute reduces utility (less likely to accept)
    # Positive values mean the attribute increases utility (more likely to accept)
    weight_keys = [
        "walking_time_to_origin", 
        "wait_time", 
        "in_vehicle_time", 
        "walking_time_from_destination"
    ]
    
    for i in range(num_iterations):
        logger.info(f"Iteration {i+1}/{num_iterations}")
        
        for user_id in user_ids:
            # Get current weights (logit coefficients)
            profile = profile_manager.get_profile(user_id)
            if not profile:
                continue
                
            # Simulate acceptance decision (random for this example)
            accepted = random.random() > 0.3  # 70% chance of acceptance
            
            # Record the trip
            profile.add_trip(
                accepted=accepted,
                rating=round(random.uniform(3.0, 5.0), 1) if accepted else None,
                reason=f"Simulation iteration {i+1}"
            )
            
            # Update weights based on acceptance
            if accepted:
                # If user accepted, make wait time less negative (or more positive)
                # This reflects that they're more tolerant of waiting
                weight_change = {
                    "wait_time": profile.weights["wait_time"] * 0.95,  # Reduces the negative effect
                    "in_vehicle_time": profile.weights["in_vehicle_time"] * 1.05  # Increases the negative effect
                }
                profile_manager.update_user_weights(
                    user_id, weight_change, 
                    reason=f"Accepted ride in iteration {i+1}"
                )
            else:
                # If user rejected, make wait time more negative (or less positive)
                # This reflects that they're less tolerant of waiting
                weight_change = {
                    "wait_time": profile.weights["wait_time"] * 1.1,  # Increases the negative effect
                    "in_vehicle_time": profile.weights["in_vehicle_time"] * 0.9  # Reduces the negative effect
                }
                profile_manager.update_user_weights(
                    user_id, weight_change, 
                    reason=f"Rejected ride in iteration {i+1}"
                )
            
            # Every few iterations, make a more significant weight adjustment for some users
            if i > 0 and i % 3 == 0 and random.random() > 0.7:
                # Randomly pick a weight to adjust significantly
                key_to_adjust = random.choice(weight_keys)
                current_val = profile.weights[key_to_adjust]
                
                # Significant adjustment (up or down)
                adjustment = random.choice([0.7, 1.3])
                weight_change = {key_to_adjust: current_val * adjustment}
                
                profile_manager.update_user_weights(
                    user_id, weight_change,
                    reason=f"Significant preference shift for {key_to_adjust}"
                )
                
                logger.info(f"Made significant adjustment to {user_id}'s {key_to_adjust} weight")

def main():
    """Main function to demonstrate weight updates in a simulation."""
    parser = argparse.ArgumentParser(description='Demonstrate weight updates in a simulation')
    parser.add_argument('--profiles', type=str, default='data/users/user_profiles',
                        help='Directory containing user profiles')
    parser.add_argument('--study', type=str, default='example_study',
                        help='Study ID for this simulation')
    parser.add_argument('--simulation', type=str, default='example_sim',
                        help='Simulation ID for this run')
    parser.add_argument('--iterations', type=int, default=10,
                        help='Number of iterations to simulate')
    
    args = parser.parse_args()
    
    # Create a simple config for the UserProfileManager
    class SimpleConfig:
        def __init__(self, profiles_dir):
            self.user_profiles_dir_path = profiles_dir
    
    config = SimpleConfig(args.profiles)
    
    # Initialize UserProfileManager
    profile_manager = UserProfileManager(config)
    
    # Set the study context
    profile_manager.set_study_context(args.study, args.simulation)
    logger.info(f"Set study context to {args.study}/{args.simulation}")
    
    # Get all user IDs
    user_ids = profile_manager.get_all_user_ids()
    logger.info(f"Found {len(user_ids)} users")
    
    if not user_ids:
        logger.error("No user profiles found. Please create user profiles first.")
        return
    
    # Simulate acceptance decisions and weight updates
    simulate_acceptance_decisions(profile_manager, user_ids, args.iterations)
    
    # Show an example of comparing weights
    if len(user_ids) > 0:
        example_user = user_ids[0]
        
        # Compare current weights with base weights
        current_weights = profile_manager.get_user_weights(example_user)
        base_weights = profile_manager.get_base_weights(example_user)
        
        logger.info(f"User {example_user} weight comparison:")
        logger.info(f"  Base weights: {base_weights}")
        logger.info(f"  Current weights: {current_weights}")
        
        # Reset a user's weights to base values
        profile_manager.reset_user_weights_to_base(example_user, 
                                                  reason="End of simulation example")
        logger.info(f"Reset {example_user}'s weights to base values")
    
    logger.info("Simulation complete!")

if __name__ == "__main__":
    main() 