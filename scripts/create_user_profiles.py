"""
Script to create user profiles based on acceptance weights data.

This script reads the acceptance weights from a CSV file and creates
user profiles for each user, with reasonable default values and 
proper weight mappings.
"""
import os
import pandas as pd
import json
import random
import logging
import argparse
from pathlib import Path
from enum import Enum
import traceback
# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Define service preferences - simplified to what we can reasonably infer
class ServicePreference(Enum):
    """Service preferences that influence acceptance weights"""
    SPEED = "speed"                # Prefers faster service (in-vehicle time)
    RELIABILITY = "reliability"    # Prefers reliable arrival times (wait time)

def create_user_profiles(weights_file, output_dir, profile_folder='user_profiles'):
    """
    Create user profiles based on the acceptance weights file.
    
    Args:
        weights_file: Path to the acceptance weights CSV file
        output_dir: Base directory to save profiles
        profile_folder: Subfolder to store user profiles
    """
    # Read the weights data
    try:
        weights_df = pd.read_csv(weights_file)
        logger.info(f"Loaded weights for {len(weights_df)} users from {weights_file}")
    except Exception as e:
        logger.error(f"Error loading weights file: {e}")
        return

    # Create output directory
    profiles_dir = os.path.join(output_dir, profile_folder)
    os.makedirs(profiles_dir, exist_ok=True)
    
    # Create a directory for any summary data
    stats_dir = os.path.join(output_dir, 'stats')
    os.makedirs(stats_dir, exist_ok=True)
    
    # Summary statistics
    profile_stats = {
        'total_profiles': len(weights_df),
        'service_preferences': {},
        'avg_historical_trips': 0,
        'avg_max_walking_time_to_origin': 0,
        'avg_max_walking_time_from_destination': 0,
        'avg_max_waiting_time': 0,
        'avg_max_in_vehicle_time': 0,
    }
    
    # Process each user
    for _, row in weights_df.iterrows():
        user_id = row['id']
        
        # Map CSV weights to feature name
        walking_time_to_origin_weight = row['access']
        wait_time_weight = row['wait']
        in_vehicle_time_weight = row['ivt']
        walking_time_from_destination_weight = row['egress']
        
        # Normalize weights to sum to 1.0
        total_weight = walking_time_to_origin_weight + wait_time_weight + in_vehicle_time_weight + walking_time_from_destination_weight
        if total_weight > 0:
            walking_time_to_origin_weight /= total_weight
            wait_time_weight /= total_weight
            in_vehicle_time_weight /= total_weight
            walking_time_from_destination_weight /= total_weight
        else:
            # Default weights if all are 0
            walking_time_to_origin_weight = 0.4
            wait_time_weight = 0.3
            in_vehicle_time_weight = 0.2
            walking_time_from_destination_weight = 0.1
        
        # Generate reasonable values for time preferences based on weights
        max_walking_time_to_origin = random.uniform(
            1.0 if walking_time_to_origin_weight > 0.4 else 2.0,
            4.0 if walking_time_to_origin_weight > 0.4 else 6.0
        )
        
        max_walking_time_from_destination = random.uniform(
            1.0 if walking_time_from_destination_weight > 0.4 else 2.0,
            4.0 if walking_time_from_destination_weight > 0.4 else 6.0
        )
        
        # Users who value waiting time less may accept longer waits
        max_waiting_time = random.uniform(
            5.0 if wait_time_weight > 0.4 else 7.0,
            12.0 if wait_time_weight > 0.4 else 15.0
        )
        
        # Users who value travel time less may accept longer trips
        max_in_vehicle_time = random.uniform(
            15.0 if in_vehicle_time_weight > 0.4 else 20.0,
            30.0 if in_vehicle_time_weight > 0.4 else 40.0
        )
        
        # Generate reasonable value for max cost
        max_cost = random.uniform(20.0, 40.0)
        
        # Generate reasonable value for acceptable delay
        max_acceptable_delay = random.uniform(5.0, 10.0)
        
        # Simplified service preference determination based only on what we can reasonably infer
        # If in-vehicle time weight is higher than wait time weight, prefer SPEED, otherwise RELIABILITY
        service_preference = ServicePreference.SPEED if in_vehicle_time_weight > wait_time_weight else ServicePreference.RELIABILITY
        
        # Track statistics
        if service_preference.value in profile_stats['service_preferences']:
            profile_stats['service_preferences'][service_preference.value] += 1
        else:
            profile_stats['service_preferences'][service_preference.value] = 1
        
        # Generate historical data
        historical_trips = random.randint(0, 50)
        historical_acceptance_rate = random.uniform(0.5, 0.95)
        historical_ratings = [round(random.uniform(3.0, 5.0), 1) for _ in range(min(5, historical_trips))]
        
        # Update stats
        profile_stats['avg_historical_trips'] += historical_trips
        profile_stats['avg_max_walking_time_to_origin'] += max_walking_time_to_origin
        profile_stats['avg_max_walking_time_from_destination'] += max_walking_time_from_destination
        profile_stats['avg_max_waiting_time'] += max_waiting_time
        profile_stats['avg_max_in_vehicle_time'] += max_in_vehicle_time
        
        # Create user profile matching the UserProfile.to_dict() format
        profile = {
            "id": user_id,
            "max_walking_time_to_origin": max_walking_time_to_origin,
            "max_walking_time_from_destination": max_walking_time_from_destination,
            "max_waiting_time": max_waiting_time,
            "max_in_vehicle_time": max_in_vehicle_time,
            "max_cost": max_cost,
            "max_acceptable_delay": max_acceptable_delay,
            "service_preference": service_preference.value,
            "weights": {
                "walking_time_to_origin": walking_time_to_origin_weight,
                "wait_time": wait_time_weight,
                "in_vehicle_time": in_vehicle_time_weight,
                "walking_time_from_destination": walking_time_from_destination_weight,
                "time_of_day": 0.0,
                "day_of_week": 0.0,
                "distance_to_pickup": 0.0
            },
            "historical_trips": historical_trips,
            "historical_acceptance_rate": historical_acceptance_rate,
            "historical_ratings": historical_ratings
        }
        
        # Save profile to JSON file
        profile_path = os.path.join(profiles_dir, f"{user_id}.json")
        with open(profile_path, 'w') as f:
            json.dump(profile, f, indent=2)
        
        if int(user_id.split('U')[1]) % 5 == 0:
            logger.info(f"Created {user_id} profile")
    
    # Calculate averages for stats
    if profile_stats['total_profiles'] > 0:
        profile_stats['avg_historical_trips'] /= profile_stats['total_profiles']
        profile_stats['avg_max_walking_time_to_origin'] /= profile_stats['total_profiles']
        profile_stats['avg_max_walking_time_from_destination'] /= profile_stats['total_profiles']
        profile_stats['avg_max_waiting_time'] /= profile_stats['total_profiles']
        profile_stats['avg_max_in_vehicle_time'] /= profile_stats['total_profiles']
    
    # Save summary statistics
    stats_path = os.path.join(stats_dir, 'profile_stats.json')
    with open(stats_path, 'w') as f:
        json.dump(profile_stats, f, indent=2)
    
    logger.info(f"Created {profile_stats['total_profiles']} user profiles in {profiles_dir}")
    logger.info(f"Summary statistics saved to {stats_path}")
    
    # Return the paths for verification
    return {
        'profiles_dir': profiles_dir,
        'stats_path': stats_path,
        'total_profiles': profile_stats['total_profiles']
    }

def create_weight_mapping_file(output_dir):
    """
    Create a mapping file explaining how CSV weights map to feature names.
    
    Args:
        output_dir: Directory to save the mapping file
    """
    mapping = {
        "csv_column": "feature_name",
        "access": "walking_time_to_origin",
        "wait": "wait_time",
        "ivt": "in_vehicle_time",
        "egress": "walking_time_from_destination"
    }
    
    mapping_path = os.path.join(output_dir, 'weight_mapping.json')
    with open(mapping_path, 'w') as f:
        json.dump(mapping, f, indent=2)
    
    logger.info(f"Weight mapping saved to {mapping_path}")
    return mapping_path

def main():
    """Main function to execute the profile creation process."""
    parser = argparse.ArgumentParser(description='Create user profiles from acceptance weights')
    parser.add_argument('--weights', type=str, default='data/user/acceptance_weights.csv', 
                        help='Path to the acceptance weights CSV file')
    parser.add_argument('--output', type=str, default='data/users', 
                        help='Base directory to save user profiles and analysis')
    parser.add_argument('--analyze', action='store_true',
                        help='Perform analysis on weights data (requires matplotlib and seaborn)')
    
    args = parser.parse_args()
    
    # Normalize paths
    weights_file = Path(args.weights)
    output_dir = Path(args.output)
    
    # Ensure the weights file exists
    if not weights_file.exists():
        logger.error(f"Weights file not found: {weights_file}")
        return
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Create profiles
    result = create_user_profiles(weights_file, output_dir)
    
    # Create weight mapping
    create_weight_mapping_file(output_dir)
    
    # Analyze weights if requested
    if args.analyze:
        try:
            import matplotlib.pyplot as plt
            import seaborn as sns
            
            # Create output directory for plots
            plots_dir = os.path.join(output_dir, 'plots')
            os.makedirs(plots_dir, exist_ok=True)
            
            # Read the weights data
            weights_df = pd.read_csv(weights_file)
            
            # Calculate correlations - exclude the 'id' column
            numeric_columns = ['access', 'wait', 'ivt', 'egress']
            corr = weights_df[numeric_columns].corr()
            
            # Plot correlation heatmap
            plt.figure(figsize=(10, 8))
            sns.heatmap(corr, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
            plt.title('Correlation Between Weight Parameters')
            plt.tight_layout()
            plt.savefig(os.path.join(plots_dir, 'weight_correlations.png'))
            
            # Plot distributions of each weight
            plt.figure(figsize=(12, 10))
            for i, column in enumerate(numeric_columns):
                plt.subplot(2, 2, i+1)
                sns.histplot(weights_df[column], kde=True)
                plt.title(f'Distribution of {column}')
                plt.axvline(x=0, color='r', linestyle='--')
            plt.tight_layout()
            plt.savefig(os.path.join(plots_dir, 'weight_distributions.png'))
            
            logger.info(f"Weight analysis plots saved to {plots_dir}")
        except ImportError:
            logger.warning("matplotlib and seaborn are required for weight analysis. Skipping.")
        except Exception as e:
            logger.error(f"Error during weight analysis: {traceback.format_exc()}")
    logger.info("Profile creation process completed")

if __name__ == "__main__":
    main()