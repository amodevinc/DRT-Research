"""
Script to create user profiles based on acceptance weights data.

This script reads the acceptance weights from a CSV file or generates synthetic weights
to create user profiles with reasonable default values and proper weight mappings.
It also sets up logging for tracking changes to profiles over time, especially weight
modifications across simulation studies.
"""
import os
import pandas as pd
import json
import random
import logging
import argparse
from pathlib import Path
import traceback
from datetime import datetime
import numpy as np

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def generate_synthetic_weights(num_users=2000):
    """
    Generate synthetic weights for user profiles based on realistic distributions.
    
    Args:
        num_users: Number of user profiles to generate
        
    Returns:
        DataFrame containing synthetic weights
    """
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Generate synthetic weights with realistic distributions
    # Using normal distributions with different means and standard deviations
    # for each parameter to create heterogeneity in preferences
    
    # Walking time to origin (access) - typically negative, more sensitive
    access_weights = np.random.normal(-0.8, 0.2, num_users)
    
    # Waiting time - typically negative, very sensitive
    wait_weights = np.random.normal(-1.0, 0.15, num_users)
    
    # In-vehicle time - typically negative, less sensitive
    ivt_weights = np.random.normal(-0.5, 0.1, num_users)
    
    # Walking time from destination (egress) - typically negative, moderate sensitivity
    egress_weights = np.random.normal(-0.6, 0.15, num_users)
    
    # Create DataFrame
    weights_df = pd.DataFrame({
        'id': [f'U{i+1:04d}' for i in range(num_users)],
        'access': access_weights,
        'wait': wait_weights,
        'ivt': ivt_weights,
        'egress': egress_weights
    })
    
    # Ensure all weights are negative (as per utility theory)
    for col in ['access', 'wait', 'ivt', 'egress']:
        weights_df[col] = weights_df[col].apply(lambda x: -abs(x))
    
    # Add some correlation between weights to make them more realistic
    # For example, correlation between access and egress
    correlation = 0.7
    weights_df['egress'] = correlation * weights_df['access'] + \
                          (1 - correlation) * weights_df['egress']
    
    logger.info(f"Generated synthetic weights for {num_users} users")
    return weights_df

def create_user_profiles(weights_df, output_dir, profile_folder='user_profiles'):
    """
    Create user profiles based on the weights DataFrame.
    
    Args:
        weights_df: DataFrame containing user weights
        output_dir: Base directory to save profiles
        profile_folder: Subfolder to store user profiles
    """
    # Create output directory
    profiles_dir = os.path.join(output_dir, profile_folder)
    os.makedirs(profiles_dir, exist_ok=True)
    
    # Create a directory for any summary data
    stats_dir = os.path.join(output_dir, 'stats')
    os.makedirs(stats_dir, exist_ok=True)
    
    # Create directory structure for change logs
    change_logs_dir = os.path.join(output_dir, 'change_logs')
    os.makedirs(change_logs_dir, exist_ok=True)
    
    # Default studies and simulations directories
    default_study_dir = os.path.join(change_logs_dir, 'default_study')
    os.makedirs(default_study_dir, exist_ok=True)
    default_sim_dir = os.path.join(default_study_dir, 'base')
    os.makedirs(default_sim_dir, exist_ok=True)
    
    # Summary statistics
    profile_stats = {
        'total_profiles': len(weights_df),
        'real_profiles': len(weights_df[~weights_df['is_synthetic']]),
        'synthetic_profiles': len(weights_df[weights_df['is_synthetic']]),
        'avg_max_walking_time_to_origin': 0,
        'avg_max_walking_time_from_destination': 0,
        'avg_max_waiting_time': 0,
        'avg_max_in_vehicle_time': 0,
    }
    
    # Process each user
    for _, row in weights_df.iterrows():
        user_id = row['id']
        
        # Map CSV weights to feature name - use exact weights from CSV
        walking_time_to_origin_weight = row['access']
        wait_time_weight = row['wait']
        in_vehicle_time_weight = row['ivt']
        walking_time_from_destination_weight = row['egress']
        
        # Create base weights dictionary - use exact coefficients without normalization
        base_weights = {
            "walking_time_to_origin": walking_time_to_origin_weight,
            "waiting_time": wait_time_weight,
            "in_vehicle_time": in_vehicle_time_weight,
            "walking_time_from_destination": walking_time_from_destination_weight,
        }
        
        # Set reasonable default values for time preferences (not based on weights)
        max_walking_time_to_origin = 5.0  # minutes
        max_walking_time_from_destination = 5.0  # minutes
        max_waiting_time = 10.0  # minutes
        max_in_vehicle_time = 20.0  # minutes
        max_price = 1.0  # currency units
        max_acceptable_delay = 7.0  # minutes
        
        # Update stats
        profile_stats['avg_max_walking_time_to_origin'] += max_walking_time_to_origin
        profile_stats['avg_max_walking_time_from_destination'] += max_walking_time_from_destination
        profile_stats['avg_max_waiting_time'] += max_waiting_time
        profile_stats['avg_max_in_vehicle_time'] += max_in_vehicle_time
        
        # Create weight history to track changes over time
        weight_history = [{
            "timestamp": datetime.now().isoformat(),
            "weights": base_weights.copy(),
            "study_id": "default_study",
            "simulation_id": "base",
            "reason": "Initial profile creation"
        }]
        
        # Create user profile - no historical trips
        profile = {
            "id": user_id,
            "is_synthetic": row['is_synthetic'],
            "max_walking_time_to_origin": max_walking_time_to_origin,
            "max_walking_time_from_destination": max_walking_time_from_destination,
            "max_waiting_time": max_waiting_time,
            "max_in_vehicle_time": max_in_vehicle_time,
            "max_price": max_price,
            "max_acceptable_delay": max_acceptable_delay,
            "base_weights": base_weights.copy(),
            "weights": base_weights.copy(),
            "weight_history": weight_history,
            "historical_trips": 0,
            "historical_acceptance_rate": 0.0,
            "historical_ratings": [],
            "created_at": datetime.now().isoformat(),
            "last_updated": datetime.now().isoformat()
        }
        
        # Save profile to JSON file
        profile_path = os.path.join(profiles_dir, f"{user_id}.json")
        with open(profile_path, 'w') as f:
            json.dump(profile, f, indent=2)
        
        # Initialize change log for this user
        change_log = [{
            "timestamp": datetime.now().isoformat(),
            "type": "creation",
            "study_id": "default_study",
            "simulation_id": "base",
            "description": "Profile created",
            "weights": base_weights.copy()
        }]
        
        # Save change log
        change_log_path = os.path.join(default_sim_dir, f"{user_id}_changes.json")
        with open(change_log_path, 'w') as f:
            json.dump(change_log, f, indent=2)
        
        if int(user_id.split('U')[1]) % 5 == 0:
            logger.info(f"Created {user_id} profile and change log")
    
    # Calculate averages for stats
    if profile_stats['total_profiles'] > 0:
        profile_stats['avg_max_walking_time_to_origin'] /= profile_stats['total_profiles']
        profile_stats['avg_max_walking_time_from_destination'] /= profile_stats['total_profiles']
        profile_stats['avg_max_waiting_time'] /= profile_stats['total_profiles']
        profile_stats['avg_max_in_vehicle_time'] /= profile_stats['total_profiles']
    
    # Save summary statistics
    stats_path = os.path.join(stats_dir, 'profile_stats.json')
    with open(stats_path, 'w') as f:
        json.dump(profile_stats, f, indent=2)
    
    logger.info(f"Created {profile_stats['total_profiles']} user profiles in {profiles_dir}")
    logger.info(f"Real profiles: {profile_stats['real_profiles']}, Synthetic profiles: {profile_stats['synthetic_profiles']}")
    logger.info(f"Created change logs in {change_logs_dir}")
    logger.info(f"Summary statistics saved to {stats_path}")
    
    # Return the paths for verification
    return {
        'profiles_dir': profiles_dir,
        'change_logs_dir': change_logs_dir,
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
        "wait": "waiting_time",
        "ivt": "in_vehicle_time",
        "egress": "walking_time_from_destination"
    }
    
    mapping_path = os.path.join(output_dir, 'weight_mapping.json')
    with open(mapping_path, 'w') as f:
        json.dump(mapping, f, indent=2)
    
    logger.info(f"Weight mapping saved to {mapping_path}")
    return mapping_path

def log_profile_change(user_id, change_logs_dir, study_id, simulation_id, change_type, description, weights=None):
    """
    Log changes to a user profile.
    
    Args:
        user_id: ID of the user whose profile was changed
        change_logs_dir: Base directory containing change logs
        study_id: ID of the current study
        simulation_id: ID of the current simulation run
        change_type: Type of change (e.g., 'weight_update', 'preference_change')
        description: Description of the change
        weights: New weights if applicable
    """
    # Create study and simulation directories if they don't exist
    study_dir = os.path.join(change_logs_dir, study_id)
    os.makedirs(study_dir, exist_ok=True)
    
    sim_dir = os.path.join(study_dir, simulation_id)
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
        "study_id": study_id,
        "simulation_id": simulation_id,
        "description": description
    }
    
    # Add weights if provided
    if weights:
        log_entry["weights"] = weights.copy()
    
    # Add to log and save
    change_log.append(log_entry)
    try:
        with open(change_log_path, 'w') as f:
            json.dump(change_log, f, indent=2)
        logger.debug(f"Updated change log for {user_id} in study {study_id}, simulation {simulation_id}")
    except Exception as e:
        logger.error(f"Error saving change log for {user_id}: {e}")

def main():
    """Main function to execute the profile creation process."""
    parser = argparse.ArgumentParser(description='Create user profiles from acceptance weights')
    parser.add_argument('--weights', type=str, default='data/user/acceptance_weights.csv', 
                        help='Path to the acceptance weights CSV file')
    parser.add_argument('--output', type=str, default='data/users', 
                        help='Base directory to save user profiles and analysis')
    parser.add_argument('--analyze', action='store_true',
                        help='Perform analysis on weights data (requires matplotlib and seaborn)')
    parser.add_argument('--num-users', type=int, default=2000,
                        help='Total number of user profiles to generate')
    
    args = parser.parse_args()
    
    # Normalize paths
    weights_file = Path(args.weights)
    output_dir = Path(args.output)
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load real weights if available
    real_weights_df = None
    if weights_file.exists():
        try:
            real_weights_df = pd.read_csv(weights_file)
            real_weights_df['is_synthetic'] = False
            logger.info(f"Loaded {len(real_weights_df)} real weights from {weights_file}")
        except Exception as e:
            logger.error(f"Error loading weights file: {e}")
            real_weights_df = None
    
    # Generate synthetic weights for remaining users
    num_real_users = len(real_weights_df) if real_weights_df is not None else 0
    num_synthetic_users = max(0, args.num_users - num_real_users)
    
    if num_synthetic_users > 0:
        synthetic_weights_df = generate_synthetic_weights(num_synthetic_users)
        synthetic_weights_df['is_synthetic'] = True
        # Adjust IDs to continue from where real users left off
        synthetic_weights_df['id'] = [f'U{i+1+num_real_users:04d}' for i in range(num_synthetic_users)]
        logger.info(f"Generated {num_synthetic_users} synthetic weights")
        
        # Combine real and synthetic weights
        if real_weights_df is not None:
            weights_df = pd.concat([real_weights_df, synthetic_weights_df], ignore_index=True)
        else:
            weights_df = synthetic_weights_df
    else:
        weights_df = real_weights_df
    
    # Save combined weights for reference
    weights_file = output_dir / 'combined_weights.csv'
    weights_df.to_csv(weights_file, index=False)
    logger.info(f"Saved combined weights to {weights_file}")
    
    # Create profiles
    result = create_user_profiles(weights_df, output_dir)
    
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
            
            # Calculate correlations - exclude the 'id' and 'is_synthetic' columns
            numeric_columns = ['access', 'wait', 'ivt', 'egress']
            corr = weights_df[numeric_columns].corr()
            
            # Plot correlation heatmap
            plt.figure(figsize=(10, 8))
            sns.heatmap(corr, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
            plt.title('Correlation Between Weight Parameters')
            plt.tight_layout()
            plt.savefig(os.path.join(plots_dir, 'weight_correlations.png'))
            
            # Plot distributions of each weight, separated by real vs synthetic
            plt.figure(figsize=(15, 10))
            for i, column in enumerate(numeric_columns):
                plt.subplot(2, 2, i+1)
                sns.histplot(data=weights_df, x=column, hue='is_synthetic', multiple="layer", alpha=0.5)
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