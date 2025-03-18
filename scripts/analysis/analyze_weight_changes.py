"""
Script to analyze weight changes over time across simulation studies.

This script provides tools to analyze how user preferences (weights) evolve
over time as they interact with the system, allowing researchers to better
understand adaptation patterns and user learning.
"""
import os
import json
import argparse
import logging
from pathlib import Path
from datetime import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Any, Optional, Tuple

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def load_user_profile(profile_path: str) -> Dict:
    """
    Load user profile from a JSON file.
    
    Args:
        profile_path: Path to the user profile JSON file
        
    Returns:
        Dict: User profile data
    """
    try:
        with open(profile_path, 'r') as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Error loading profile {profile_path}: {e}")
        return {}

def load_change_logs(change_logs_dir: str, user_id: str = None, study_id: str = None, simulation_id: str = None) -> Dict:
    """
    Load change logs for one or more users.
    
    Args:
        change_logs_dir: Directory containing change logs
        user_id: Optional user ID to filter
        study_id: Optional study ID to filter
        simulation_id: Optional simulation ID to filter (requires study_id)
        
    Returns:
        Dict: Change logs organized by user ID, study ID, and simulation ID
    """
    result = {}
    
    if not os.path.exists(change_logs_dir):
        logger.error(f"Change logs directory not found: {change_logs_dir}")
        return result
    
    # If specific user, load only that user's logs
    if user_id:
        return load_user_change_logs(change_logs_dir, user_id, study_id, simulation_id)
    
    # Load all user logs
    study_dirs = [d for d in os.listdir(change_logs_dir) if os.path.isdir(os.path.join(change_logs_dir, d))]
    
    # Filter by study if specified
    if study_id:
        study_dirs = [d for d in study_dirs if d == study_id]
    
    for study in study_dirs:
        study_path = os.path.join(change_logs_dir, study)
        result[study] = {}
        
        sim_dirs = [d for d in os.listdir(study_path) if os.path.isdir(os.path.join(study_path, d))]
        
        # Filter by simulation if specified
        if simulation_id and study_id:
            sim_dirs = [d for d in sim_dirs if d == simulation_id]
        
        for sim in sim_dirs:
            sim_path = os.path.join(study_path, sim)
            result[study][sim] = {}
            
            log_files = [f for f in os.listdir(sim_path) if f.endswith('_changes.json')]
            
            for log_file in log_files:
                uid = log_file.split('_changes.json')[0]
                log_path = os.path.join(sim_path, log_file)
                
                try:
                    with open(log_path, 'r') as f:
                        result[study][sim][uid] = json.load(f)
                except Exception as e:
                    logger.error(f"Error loading change log {log_path}: {e}")
                    result[study][sim][uid] = []
    
    return result

def load_user_change_logs(change_logs_dir: str, user_id: str, study_id: str = None, simulation_id: str = None) -> Dict:
    """
    Load change logs for a specific user.
    
    Args:
        change_logs_dir: Directory containing change logs
        user_id: User ID to load logs for
        study_id: Optional study ID to filter
        simulation_id: Optional simulation ID to filter (requires study_id)
        
    Returns:
        Dict: Change logs organized by study ID and simulation ID
    """
    result = {}
    
    if not os.path.exists(change_logs_dir):
        logger.error(f"Change logs directory not found: {change_logs_dir}")
        return result
    
    study_dirs = [d for d in os.listdir(change_logs_dir) if os.path.isdir(os.path.join(change_logs_dir, d))]
    
    # Filter by study if specified
    if study_id:
        study_dirs = [d for d in study_dirs if d == study_id]
    
    for study in study_dirs:
        study_path = os.path.join(change_logs_dir, study)
        result[study] = {}
        
        sim_dirs = [d for d in os.listdir(study_path) if os.path.isdir(os.path.join(study_path, d))]
        
        # Filter by simulation if specified
        if simulation_id and study_id:
            sim_dirs = [d for d in sim_dirs if d == simulation_id]
        
        for sim in sim_dirs:
            sim_path = os.path.join(study_path, sim)
            log_path = os.path.join(sim_path, f"{user_id}_changes.json")
            
            if os.path.exists(log_path):
                try:
                    with open(log_path, 'r') as f:
                        result[study][sim] = json.load(f)
                except Exception as e:
                    logger.error(f"Error loading change log {log_path}: {e}")
                    result[study][sim] = []
            else:
                result[study][sim] = []
    
    return result

def analyze_weight_changes(change_logs: Dict, weight_names: List[str] = None) -> Dict:
    """
    Analyze weight changes across users, studies, and simulations.
    
    Args:
        change_logs: Change logs organized by user ID, study ID, and simulation ID
        weight_names: Optional list of weight names to analyze
        
    Returns:
        Dict: Analysis results
    """
    # Initialize statistics
    stats = {
        "updates_per_user": {},
        "updates_per_study": {},
        "updates_per_simulation": {},
        "weight_changes": {},
        "total_changes": 0
    }
    
    # Extract weight names if not provided
    if not weight_names:
        weight_names = set()
        for study in change_logs.values():
            for sim in study.values():
                for user_logs in sim.values():
                    for entry in user_logs:
                        if "weights" in entry:
                            weight_names.update(entry["weights"].keys())
        weight_names = sorted(list(weight_names))
    
    # Initialize weight change stats
    for weight_name in weight_names:
        stats["weight_changes"][weight_name] = {
            "avg_change": 0.0,
            "max_increase": 0.0,
            "max_decrease": 0.0,
            "total_updates": 0,
            "changes": []
        }
    
    # Process change logs
    for study_id, study_data in change_logs.items():
        if study_id not in stats["updates_per_study"]:
            stats["updates_per_study"][study_id] = 0
        
        for sim_id, sim_data in study_data.items():
            if sim_id not in stats["updates_per_simulation"]:
                stats["updates_per_simulation"][sim_id] = 0
            
            for user_id, user_logs in sim_data.items():
                if user_id not in stats["updates_per_user"]:
                    stats["updates_per_user"][user_id] = 0
                
                # Filter to just weight update entries
                weight_updates = [entry for entry in user_logs if entry.get("type") == "weight_update"]
                stats["updates_per_user"][user_id] += len(weight_updates)
                stats["updates_per_study"][study_id] += len(weight_updates)
                stats["updates_per_simulation"][sim_id] += len(weight_updates)
                stats["total_changes"] += len(weight_updates)
                
                # Analyze weight changes
                for update in weight_updates:
                    if "weights" in update and "prev_weights" in update:
                        for weight_name in weight_names:
                            if weight_name in update["weights"] and weight_name in update["prev_weights"]:
                                new_value = update["weights"][weight_name]
                                old_value = update["prev_weights"][weight_name]
                                change = new_value - old_value
                                
                                stats["weight_changes"][weight_name]["total_updates"] += 1
                                stats["weight_changes"][weight_name]["changes"].append(change)
                                
                                if change > stats["weight_changes"][weight_name]["max_increase"]:
                                    stats["weight_changes"][weight_name]["max_increase"] = change
                                
                                if change < stats["weight_changes"][weight_name]["max_decrease"]:
                                    stats["weight_changes"][weight_name]["max_decrease"] = change
    
    # Calculate averages
    for weight_name in weight_names:
        if stats["weight_changes"][weight_name]["total_updates"] > 0:
            changes = stats["weight_changes"][weight_name]["changes"]
            stats["weight_changes"][weight_name]["avg_change"] = sum(changes) / len(changes)
    
    return stats

def plot_weight_evolution(change_logs: Dict, user_id: str, weight_names: List[str], output_dir: str) -> None:
    """
    Plot the evolution of weights over time for a specific user.
    
    Args:
        change_logs: Change logs organized by study ID and simulation ID
        user_id: User ID to plot
        weight_names: List of weight names to plot
        output_dir: Directory to save the plots
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Extract all weight updates across studies and simulations
    weight_updates = []
    
    for study_id, study_data in change_logs.items():
        for sim_id, sim_logs in study_data.items():
            for entry in sim_logs:
                if "weights" in entry and "timestamp" in entry:
                    # Convert timestamp to datetime
                    try:
                        dt = datetime.fromisoformat(entry["timestamp"])
                    except ValueError:
                        dt = datetime.strptime(entry["timestamp"], "%Y-%m-%dT%H:%M:%S.%f")
                    
                    update = {
                        "timestamp": dt,
                        "study_id": study_id,
                        "simulation_id": sim_id,
                        "reason": entry.get("reason", "Unknown")
                    }
                    
                    # Add weight values
                    for weight_name in weight_names:
                        if weight_name in entry["weights"]:
                            update[weight_name] = entry["weights"][weight_name]
                    
                    weight_updates.append(update)
    
    # Sort by timestamp
    weight_updates.sort(key=lambda x: x["timestamp"])
    
    if not weight_updates:
        logger.warning(f"No weight updates found for user {user_id}")
        return
    
    # Create DataFrame
    df = pd.DataFrame(weight_updates)
    
    # Plot evolution of each weight
    plt.figure(figsize=(12, 8))
    
    for i, weight_name in enumerate(weight_names):
        if weight_name in df.columns:
            plt.subplot(len(weight_names), 1, i+1)
            plt.plot(df["timestamp"], df[weight_name], marker='o', linestyle='-')
            plt.ylabel(weight_name)
            plt.grid(True, linestyle='--', alpha=0.7)
            
            # Add study/simulation transitions
            prev_study_sim = None
            for idx, row in df.iterrows():
                current_study_sim = (row["study_id"], row["simulation_id"])
                if prev_study_sim is not None and current_study_sim != prev_study_sim:
                    plt.axvline(x=row["timestamp"], color='r', linestyle='--', alpha=0.5)
                prev_study_sim = current_study_sim
    
    plt.title(f"Weight Evolution for User {user_id}")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{user_id}_weight_evolution.png"))
    plt.close()
    
    # Create heatmap of weight correlations
    weight_cols = [col for col in df.columns if col in weight_names]
    if len(weight_cols) > 1:
        plt.figure(figsize=(10, 8))
        corr = df[weight_cols].corr()
        sns.heatmap(corr, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
        plt.title(f"Weight Correlations for User {user_id}")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"{user_id}_weight_correlations.png"))
        plt.close()

def compare_user_adaptations(change_logs: Dict, weight_name: str, output_dir: str) -> None:
    """
    Compare how different users adapt their weights over time.
    
    Args:
        change_logs: Change logs organized by user ID, study ID, and simulation ID
        weight_name: Weight name to compare
        output_dir: Directory to save the comparison plot
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Extract weight evolution for each user
    user_weights = {}
    
    for study_id, study_data in change_logs.items():
        for sim_id, sim_data in study_data.items():
            for user_id, user_logs in sim_data.items():
                if user_id not in user_weights:
                    user_weights[user_id] = []
                
                for entry in user_logs:
                    if "weights" in entry and weight_name in entry["weights"] and "timestamp" in entry:
                        # Convert timestamp to datetime
                        try:
                            dt = datetime.fromisoformat(entry["timestamp"])
                        except ValueError:
                            dt = datetime.strptime(entry["timestamp"], "%Y-%m-%dT%H:%M:%S.%f")
                        
                        user_weights[user_id].append({
                            "timestamp": dt,
                            "weight": entry["weights"][weight_name],
                            "study_id": study_id,
                            "simulation_id": sim_id
                        })
    
    # Sort by timestamp for each user
    for user_id in user_weights:
        user_weights[user_id].sort(key=lambda x: x["timestamp"])
    
    # Keep only users with sufficient data
    active_users = {uid: data for uid, data in user_weights.items() if len(data) >= 3}
    
    if not active_users:
        logger.warning(f"No users with sufficient weight update data for {weight_name}")
        return
    
    # Plot comparison
    plt.figure(figsize=(12, 8))
    
    for user_id, updates in active_users.items():
        timestamps = [u["timestamp"] for u in updates]
        weights = [u["weight"] for u in updates]
        plt.plot(timestamps, weights, marker='o', linestyle='-', label=user_id)
    
    plt.xlabel("Time")
    plt.ylabel(f"{weight_name} Weight")
    plt.title(f"Comparison of {weight_name} Weight Adaptation Across Users")
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{weight_name}_user_comparison.png"))
    plt.close()

def main():
    """Main function to execute the analysis."""
    parser = argparse.ArgumentParser(description='Analyze weight changes across simulation studies')
    parser.add_argument('--profiles', type=str, default='data/users/user_profiles',
                       help='Directory containing user profiles')
    parser.add_argument('--logs', type=str, default='data/users/change_logs',
                       help='Directory containing change logs')
    parser.add_argument('--output', type=str, default='data/analysis',
                       help='Directory to save analysis results')
    parser.add_argument('--user', type=str, default=None,
                       help='Optional user ID to analyze specifically')
    parser.add_argument('--study', type=str, default=None,
                       help='Optional study ID to analyze specifically')
    parser.add_argument('--simulation', type=str, default=None,
                       help='Optional simulation ID to analyze specifically')
    parser.add_argument('--weights', type=str, nargs='+', default=None,
                       help='Optional specific weights to analyze')
    
    args = parser.parse_args()
    
    # Ensure output directory exists
    os.makedirs(args.output, exist_ok=True)
    
    # Load change logs
    logger.info("Loading change logs...")
    change_logs = load_change_logs(args.logs, args.user, args.study, args.simulation)
    
    if not change_logs:
        logger.error("No change logs found. Aborting analysis.")
        return
    
    # Analyze weight changes
    logger.info("Analyzing weight changes...")
    stats = analyze_weight_changes(change_logs, args.weights)
    
    # Save statistics
    stats_path = os.path.join(args.output, 'weight_change_stats.json')
    with open(stats_path, 'w') as f:
        json.dump(stats, f, indent=2)
    logger.info(f"Saved weight change statistics to {stats_path}")
    
    # Generate plots
    if args.user:
        logger.info(f"Generating plots for user {args.user}...")
        user_logs = load_user_change_logs(args.logs, args.user, args.study, args.simulation)
        
        # Determine weight names to plot
        weight_names = args.weights
        if not weight_names:
            weight_names = set()
            for study in user_logs.values():
                for sim in study.values():
                    for entry in sim:
                        if "weights" in entry:
                            weight_names.update(entry["weights"].keys())
            weight_names = sorted(list(weight_names))
        
        plot_weight_evolution(user_logs, args.user, weight_names, args.output)
    else:
        # Compare adaptations across users for each weight
        if args.weights:
            for weight_name in args.weights:
                logger.info(f"Comparing user adaptations for weight {weight_name}...")
                compare_user_adaptations(change_logs, weight_name, args.output)
    
    logger.info("Analysis complete!")

if __name__ == "__main__":
    main() 