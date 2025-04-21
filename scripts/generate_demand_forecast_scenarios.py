import os
import yaml
import pandas as pd
from typing import Dict, Tuple
from drt_sim.core.demand.od_converter import ODConverter
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_zone_centroids(centroids_file: str) -> Dict[int, Tuple[float, float]]:
    """
    Load zone centroids from CSV file
    
    Args:
        centroids_file: Path to CSV file with zone centroids
        
    Returns:
        Dictionary mapping zone IDs to (lat, lon) coordinates
    """
    df = pd.read_csv(centroids_file)
    return {
        row['zone_id']: (row['lat'], row['lon'])
        for _, row in df.iterrows()
    }

def generate_scenarios(
    od_dir: str,
    output_dir: str,
    centroids_file: str,
    start_time: str = "2024-10-24 07:00:00",
    end_time: str = "2024-10-24 22:00:00"
):
    """
    Generate real-time demand files for all OD matrices
    
    Args:
        od_dir: Directory containing OD matrix files
        output_dir: Directory to save real-time demand files
        centroids_file: Path to CSV file with zone centroids
        start_time: Start time for demand generation
        end_time: End time for demand generation
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Load zone centroids
    zone_centroids = load_zone_centroids(centroids_file)
    logger.info(f"Loaded {len(zone_centroids)} zone centroids")
    
    # Process each OD matrix file
    for filename in os.listdir(od_dir):
        if not filename.endswith('.csv'):
            continue
            
        # Get scenario name from filename
        scenario_name = filename.replace('.csv', '')
        
        # Create converter and generate demand
        od_file = os.path.join(od_dir, filename)
        output_file = os.path.join(output_dir, f"{scenario_name}_real_time.csv")
        
        converter = ODConverter(
            od_file=od_file,
            zone_centroids=zone_centroids,
            start_time=start_time,
            end_time=end_time
        )
        
        converter.save_demand(output_file)
        logger.info(f"Generated demand for scenario {scenario_name}")

def update_study_config(
    config_file: str,
    output_dir: str
):
    """
    Update study configuration with demand forecast error scenarios
    
    Args:
        config_file: Path to study configuration file
        output_dir: Directory containing real-time demand files
    """
    # Load study configuration
    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)
    
    # Create parameter sets for each scenario
    parameter_sets = {}
    
    # Add all scenarios
    for filename in os.listdir(output_dir):
        if not filename.endswith('_real_time.csv'):
            continue
            
        # Get scenario name from filename
        scenario_name = filename.replace('_real_time.csv', '')
        
        # Create parameter set for this scenario
        parameter_sets[scenario_name] = {
            'name': f'Forecast Error: {scenario_name}',
            'description': f'Demand forecast error scenario {scenario_name}',
            'demand': {
                'generator_type': 'csv',
                'csv_config': {
                    'files': [
                        {
                            'file_path': os.path.join(output_dir, filename),
                            'weight': 1.0
                        }
                    ]
                }
            }
        }
    
    # Update configuration
    config['parameter_sets'] = parameter_sets
    
    # Save updated configuration
    with open(config_file, 'w') as f:
        yaml.dump(config, f)
    
    logger.info(f"Updated study configuration with {len(parameter_sets)} scenarios")

def main():
    # Configuration
    OD_DIR = 'data/demands/rules'  # Directory containing OD matrices
    OUTPUT_DIR = 'data/demands/forecast_scenarios'  # Directory for generated real-time demand files
    CENTROIDS_FILE = 'data/zone_centroids.csv'  # Zone centroid coordinates
    
    # Generate real-time demand files
    generate_scenarios(
        od_dir=OD_DIR,
        output_dir=OUTPUT_DIR,
        centroids_file=CENTROIDS_FILE
    )

if __name__ == '__main__':
    main() 