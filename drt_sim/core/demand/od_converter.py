import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random
from typing import List, Dict, Tuple
import logging

logger = logging.getLogger(__name__)

class ODConverter:
    """Converts OD matrices into real-time demand files"""
    
    def __init__(
        self,
        od_file: str,
        zone_centroids: Dict[int, Tuple[float, float]],
        start_time: str = "2024-10-24 07:00:00",
        end_time: str = "2024-10-24 19:00:00",
        time_step: int = 60,
        random_seed: int = 42
    ):
        """
        Initialize the OD converter
        
        Args:
            od_file: Path to the OD matrix CSV file
            zone_centroids: Dictionary mapping zone IDs to (lat, lon) coordinates
            start_time: Start time for demand generation
            end_time: End time for demand generation
            time_step: Time step in seconds for demand generation
            random_seed: Random seed for reproducibility
        """
        self.od_file = od_file
        self.zone_centroids = zone_centroids
        self.start_time = pd.to_datetime(start_time)
        self.end_time = pd.to_datetime(end_time)
        self.time_step = time_step
        self.random_seed = random_seed
        random.seed(random_seed)
        
        # Load OD matrix
        self.od_matrix = pd.read_csv(od_file)
        
        # Validate OD matrix format
        required_columns = ['o', 'd'] + [f't_{h}' for h in range(7, 23)]
        missing_columns = set(required_columns) - set(self.od_matrix.columns)
        if missing_columns:
            raise ValueError(f"OD matrix missing required columns: {missing_columns}")
            
        # Get unique zones from OD matrix
        od_zones = set(self.od_matrix['o'].unique()) | set(self.od_matrix['d'].unique())
        
        # Create mapping from OD matrix zone IDs to centroid zone IDs
        self.zone_mapping = {}
        for zone in od_zones:
            # Find the closest centroid
            min_dist = float('inf')
            closest_centroid = None
            
            for centroid_id, (centroid_lat, centroid_lon) in zone_centroids.items():
                # Simple distance calculation (can be improved with haversine)
                dist = abs(zone - centroid_id)
                if dist < min_dist:
                    min_dist = dist
                    closest_centroid = centroid_id
            
            self.zone_mapping[zone] = closest_centroid
        
        logger.info(f"Created zone mapping for {len(self.zone_mapping)} zones")
    
    def generate_demand(self) -> pd.DataFrame:
        """
        Generate real-time demand from OD matrix
        
        Returns:
            DataFrame with real-time demand records
        """
        records = []
        idx = 0
        
        # Generate demand for each hour
        for hour in range(7, 23):
            # Get trips for this hour
            hour_trips = self.od_matrix[['o', 'd', f't_{hour}']].copy()
            hour_trips = hour_trips[hour_trips[f't_{hour}'] > 0]
            
            if len(hour_trips) == 0:
                continue
                
            # Calculate time range for this hour
            hour_start = self.start_time + timedelta(hours=hour-7)
            hour_end = hour_start + timedelta(hours=1)
            
            # Generate trips for each OD pair
            for _, row in hour_trips.iterrows():
                o_zone = int(row['o'])
                d_zone = int(row['d'])
                n_trips = int(row[f't_{hour}'])
                
                # Get mapped zone centroids
                o_centroid = self.zone_mapping[o_zone]
                d_centroid = self.zone_mapping[d_zone]
                
                # Get zone centroids
                o_lat, o_lon = self.zone_centroids[o_centroid]
                d_lat, d_lon = self.zone_centroids[d_centroid]
                
                # Generate n_trips requests with random times within the hour
                for _ in range(n_trips):
                    # Generate random time within the hour
                    random_seconds = random.randint(0, 3600)
                    request_time = hour_start + timedelta(seconds=random_seconds)
                    
                    # Add small random offsets to coordinates to avoid exact zone centroids
                    o_lat_offset = random.uniform(-0.001, 0.001)
                    o_lon_offset = random.uniform(-0.001, 0.001)
                    d_lat_offset = random.uniform(-0.001, 0.001)
                    d_lon_offset = random.uniform(-0.001, 0.001)
                    
                    records.append({
                        'idx': idx,
                        'o': o_zone,  # Keep original zone IDs
                        'd': d_zone,  # Keep original zone IDs
                        'hour': hour,
                        'o_h3': '',  # H3 index not used in this implementation
                        'd_h3': '',  # H3 index not used in this implementation
                        'o_x': o_lon + o_lon_offset,
                        'o_y': o_lat + o_lat_offset,
                        'd_x': d_lon + d_lon_offset,
                        'd_y': d_lat + d_lat_offset,
                        'time': request_time.strftime("%Y-%m-%d %H:%M:%S")
                    })
                    idx += 1
        
        # Convert to DataFrame and sort by time
        df = pd.DataFrame(records)
        df = df.sort_values('time')
        df['idx'] = range(len(df))  # Reindex after sorting
        
        return df
    
    def save_demand(self, output_file: str):
        """
        Generate and save real-time demand to CSV file
        
        Args:
            output_file: Path to save the output CSV file
        """
        df = self.generate_demand()
        df.to_csv(output_file, index=False)
        logger.info(f"Saved {len(df)} demand records to {output_file}") 