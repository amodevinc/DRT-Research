import geopandas as gpd
import pandas as pd
import os
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def check_shapefile_exists(shapefile_path: str) -> bool:
    """
    Check if all required shapefile components exist
    
    Args:
        shapefile_path: Path to the shapefile (without extension)
        
    Returns:
        bool: True if all components exist, False otherwise
    """
    required_extensions = ['.shp', '.shx', '.dbf', '.prj']
    base_path = Path(shapefile_path)
    
    # Check if directory exists
    if not base_path.parent.exists():
        logger.error(f"Directory does not exist: {base_path.parent}")
        return False
        
    # Check for required files
    missing_files = []
    for ext in required_extensions:
        if not (base_path.parent / f"{base_path.name}{ext}").exists():
            missing_files.append(f"{base_path.name}{ext}")
    
    if missing_files:
        logger.error(f"Missing shapefile components: {', '.join(missing_files)}")
        return False
        
    return True

def convert_zone_centroids(
    centroid_shapefile: str,
    output_file: str,
    id_field: str = "Zone_ID",  # Updated to match your shapefile
    x_field: str = "center_x",  # Updated to match your shapefile
    y_field: str = "center_y"   # Updated to match your shapefile
):
    """
    Convert zone centroid shapefile to CSV format
    
    Args:
        centroid_shapefile: Path to the zone centroid shapefile (without extension)
        output_file: Path to save the output CSV file
        id_field: Name of the field containing zone IDs
        x_field: Name of the field containing x coordinates
        y_field: Name of the field containing y coordinates
    """
    try:
        # Check if shapefile exists
        if not check_shapefile_exists(centroid_shapefile):
            raise FileNotFoundError(f"Shapefile components missing for {centroid_shapefile}")
        
        # Read the shapefile
        logger.info(f"Reading shapefile: {centroid_shapefile}.shp")
        gdf = gpd.read_file(f"{centroid_shapefile}.shp")
        
        # Log available fields
        logger.info(f"Available fields in shapefile: {', '.join(gdf.columns)}")
        
        # Create output DataFrame using the specified fields
        df = pd.DataFrame({
            'zone_id': gdf[id_field],
            'lat': gdf[y_field],  # Note: y is latitude
            'lon': gdf[x_field]   # Note: x is longitude
        })
        
        # Save to CSV
        df.to_csv(output_file, index=False)
        logger.info(f"Saved {len(df)} zone centroids to {output_file}")
        
        # Print first few rows for verification
        logger.info("\nFirst few rows of converted data:")
        logger.info(df.head())
        
        return df
        
    except Exception as e:
        logger.error(f"Error converting shapefile: {str(e)}")
        raise

def main():
    # Configuration
    SHAPEFILE_DIR = "data/zone_shapefiles"  # Directory containing your shapefiles
    CENTROID_SHAPEFILE = os.path.join(SHAPEFILE_DIR, "Zone_centroid")
    OUTPUT_FILE = "data/zone_centroids.csv"
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    
    # Convert shapefiles
    convert_zone_centroids(
        centroid_shapefile=CENTROID_SHAPEFILE,
        output_file=OUTPUT_FILE
    )

if __name__ == '__main__':
    main() 