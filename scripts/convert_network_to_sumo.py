#!/usr/bin/env python3
"""
Network Converter for SUMO Integration

This script converts network files from various formats to SUMO format.
Supported input formats:
- GraphML
- GeoJSON
- OSM (OpenStreetMap)

Usage:
    python convert_network_to_sumo.py --input <input_file> --output <output_dir> [--format <format>]
"""

import os
import sys
import argparse
import logging
import tempfile
import subprocess
from pathlib import Path
import networkx as nx
import xml.etree.ElementTree as ET
import json
import geopandas as gpd
from shapely.geometry import Point, LineString

# Try to import SUMO Python modules
try:
    # Add SUMO_HOME to path if it exists
    if 'SUMO_HOME' in os.environ:
        sys.path.append(os.path.join(os.environ['SUMO_HOME'], 'tools'))
    import sumolib
    SUMO_AVAILABLE = True
except ImportError:
    SUMO_AVAILABLE = False
    logging.warning("SUMO Python modules not found. Some functionality may be limited.")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class NetworkConverter:
    """Converts network files to SUMO format."""
    
    def __init__(self, input_file: str, output_dir: str, format: str = None):
        """
        Initialize the network converter.
        
        Args:
            input_file: Path to input network file
            output_dir: Directory to save output files
            format: Input file format (auto-detected if None)
        """
        self.input_file = Path(input_file)
        self.output_dir = Path(output_dir)
        self.format = format or self._detect_format()
        
        # Create output directory if it doesn't exist
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Check if SUMO is available
        if not SUMO_AVAILABLE:
            logger.warning("SUMO Python modules not available. Some functionality may be limited.")
    
    def _detect_format(self) -> str:
        """
        Detect input file format based on file extension.
        
        Returns:
            Detected format
        """
        suffix = self.input_file.suffix.lower()
        if suffix == '.graphml':
            return 'graphml'
        elif suffix in ['.geojson', '.json']:
            return 'geojson'
        elif suffix in ['.osm', '.xml']:
            return 'osm'
        else:
            raise ValueError(f"Unsupported file format: {suffix}")
    
    def convert(self) -> str:
        """
        Convert network file to SUMO format.
        
        Returns:
            Path to generated SUMO network file
        """
        logger.info(f"Converting {self.input_file} to SUMO format")
        
        if self.format == 'osm':
            return self._convert_osm()
        elif self.format == 'graphml':
            return self._convert_graphml()
        elif self.format == 'geojson':
            return self._convert_geojson()
        else:
            raise ValueError(f"Unsupported format: {self.format}")
    
    def _convert_osm(self) -> str:
        """
        Convert OSM file to SUMO format using SUMO's netconvert.
        
        Returns:
            Path to generated SUMO network file
        """
        logger.info("Converting OSM file to SUMO format")
        
        # Output file path
        output_file = self.output_dir / f"{self.input_file.stem}.net.xml"
        
        # Build netconvert command
        cmd = [
            "netconvert",
            "--osm-files", str(self.input_file),
            "--output-file", str(output_file),
            "--geometry.remove", "true",
            "--roundabouts.guess", "true",
            "--ramps.guess", "true",
            "--junctions.join", "true",
            "--tls.guess", "true",
            "--tls.join", "true",
            "--edges.join", "true"
        ]
        
        # Run netconvert
        try:
            logger.info(f"Running command: {' '.join(cmd)}")
            subprocess.run(cmd, check=True)
            logger.info(f"Successfully converted OSM file to SUMO format: {output_file}")
            return str(output_file)
        except subprocess.CalledProcessError as e:
            logger.error(f"Error converting OSM file: {e}")
            raise
    
    def _convert_graphml(self) -> str:
        """
        Convert GraphML file to SUMO format.
        
        Returns:
            Path to generated SUMO network file
        """
        logger.info("Converting GraphML file to SUMO format")
        
        try:
            # Load GraphML file
            G = nx.read_graphml(self.input_file)
            
            # Create node and edge files
            node_file = self.output_dir / f"{self.input_file.stem}.nod.xml"
            edge_file = self.output_dir / f"{self.input_file.stem}.edg.xml"
            
            # Create node file
            self._create_node_file_from_graph(G, node_file)
            
            # Create edge file
            self._create_edge_file_from_graph(G, edge_file)
            
            # Create network file using netconvert
            output_file = self.output_dir / f"{self.input_file.stem}.net.xml"
            self._run_netconvert(node_file, edge_file, output_file)
            
            return str(output_file)
        except Exception as e:
            logger.error(f"Error converting GraphML file: {e}")
            raise
    
    def _convert_geojson(self) -> str:
        """
        Convert GeoJSON file to SUMO format.
        
        Returns:
            Path to generated SUMO network file
        """
        logger.info("Converting GeoJSON file to SUMO format")
        
        try:
            # Load GeoJSON file
            gdf = gpd.read_file(self.input_file)
            
            # Create node and edge files
            node_file = self.output_dir / f"{self.input_file.stem}.nod.xml"
            edge_file = self.output_dir / f"{self.input_file.stem}.edg.xml"
            
            # Create node file
            self._create_node_file_from_geojson(gdf, node_file)
            
            # Create edge file
            self._create_edge_file_from_geojson(gdf, edge_file)
            
            # Create network file using netconvert
            output_file = self.output_dir / f"{self.input_file.stem}.net.xml"
            self._run_netconvert(node_file, edge_file, output_file)
            
            return str(output_file)
        except Exception as e:
            logger.error(f"Error converting GeoJSON file: {e}")
            raise
    
    def _create_node_file_from_graph(self, G: nx.Graph, output_file: Path) -> None:
        """
        Create SUMO node file from NetworkX graph.
        
        Args:
            G: NetworkX graph
            output_file: Output file path
        """
        # Create XML structure
        root = ET.Element("nodes")
        
        # Add nodes
        for node_id, data in G.nodes(data=True):
            node = ET.SubElement(root, "node")
            node.set("id", str(node_id))
            
            # Get coordinates
            x = data.get('x', data.get('lon', 0))
            y = data.get('y', data.get('lat', 0))
            
            node.set("x", str(x))
            node.set("y", str(y))
            
            # Add optional attributes
            if 'type' in data:
                node.set("type", str(data['type']))
        
        # Write to file
        tree = ET.ElementTree(root)
        tree.write(output_file, encoding="utf-8", xml_declaration=True)
        logger.info(f"Created node file: {output_file}")
    
    def _create_edge_file_from_graph(self, G: nx.Graph, output_file: Path) -> None:
        """
        Create SUMO edge file from NetworkX graph.
        
        Args:
            G: NetworkX graph
            output_file: Output file path
        """
        # Create XML structure
        root = ET.Element("edges")
        
        # Add edges
        for u, v, data in G.edges(data=True):
            edge = ET.SubElement(root, "edge")
            edge.set("id", f"{u}to{v}")
            edge.set("from", str(u))
            edge.set("to", str(v))
            
            # Add optional attributes
            if 'weight' in data:
                edge.set("speed", str(13.89))  # Default 50 km/h
                edge.set("length", str(data['weight']))
            if 'lanes' in data:
                edge.set("numLanes", str(data['lanes']))
            if 'speed' in data:
                edge.set("speed", str(data['speed']))
            if 'type' in data:
                edge.set("type", str(data['type']))
            
            # Default values if not provided
            if 'speed' not in data and 'weight' not in data:
                edge.set("speed", "13.89")  # Default 50 km/h
                edge.set("length", "100")  # Default length
            if 'lanes' not in data:
                edge.set("numLanes", "1")  # Default 1 lane
        
        # Write to file
        tree = ET.ElementTree(root)
        tree.write(output_file, encoding="utf-8", xml_declaration=True)
        logger.info(f"Created edge file: {output_file}")
    
    def _create_node_file_from_geojson(self, gdf: gpd.GeoDataFrame, output_file: Path) -> None:
        """
        Create SUMO node file from GeoDataFrame.
        
        Args:
            gdf: GeoDataFrame
            output_file: Output file path
        """
        # Create XML structure
        root = ET.Element("nodes")
        
        # Extract nodes from points and linestrings
        nodes = {}
        
        # Process point features
        point_features = gdf[gdf.geometry.type == 'Point']
        for idx, row in point_features.iterrows():
            node_id = str(row.get('id', idx))
            nodes[node_id] = (row.geometry.x, row.geometry.y)
        
        # Process linestring features to extract endpoints
        line_features = gdf[gdf.geometry.type == 'LineString']
        for idx, row in line_features.iterrows():
            coords = list(row.geometry.coords)
            if coords:
                start_id = f"node_{idx}_start"
                end_id = f"node_{idx}_end"
                nodes[start_id] = (coords[0][0], coords[0][1])
                nodes[end_id] = (coords[-1][0], coords[-1][1])
        
        # Add nodes to XML
        for node_id, (x, y) in nodes.items():
            node = ET.SubElement(root, "node")
            node.set("id", node_id)
            node.set("x", str(x))
            node.set("y", str(y))
        
        # Write to file
        tree = ET.ElementTree(root)
        tree.write(output_file, encoding="utf-8", xml_declaration=True)
        logger.info(f"Created node file: {output_file}")
    
    def _create_edge_file_from_geojson(self, gdf: gpd.GeoDataFrame, output_file: Path) -> None:
        """
        Create SUMO edge file from GeoDataFrame.
        
        Args:
            gdf: GeoDataFrame
            output_file: Output file path
        """
        # Create XML structure
        root = ET.Element("edges")
        
        # Process linestring features
        line_features = gdf[gdf.geometry.type == 'LineString']
        for idx, row in line_features.iterrows():
            edge = ET.SubElement(root, "edge")
            
            # Set edge attributes
            edge_id = str(row.get('id', f"edge_{idx}"))
            edge.set("id", edge_id)
            
            # Set from/to nodes
            start_id = f"node_{idx}_start"
            end_id = f"node_{idx}_end"
            edge.set("from", start_id)
            edge.set("to", end_id)
            
            # Set optional attributes
            if 'lanes' in row:
                edge.set("numLanes", str(row['lanes']))
            else:
                edge.set("numLanes", "1")
                
            if 'speed' in row:
                edge.set("speed", str(row['speed']))
            else:
                edge.set("speed", "13.89")  # Default 50 km/h
                
            if 'length' in row:
                edge.set("length", str(row['length']))
            else:
                # Calculate length from geometry
                edge.set("length", str(row.geometry.length))
        
        # Write to file
        tree = ET.ElementTree(root)
        tree.write(output_file, encoding="utf-8", xml_declaration=True)
        logger.info(f"Created edge file: {output_file}")
    
    def _run_netconvert(self, node_file: Path, edge_file: Path, output_file: Path) -> None:
        """
        Run SUMO's netconvert tool to create a network file.
        
        Args:
            node_file: Path to node file
            edge_file: Path to edge file
            output_file: Path to output network file
        """
        if not SUMO_AVAILABLE:
            logger.warning("SUMO Python modules not available. Using direct netconvert command.")
        
        # Try to find netconvert in SUMO_HOME or PATH
        netconvert_cmd = "netconvert"
        if 'SUMO_HOME' in os.environ:
            netconvert_path = os.path.join(os.environ['SUMO_HOME'], 'bin', 'netconvert')
            if os.path.exists(netconvert_path):
                netconvert_cmd = netconvert_path
        
        try:
            cmd = [
                netconvert_cmd,
                "--node-files", str(node_file),
                "--edge-files", str(edge_file),
                "--output-file", str(output_file),
                "--geometry.remove", "true"
            ]
            logger.info(f"Running command: {' '.join(cmd)}")
            subprocess.run(cmd, check=True, capture_output=True, text=True)
            logger.info(f"Network file created at {output_file}")
        except subprocess.CalledProcessError as e:
            logger.error(f"Error running netconvert: {e.stderr}")
            raise RuntimeError(f"Error running netconvert: {e.stderr}")
        except FileNotFoundError:
            logger.error("netconvert command not found. Please ensure SUMO is installed and in your PATH.")
            raise RuntimeError("netconvert command not found. Please ensure SUMO is installed and in your PATH.")

def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Convert network files to SUMO format")
    parser.add_argument("--input", "-i", required=True, help="Input network file")
    parser.add_argument("--output", "-o", required=True, help="Output directory")
    parser.add_argument("--format", "-f", choices=["graphml", "geojson", "osm"], help="Input file format (auto-detected if not specified)")
    
    args = parser.parse_args()
    
    try:
        converter = NetworkConverter(args.input, args.output, args.format)
        output_file = converter.convert()
        logger.info(f"Conversion completed successfully. Output file: {output_file}")
    except Exception as e:
        logger.error(f"Error during conversion: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 