"""
SUMO Integration Module for DRT Research Platform

This module provides integration with the SUMO (Simulation of Urban MObility) traffic simulation
platform. It enables the DRT simulation to use SUMO for realistic traffic simulation, vehicle
movement, and visualization.

Key features:
- SUMO network conversion and import
- Vehicle movement synchronization
- Traffic state updates
- Visualization capabilities
"""

import os
import sys
import logging
import subprocess
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union
import xml.etree.ElementTree as ET
import json
import asyncio
from datetime import datetime, timedelta

# Try to import SUMO Python modules
try:
    # Add SUMO_HOME to path if it exists
    if 'SUMO_HOME' in os.environ:
        sys.path.append(os.path.join(os.environ['SUMO_HOME'], 'tools'))
    import traci
    import sumolib
    from sumolib import checkBinary
    SUMO_AVAILABLE = True
except ImportError:
    SUMO_AVAILABLE = False
    logging.warning("SUMO Python modules not found. SUMO integration will not be available.")

from drt_sim.config.config import NetworkConfig, VehicleConfig
from drt_sim.models.location import Location
from drt_sim.models.vehicle import Vehicle
from drt_sim.models.state import SimulationStatus

logger = logging.getLogger(__name__)

class SUMOIntegrationError(Exception):
    """Exception raised for errors in the SUMO integration."""
    pass

class SUMOConfig:
    """Configuration for SUMO integration."""
    
    def __init__(
        self,
        enabled: bool = False,
        sumo_binary: str = "sumo-gui",  # Use "sumo" for headless mode
        network_file: Optional[str] = None,
        route_file: Optional[str] = None,
        additional_files: List[str] = None,
        gui_settings_file: Optional[str] = None,
        step_length: float = 1.0,
        begin_time: float = 0.0,
        end_time: float = 86400.0,
        use_geo_coordinates: bool = True,
        port: int = 8813,
        seed: int = 42,
        auto_convert_network: bool = True,
        visualization: bool = True,
        custom_params: Dict[str, Any] = None
    ):
        """
        Initialize SUMO configuration.
        
        Args:
            enabled: Whether SUMO integration is enabled
            sumo_binary: SUMO binary to use ("sumo" or "sumo-gui")
            network_file: Path to SUMO network file (.net.xml)
            route_file: Path to SUMO route file (.rou.xml)
            additional_files: List of additional SUMO files
            gui_settings_file: Path to SUMO GUI settings file
            step_length: SUMO simulation step length in seconds
            begin_time: SUMO simulation begin time in seconds
            end_time: SUMO simulation end time in seconds
            use_geo_coordinates: Whether to use geo-coordinates
            port: TraCI port for SUMO connection
            seed: Random seed for SUMO
            auto_convert_network: Whether to automatically convert network
            visualization: Whether to enable visualization
            custom_params: Additional custom parameters
        """
        self.enabled = enabled
        self.sumo_binary = sumo_binary
        self.network_file = network_file
        self.route_file = route_file
        self.additional_files = additional_files or []
        self.gui_settings_file = gui_settings_file
        self.step_length = step_length
        self.begin_time = begin_time
        self.end_time = end_time
        self.use_geo_coordinates = use_geo_coordinates
        self.port = port
        self.seed = seed
        self.auto_convert_network = auto_convert_network
        self.visualization = visualization
        self.custom_params = custom_params or {}
        
        # Validate configuration
        if self.enabled:
            self._validate_config()
    
    def _validate_config(self):
        """Validate SUMO configuration."""
        if not SUMO_AVAILABLE:
            raise SUMOIntegrationError("SUMO Python modules not available. Cannot enable SUMO integration.")
        
        if self.network_file and not os.path.exists(self.network_file):
            raise SUMOIntegrationError(f"SUMO network file not found: {self.network_file}")
        
        if self.route_file and not os.path.exists(self.route_file):
            raise SUMOIntegrationError(f"SUMO route file not found: {self.route_file}")
        
        for file in self.additional_files:
            if not os.path.exists(file):
                raise SUMOIntegrationError(f"SUMO additional file not found: {file}")
        
        if self.gui_settings_file and not os.path.exists(self.gui_settings_file):
            raise SUMOIntegrationError(f"SUMO GUI settings file not found: {self.gui_settings_file}")

class SUMOIntegration:
    """
    Integration with SUMO traffic simulation.
    
    This class provides methods to:
    1. Convert DRT network to SUMO network
    2. Start and control SUMO simulation
    3. Synchronize vehicle movements between DRT and SUMO
    4. Retrieve traffic information from SUMO
    5. Visualize simulation in SUMO GUI
    """
    
    def __init__(self, config: SUMOConfig):
        """
        Initialize SUMO integration.
        
        Args:
            config: SUMO configuration
        """
        self.config = config
        self.sumo_process = None
        self.traci_connection = None
        self.vehicle_mapping = {}  # Maps DRT vehicle IDs to SUMO vehicle IDs
        self.network = None
        self.initialized = False
        self.running = False
        self.temp_files = []  # Track temporary files to clean up
        
        if not self.config.enabled:
            logger.info("SUMO integration is disabled.")
            return
        
        if not SUMO_AVAILABLE:
            raise SUMOIntegrationError("SUMO Python modules not available. Cannot initialize SUMO integration.")
        
        # Initialize SUMO network
        if self.config.network_file:
            try:
                self.network = sumolib.net.readNet(self.config.network_file)
                logger.info(f"Loaded SUMO network from {self.config.network_file}")
            except Exception as e:
                raise SUMOIntegrationError(f"Failed to load SUMO network: {str(e)}")
    
    async def initialize(self, drt_network_config: NetworkConfig = None):
        """
        Initialize SUMO integration.
        
        Args:
            drt_network_config: DRT network configuration (for auto-conversion)
        """
        if not self.config.enabled:
            return
        
        try:
            # Auto-convert network if needed
            if self.config.auto_convert_network and drt_network_config and not self.config.network_file:
                network_file = await self.convert_network(drt_network_config)
                self.config.network_file = network_file
                self.network = sumolib.net.readNet(network_file)
            
            # Create route file if it doesn't exist
            if not self.config.route_file:
                route_file = self._create_empty_route_file()
                self.config.route_file = route_file
            
            self.initialized = True
            logger.info("SUMO integration initialized successfully.")
        except Exception as e:
            raise SUMOIntegrationError(f"Failed to initialize SUMO integration: {str(e)}")
    
    async def start(self):
        """Start SUMO simulation."""
        if not self.config.enabled or not self.initialized:
            return
        
        try:
            # Determine SUMO binary
            sumo_binary = checkBinary(self.config.sumo_binary)
            
            # Build command
            cmd = [
                sumo_binary,
                "-n", self.config.network_file,
                "--remote-port", str(self.config.port),
                "--step-length", str(self.config.step_length),
                "--begin", str(self.config.begin_time),
                "--seed", str(self.config.seed),
                "--no-warnings", "true"
            ]
            
            if self.config.route_file:
                cmd.extend(["-r", self.config.route_file])
            
            for file in self.config.additional_files:
                cmd.extend(["-a", file])
            
            if self.config.gui_settings_file:
                cmd.extend(["--gui-settings-file", self.config.gui_settings_file])
            
            # Start SUMO as a subprocess
            logger.info(f"Starting SUMO with command: {' '.join(cmd)}")
            self.sumo_process = subprocess.Popen(cmd)
            
            # Connect to SUMO via TraCI
            await asyncio.sleep(1)  # Give SUMO time to start
            traci.init(self.config.port)
            self.traci_connection = traci.getConnection()
            
            self.running = True
            logger.info("SUMO simulation started successfully.")
        except Exception as e:
            if self.sumo_process:
                self.sumo_process.terminate()
                self.sumo_process = None
            raise SUMOIntegrationError(f"Failed to start SUMO simulation: {str(e)}")
    
    async def stop(self):
        """Stop SUMO simulation."""
        if not self.running:
            return
        
        try:
            if self.traci_connection:
                self.traci_connection.close()
                self.traci_connection = None
            
            if self.sumo_process:
                self.sumo_process.terminate()
                self.sumo_process = None
            
            self.running = False
            logger.info("SUMO simulation stopped.")
            
            # Clean up temporary files
            for file in self.temp_files:
                try:
                    os.remove(file)
                except:
                    pass
            self.temp_files = []
        except Exception as e:
            logger.error(f"Error stopping SUMO simulation: {str(e)}")
    
    async def step(self, time_step: float):
        """
        Advance SUMO simulation by one step.
        
        Args:
            time_step: Time step in seconds
        """
        if not self.running:
            return
        
        try:
            self.traci_connection.simulationStep()
        except Exception as e:
            logger.error(f"Error during SUMO simulation step: {str(e)}")
            await self.stop()
            raise SUMOIntegrationError(f"SUMO simulation failed: {str(e)}")
    
    async def add_vehicle(self, vehicle: Vehicle, route: List[Location] = None):
        """
        Add a vehicle to SUMO simulation.
        
        Args:
            vehicle: DRT vehicle
            route: Initial route for the vehicle
        """
        if not self.running:
            return
        
        try:
            # Create SUMO vehicle ID
            sumo_vehicle_id = f"drt_{vehicle.id}"
            
            # Convert route to SUMO edges
            if route:
                edge_ids = await self._convert_route_to_edges(route)
                if not edge_ids:
                    logger.warning(f"Could not convert route to SUMO edges for vehicle {vehicle.id}")
                    return
                
                # Add route to SUMO
                route_id = f"route_{vehicle.id}"
                self.traci_connection.route.add(route_id, edge_ids)
                
                # Add vehicle to SUMO
                self.traci_connection.vehicle.add(
                    sumo_vehicle_id,
                    route_id,
                    typeID="drt_vehicle",
                    depart="now",
                    departPos=0,
                    departSpeed=0,
                    departLane="best"
                )
                
                # Set vehicle color (e.g., blue for DRT vehicles)
                self.traci_connection.vehicle.setColor(sumo_vehicle_id, (0, 0, 255, 255))
                
                # Store vehicle mapping
                self.vehicle_mapping[vehicle.id] = sumo_vehicle_id
                logger.info(f"Added vehicle {vehicle.id} to SUMO simulation")
            else:
                logger.warning(f"No route provided for vehicle {vehicle.id}, not adding to SUMO")
        except Exception as e:
            logger.error(f"Error adding vehicle to SUMO: {str(e)}")
    
    async def update_vehicle_position(self, vehicle_id: str, position: Location):
        """
        Update vehicle position in SUMO.
        
        Args:
            vehicle_id: DRT vehicle ID
            position: New vehicle position
        """
        if not self.running:
            return
        
        try:
            sumo_vehicle_id = self.vehicle_mapping.get(vehicle_id)
            if not sumo_vehicle_id:
                logger.warning(f"Vehicle {vehicle_id} not found in SUMO")
                return
            
            # Convert position to SUMO coordinates
            x, y = position.lon, position.lat
            if self.config.use_geo_coordinates:
                lon, lat = x, y
                x, y = self.network.convertLonLat2XY(lon, lat)
            
            # Find closest edge
            edge = self.network.getClosestEdge((x, y))
            if edge:
                # Move vehicle to position
                self.traci_connection.vehicle.moveToXY(
                    sumo_vehicle_id,
                    edge.getID(),
                    0,  # lane index
                    x, y,
                    angle=-1,  # auto-determine angle
                    keepRoute=2  # auto-determine route
                )
            else:
                logger.warning(f"Could not find edge for position {position} for vehicle {vehicle_id}")
        except Exception as e:
            logger.error(f"Error updating vehicle position in SUMO: {str(e)}")
    
    async def update_vehicle_route(self, vehicle_id: str, route: List[Location]):
        """
        Update vehicle route in SUMO.
        
        Args:
            vehicle_id: DRT vehicle ID
            route: New vehicle route
        """
        if not self.running:
            return
        
        try:
            sumo_vehicle_id = self.vehicle_mapping.get(vehicle_id)
            if not sumo_vehicle_id:
                logger.warning(f"Vehicle {vehicle_id} not found in SUMO")
                return
            
            # Convert route to SUMO edges
            edge_ids = await self._convert_route_to_edges(route)
            if not edge_ids:
                logger.warning(f"Could not convert route to SUMO edges for vehicle {vehicle_id}")
                return
            
            # Update vehicle route
            self.traci_connection.vehicle.setRoute(sumo_vehicle_id, edge_ids)
            logger.debug(f"Updated route for vehicle {vehicle_id} in SUMO")
        except Exception as e:
            logger.error(f"Error updating vehicle route in SUMO: {str(e)}")
    
    async def get_traffic_info(self, edge_id: str) -> Dict[str, Any]:
        """
        Get traffic information for an edge.
        
        Args:
            edge_id: SUMO edge ID
            
        Returns:
            Dictionary with traffic information
        """
        if not self.running:
            return {}
        
        try:
            # Get edge information
            mean_speed = self.traci_connection.edge.getLastStepMeanSpeed(edge_id)
            travel_time = self.traci_connection.edge.getTraveltime(edge_id)
            vehicle_count = self.traci_connection.edge.getLastStepVehicleNumber(edge_id)
            occupancy = self.traci_connection.edge.getLastStepOccupancy(edge_id)
            
            return {
                "edge_id": edge_id,
                "mean_speed": mean_speed,
                "travel_time": travel_time,
                "vehicle_count": vehicle_count,
                "occupancy": occupancy
            }
        except Exception as e:
            logger.error(f"Error getting traffic information from SUMO: {str(e)}")
            return {}
    
    async def convert_network(self, drt_network_config: NetworkConfig) -> str:
        """
        Convert DRT network to SUMO network.
        
        Args:
            drt_network_config: DRT network configuration
            
        Returns:
            Path to generated SUMO network file
        """
        if not SUMO_AVAILABLE:
            raise SUMOIntegrationError("SUMO Python modules not available. Cannot convert network.")
        
        try:
            # Create temporary directory for network files
            temp_dir = tempfile.mkdtemp()
            
            # Create node file
            node_file = os.path.join(temp_dir, "nodes.nod.xml")
            await self._create_node_file(drt_network_config, node_file)
            
            # Create edge file
            edge_file = os.path.join(temp_dir, "edges.edg.xml")
            await self._create_edge_file(drt_network_config, edge_file)
            
            # Create network file using NETCONVERT
            net_file = os.path.join(temp_dir, "network.net.xml")
            netconvert_cmd = [
                "netconvert",
                "--node-files", node_file,
                "--edge-files", edge_file,
                "--output-file", net_file,
                "--geometry.remove", "true",
                "--junctions.corner-detail", "0",
                "--junctions.internal-link-detail", "0"
            ]
            
            if self.config.use_geo_coordinates:
                netconvert_cmd.extend(["--proj.utm", "true"])
            
            subprocess.run(netconvert_cmd, check=True)
            
            # Track temporary files for cleanup
            self.temp_files.extend([node_file, edge_file])
            
            logger.info(f"Converted DRT network to SUMO network: {net_file}")
            return net_file
        except Exception as e:
            raise SUMOIntegrationError(f"Failed to convert network: {str(e)}")
    
    async def _create_node_file(self, drt_network_config: NetworkConfig, output_file: str):
        """
        Create SUMO node file from DRT network.
        
        Args:
            drt_network_config: DRT network configuration
            output_file: Output file path
        """
        # This is a simplified implementation - in a real system, you would
        # extract nodes from the DRT network graph
        
        # Create XML structure
        root = ET.Element("nodes")
        
        # Add some example nodes
        # In a real implementation, you would iterate through the DRT network nodes
        node1 = ET.SubElement(root, "node")
        node1.set("id", "1")
        node1.set("x", "0")
        node1.set("y", "0")
        
        node2 = ET.SubElement(root, "node")
        node2.set("id", "2")
        node2.set("x", "100")
        node2.set("y", "0")
        
        node3 = ET.SubElement(root, "node")
        node3.set("id", "3")
        node3.set("x", "100")
        node3.set("y", "100")
        
        node4 = ET.SubElement(root, "node")
        node4.set("id", "4")
        node4.set("x", "0")
        node4.set("y", "100")
        
        # Write to file
        tree = ET.ElementTree(root)
        tree.write(output_file, encoding="utf-8", xml_declaration=True)
    
    async def _create_edge_file(self, drt_network_config: NetworkConfig, output_file: str):
        """
        Create SUMO edge file from DRT network.
        
        Args:
            drt_network_config: DRT network configuration
            output_file: Output file path
        """
        # This is a simplified implementation - in a real system, you would
        # extract edges from the DRT network graph
        
        # Create XML structure
        root = ET.Element("edges")
        
        # Add some example edges
        # In a real implementation, you would iterate through the DRT network edges
        edge1 = ET.SubElement(root, "edge")
        edge1.set("id", "1to2")
        edge1.set("from", "1")
        edge1.set("to", "2")
        edge1.set("numLanes", "1")
        edge1.set("speed", "13.89")
        
        edge2 = ET.SubElement(root, "edge")
        edge2.set("id", "2to3")
        edge2.set("from", "2")
        edge2.set("to", "3")
        edge2.set("numLanes", "1")
        edge2.set("speed", "13.89")
        
        edge3 = ET.SubElement(root, "edge")
        edge3.set("id", "3to4")
        edge3.set("from", "3")
        edge3.set("to", "4")
        edge3.set("numLanes", "1")
        edge3.set("speed", "13.89")
        
        edge4 = ET.SubElement(root, "edge")
        edge4.set("id", "4to1")
        edge4.set("from", "4")
        edge4.set("to", "1")
        edge4.set("numLanes", "1")
        edge4.set("speed", "13.89")
        
        # Write to file
        tree = ET.ElementTree(root)
        tree.write(output_file, encoding="utf-8", xml_declaration=True)
    
    def _create_empty_route_file(self) -> str:
        """
        Create an empty SUMO route file.
        
        Returns:
            Path to created route file
        """
        # Create temporary file
        fd, route_file = tempfile.mkstemp(suffix=".rou.xml")
        os.close(fd)
        
        # Create XML structure
        root = ET.Element("routes")
        
        # Add vehicle type for DRT vehicles
        vtype = ET.SubElement(root, "vType")
        vtype.set("id", "drt_vehicle")
        vtype.set("accel", "2.6")
        vtype.set("decel", "4.5")
        vtype.set("sigma", "0.5")
        vtype.set("length", "5.0")
        vtype.set("minGap", "2.5")
        vtype.set("maxSpeed", "13.89")
        vtype.set("color", "0,0,255")
        
        # Write to file
        tree = ET.ElementTree(root)
        tree.write(route_file, encoding="utf-8", xml_declaration=True)
        
        # Track for cleanup
        self.temp_files.append(route_file)
        
        return route_file
    
    async def _convert_route_to_edges(self, route: List[Location]) -> List[str]:
        """
        Convert a route of locations to SUMO edge IDs.
        
        Args:
            route: List of locations
            
        Returns:
            List of SUMO edge IDs
        """
        if not self.network or not route:
            return []
        
        edge_ids = []
        for location in route:
            # Convert to SUMO coordinates
            x, y = location.lon, location.lat
            if self.config.use_geo_coordinates:
                lon, lat = x, y
                x, y = self.network.convertLonLat2XY(lon, lat)
            
            # Find closest edge
            edge = self.network.getClosestEdge((x, y))
            if edge:
                edge_ids.append(edge.getID())
        
        # Remove duplicates while preserving order
        unique_edge_ids = []
        for edge_id in edge_ids:
            if edge_id not in unique_edge_ids:
                unique_edge_ids.append(edge_id)
        
        return unique_edge_ids