from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Set
import logging
from functools import lru_cache
import networkx as nx
import geopandas as gpd
from pyproj import Transformer
import numpy as np
from scipy.spatial import cKDTree
import threading
from concurrent.futures import ThreadPoolExecutor
import asyncio
import math
from drt_sim.config.config import NetworkConfig, NetworkInfo
from drt_sim.models.location import Location
from drt_sim.core.logging_config import setup_logger

logger = setup_logger(__name__)
class NetworkManager:
    """Enhanced network manager with optimized path finding and caching"""
    
    def __init__(self, config: NetworkConfig):
        """Initialize NetworkManager with configuration"""
        self.config = config
        self._drive_network: Optional[nx.Graph] = None
        self._walk_network: Optional[nx.Graph] = None
        self._transformer: Optional[Transformer] = None
        self._info: Dict[str, NetworkInfo] = {}
        self.spatial_indexes: Dict[str, cKDTree] = {}
        self.node_coords: Dict[str, Dict[int, Tuple[float, float]]] = {}
        self.node_lists: Dict[str, List[int]] = {}
        
        # Initialize components
        self._init_executor()
        self._init_caches()
        
        # Initialize networks
        self.initialize_networks()
        
        # Initialize spatial indexes after networks are loaded
        self._init_spatial_indexes()
        
        # Path finding thresholds
        self.path_thresholds = {
            'drive': {
                'distance_ratio': 2.5,
                'max_detour': 3000,
                'alternate_nodes': 3,
                'search_radius': 500
            },
            'walk': {
                'distance_ratio': 1.4,
                'max_detour': 500,
                'alternate_nodes': 5,
                'search_radius': 200
            }
        }
        
        logger.info("NetworkManager initialized successfully")
        
    def _init_executor(self):
        """Initialize thread pool executor"""
        self.executor = ThreadPoolExecutor(max_workers=10)
        
    def _init_caches(self):
        """Initialize thread-safe caches"""
        self.cache_lock = threading.Lock()
        self.path_cache = {}
        self.distance_cache = {}
        self.nearest_node_cache = {}
        
        # Initialize cache statistics
        self.cache_hits = 0
        self.cache_misses = 0
        
        # Set up LRU caches
        self.get_cached_path = lru_cache(10000)(self._calculate_path)
        self._get_nearest_nodes = lru_cache(1000)(self._find_nearest_node_internal)
        
    def initialize_networks(self):
        """Load and initialize networks from configuration"""
        logger.info("Initializing networks...")
        
        if self.config.network_file:
            self._drive_network = self._load_network(
                self.config.network_file,
                "drive"
            )
            logger.info(f"Drive network loaded with {self._drive_network.number_of_nodes()} nodes")
            
        if self.config.walk_network_file:
            self._walk_network = self._load_network(
                self.config.walk_network_file,
                "walk"
            )
            logger.info(f"Walk network loaded with {self._walk_network.number_of_nodes()} nodes")
    
    def _init_spatial_indexes(self):
        """Initialize spatial indexes for fast coordinate lookups"""
        logger.info("Initializing spatial indexes...")
        
        for network_type in ['drive', 'walk']:
            network = getattr(self, f'_{network_type}_network')
            if network is not None:
                # Create coordinate arrays
                coords = np.array([
                    (data['y'], data['x'])
                    for _, data in network.nodes(data=True)
                ])
                
                # Build KD-tree
                self.spatial_indexes[network_type] = cKDTree(coords)
                
                # Store node coordinates and list
                self.node_coords[network_type] = {
                    node: (data['y'], data['x'])
                    for node, data in network.nodes(data=True)
                }
                self.node_lists[network_type] = list(network.nodes())
                
                logger.info(f"Spatial index created for {network_type} network")
    
    def _init_service_area(self):
        """Initialize service area polygon if provided in config"""
        if self.config.service_area_polygon:
            from shapely.geometry import Polygon
            self.service_area = Polygon(self.config.service_area_polygon)
        else:
            self.service_area = None

    def is_point_in_service_area(self, lat: float, lon: float) -> bool:
        """Check if a point is within the service area"""
        if self.service_area is None:
            return True  # If no service area defined, accept all points
        
        from shapely.geometry import Point
        point = Point(lon, lat)  # Note: Shapely uses (lon, lat) order
        return self.service_area.contains(point)

    def filter_nodes_by_service_area(self, nodes: List[int], network_type: str) -> List[int]:
        """Filter nodes to only those within service area"""
        if self.service_area is None:
            return nodes
            
        return [
            node for node in nodes
            if self.is_point_in_service_area(*self.get_node_coordinates(node, network_type))
        ]
                
    def _load_network(self, file_path: Union[str, Path], network_type: str) -> nx.Graph:
        """Load network from file and process it"""
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"Network file not found: {file_path}")
            
        logger.info(f"Loading {network_type} network from {file_path}")
        
        try:
            # Load the graph based on file type
            if file_path.suffix == '.graphml':
                G = nx.read_graphml(file_path)
            elif file_path.suffix == '.geojson':
                G = self._load_from_geojson(file_path)
            else:
                raise ValueError(f"Unsupported file format: {file_path.suffix}")
                
            # Process the network
            G = self._process_network(G, network_type)
            
            # Store network info
            self._info[network_type] = self._create_network_info(G, file_path.stem)
            
            return G
            
        except Exception as e:
            logger.error(f"Error loading network: {str(e)}")
            raise
            
    def _load_from_geojson(self, file_path: Path) -> nx.Graph:
        """Load network from GeoJSON file"""
        gdf = gpd.read_file(file_path)
        G = nx.Graph()
        
        for idx, row in gdf.iterrows():
            if row.geometry.type == 'Point':
                G.add_node(idx, x=row.geometry.x, y=row.geometry.y)
            elif row.geometry.type == 'LineString':
                coords = list(row.geometry.coords)
                G.add_edge(coords[0], coords[-1], length=row.geometry.length)
                
        return G
        
    def _process_network(self, G: nx.Graph, network_type: str) -> nx.Graph:
        """Process loaded network to ensure required attributes"""
        logger.info(f"Processing {network_type} network...")
        
        speed = (self.config.walking_speed if network_type == 'walk' 
                else self.config.driving_speed)
                
        for u, v, data in G.edges(data=True):
            # Ensure length attribute exists
            if 'length' not in data:
                data['length'] = self._calculate_length(G, u, v)
                
            # Add travel time
            data['travel_time'] = data['length'] / speed
            
        return G
        
    def _calculate_length(self, G: nx.Graph, u: int, v: int) -> float:
        """Calculate length between two nodes"""
        if self.config.coordinate_system == "EPSG:4326":
            if not self._transformer:
                self._transformer = Transformer.from_crs(
                    "EPSG:4326",
                    "EPSG:3857",
                    always_xy=True
                )
                
            start = self._transformer.transform(G.nodes[u]['y'], G.nodes[u]['x'])
            end = self._transformer.transform(G.nodes[v]['y'], G.nodes[v]['x'])
            return math.sqrt((end[0] - start[0])**2 + (end[1] - start[1])**2)
        else:
            return math.sqrt(
                (G.nodes[v]['x'] - G.nodes[u]['x'])**2 + 
                (G.nodes[v]['y'] - G.nodes[u]['y'])**2
            )
            
    def _create_network_info(self, G: nx.Graph, name: str) -> NetworkInfo:
        """Create network info for loaded network"""
        nodes = G.nodes(data=True)
        lats = [data['y'] for _, data in nodes if 'y' in data]
        lons = [data['x'] for _, data in nodes if 'x' in data]
        
        return NetworkInfo(
            name=name,
            node_count=G.number_of_nodes(),
            edge_count=G.number_of_edges(),
            bbox=(min(lons), min(lats), max(lons), max(lats)),
            crs=self.config.coordinate_system
        )
            
    async def _find_alternate_nodes(self, lat: float, lon: float, network_type: str) -> List[int]:
        """Find multiple nearest nodes with mode-specific parameters"""
        thresholds = self.path_thresholds[network_type]
        tree = self.spatial_indexes[network_type]
        nodes = self.node_lists[network_type]
        
        coords = np.array([[lat, lon]])
        k = thresholds['alternate_nodes'] * 2
        
        distances, indices = tree.query(coords, k=k)
        
        # Filter nodes within search radius
        valid_indices = [
            idx for idx, dist in zip(indices[0], distances[0])
            if dist * 111000 <= thresholds['search_radius']  # Convert degrees to meters
        ]
        
        return [nodes[idx] for idx in valid_indices[:thresholds['alternate_nodes']]]
        
    def _find_nearest_node_internal(self, lat: float, lon: float, network_type: str) -> int:
        """Internal nearest node lookup using KD-tree"""
        tree = self.spatial_indexes[network_type]
        nodes = self.node_lists[network_type]
        
        coords = np.array([[lat, lon]])
        _, index = tree.query(coords, k=1)
        return nodes[index[0]]
        
    def _haversine_distance(self, lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        """Calculate great circle distance between points in meters"""
        R = 6371000  # Earth's radius in meters
        
        lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        
        a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
        c = 2 * np.arcsin(np.sqrt(a))
        return R * c
        
    async def get_nearest_node(self, location: Location, network_type: str = 'drive') -> int:
        """Asynchronous nearest node finder with caching"""
        cache_key = (location.lat, location.lon, network_type)
        
        with self.cache_lock:
            if cache_key in self.nearest_node_cache:
                self.cache_hits += 1
                return self.nearest_node_cache[cache_key]
            self.cache_misses += 1
            
        node = await asyncio.get_event_loop().run_in_executor(
            self.executor,
            self._find_nearest_node_internal,
            location.lat,
            location.lon,
            network_type
        )
        
        with self.cache_lock:
            self.nearest_node_cache[cache_key] = node
            
        return node
        
    async def calculate_distance(self, 
                               loc1: Location, 
                               loc2: Location, 
                               network_type: str = 'drive',
                               weight: str = 'distance') -> float:
        """Calculate the network distance between two locations"""
        cache_key = (loc1.lat, loc1.lon, loc2.lat, loc2.lon, network_type, weight)
        
        with self.cache_lock:
            if cache_key in self.distance_cache:
                self.cache_hits += 1
                return self.distance_cache[cache_key]
            self.cache_misses += 1
            
        _, distance = await self.get_shortest_path(loc1, loc2, network_type, weight)
        
        with self.cache_lock:
            self.distance_cache[cache_key] = distance
            
        return distance
        
    async def calculate_travel_time(self, 
                                  loc1: Location, 
                                  loc2: Location, 
                                  network_type: str = 'drive') -> float:
        """Calculate the travel time between two locations"""
        distance = await self.calculate_distance(loc1, loc2, network_type, weight='distance')
        speed = self.config.walking_speed if network_type == 'walk' else self.config.driving_speed
        return distance / speed
        
    async def find_nodes_within_radius(self, 
                                     location: Location, 
                                     radius: float, 
                                     network_type: str = 'drive') -> List[int]:
        """Find all nodes within a given radius of a location"""
        tree = self.spatial_indexes[network_type]
        nodes = self.node_lists[network_type]
        coords = np.array([[location.lat, location.lon]])
        
        indices = tree.query_ball_point(coords[0], radius / 111000)  # Convert meters to degrees
        return [nodes[idx] for idx in indices]
        
    def get_node_coordinates(self, node: int, network_type: str) -> Tuple[float, float]:
        """Get the coordinates of a node"""
        try:
            return self.node_coords[network_type][node]
        except KeyError:
            raise ValueError(f"Node {node} not found in {network_type} network")
            
    async def find_path_with_waypoints(self,
                                     start: Location,
                                     end: Location,
                                     waypoints: List[Location],
                                     network_type: str = 'drive') -> Tuple[List[int], float]:
        """Find a path that visits all waypoints in order"""
        try:
            if not waypoints:
                return await self.get_shortest_path(start, end, network_type)
                
            # Find nearest nodes for all points
            points = [start] + waypoints + [end]
            nodes = await asyncio.gather(*[
                self.get_nearest_node(point, network_type)
                for point in points
            ])
            
            # Calculate path segments in parallel
            path_segments = await asyncio.gather(*[
                self.get_shortest_path(
                    Location(lat=self.node_coords[network_type][nodes[i]][0],
                            lon=self.node_coords[network_type][nodes[i]][1]),
                    Location(lat=self.node_coords[network_type][nodes[i+1]][0],
                            lon=self.node_coords[network_type][nodes[i+1]][1]),
                    network_type
                )
                for i in range(len(nodes)-1)
            ])
            
            # Combine path segments
            complete_path = []
            total_distance = 0
            
            for path, distance in path_segments:
                if path:
                    if complete_path and complete_path[-1] == path[0]:
                        complete_path.extend(path[1:])
                    else:
                        complete_path.extend(path)
                    total_distance += distance
                    
            return complete_path, total_distance
            
        except Exception as e:
            logger.error(f"Error finding path with waypoints: {str(e)}", exc_info=True)
            return [], float('inf')
        
    async def get_shortest_path(self, 
                            start_location: Location, 
                            end_location: Location,
                            network_type: str = 'drive',
                            weight: str = 'distance') -> Tuple[List[int], float]:
        """Enhanced asynchronous shortest path calculation with debugging"""
        try:
            # Quick return for identical locations
            if (abs(start_location.lat - end_location.lat) < 1e-7 and 
                abs(start_location.lon - end_location.lon) < 1e-7):
                return [], 0.0

            # Log input parameters
            logger.debug(f"Finding path from ({start_location.lat}, {start_location.lon}) "
                        f"to ({end_location.lat}, {end_location.lon}) "
                        f"using {network_type} network")
                
            # Check cache
            cache_key = (
                start_location.lat, start_location.lon,
                end_location.lat, end_location.lon,
                network_type, weight
            )
            
            with self.cache_lock:
                if cache_key in self.path_cache:
                    logger.debug("Cache hit")
                    self.cache_hits += 1
                    return self.path_cache[cache_key]
                self.cache_misses += 1
                logger.debug("Cache miss")
                
            # Get alternate nodes
            start_nodes = await self._find_alternate_nodes(
                start_location.lat, start_location.lon, network_type
            )
            end_nodes = await self._find_alternate_nodes(
                end_location.lat, end_location.lon, network_type
            )
            
            logger.debug(f"Found {len(start_nodes)} start nodes and {len(end_nodes)} end nodes")
            
            if not start_nodes or not end_nodes:
                logger.debug(f"No valid nodes found: start_nodes={bool(start_nodes)}, end_nodes={bool(end_nodes)}")
                return [], float('inf')
                
            # Find best path among alternatives
            best_path = None
            best_distance = float('inf')
            paths_tried = 0
            paths_found = 0
            
            for start_node in start_nodes:
                for end_node in end_nodes:
                    paths_tried += 1
                    logger.debug(f"Trying path from node {start_node} to node {end_node}")
                    
                    path, distance = await asyncio.get_event_loop().run_in_executor(
                        self.executor,
                        self.get_cached_path,
                        start_node,
                        end_node,
                        network_type,
                        weight
                    )
                    
                    logger.debug(f"Path found: {bool(path)}, Distance: {distance}")
                    
                    if path:
                        paths_found += 1
                        
                    if path and distance < best_distance:
                        best_path = path
                        best_distance = distance
                        logger.debug(f"New best path found with distance {distance}")
                        
            logger.debug(f"Path search complete: tried {paths_tried} combinations, "
                    f"found {paths_found} valid paths, "
                    f"best distance: {best_distance}")
            
            if best_distance == float('inf'):
                logger.error("No valid path found between any node pairs")
                # Debug network connectivity
                network = getattr(self, f'_{network_type}_network')
                if network:
                    for start_node in start_nodes:
                        for end_node in end_nodes:
                            if not nx.has_path(network, start_node, end_node):
                                logger.error(f"No path exists between nodes {start_node} and {end_node}")
                            else:
                                logger.error(f"Path exists but calculation failed between {start_node} and {end_node}")
            
            result = (best_path or [], best_distance)
            
            # Cache result
            with self.cache_lock:
                self.path_cache[cache_key] = result
                
            return result
            
        except Exception as e:
            logger.error(f"Error calculating shortest path: {str(e)}", exc_info=True)
            return [], float('inf')

    def _calculate_path(self, 
                    start_node: int,
                    end_node: int,
                    network_type: str,
                    weight: str = 'distance') -> Tuple[List[int], float]:
        """Enhanced path calculation with debugging"""
        network = getattr(self, f'_{network_type}_network')
        
        try:
            # Verify nodes exist in network
            if start_node not in network:
                logger.error(f"Start node {start_node} not in network")
                return [], float('inf')
            if end_node not in network:
                logger.error(f"End node {end_node} not in network")
                return [], float('inf')
                
            # Calculate direct distance for comparison
            start_coords = self.node_coords[network_type][start_node]
            end_coords = self.node_coords[network_type][end_node]
            
            # Check if nodes are in same connected component
            if not nx.has_path(network, start_node, end_node):
                logger.error(f"No path exists between nodes {start_node} and {end_node}")
                return [], float('inf')
                
            # Try standard shortest path
            try:
                path = nx.shortest_path(network, start_node, end_node, weight=weight)
                distance = nx.shortest_path_length(network, start_node, end_node, weight=weight)
                logger.debug(f"Found path with {len(path)} nodes and distance {distance}")
                return path, distance
                
            except nx.NetworkXNoPath:
                logger.error(f"NetworkXNoPath: No path between nodes {start_node} and {end_node}")
                return [], float('inf')
                
        except Exception as e:
            logger.error(f"Error in _calculate_path: {str(e)}")
            return [], float('inf')
            
    def get_cache_stats(self) -> Dict[str, Union[int, float]]:
        """Get statistics about cache performance"""
        with self.cache_lock:
            total = self.cache_hits + self.cache_misses
            return {
                'hits': self.cache_hits,
                'misses': self.cache_misses,
                'total_requests': total,
                'hit_rate': self.cache_hits / total if total > 0 else 0,
                'path_cache_size': len(self.path_cache),
                'distance_cache_size': len(self.distance_cache),
                'nearest_node_cache_size': len(self.nearest_node_cache)
            }
            
    def clear_caches(self):
        """Clear all calculation caches and reset statistics"""
        with self.cache_lock:
            self.path_cache.clear()
            self.distance_cache.clear()
            self.nearest_node_cache.clear()
            self.get_cached_path.cache_clear()
            self._get_nearest_nodes.cache_clear()
            self.cache_hits = 0
            self.cache_misses = 0
            
    def optimize_memory_usage(self, max_cache_size: Optional[int] = None):
        """Optimize memory usage by limiting cache sizes"""
        max_size = max_cache_size or self.config.cache_size
        
        with self.cache_lock:
            if len(self.path_cache) > max_size:
                self.path_cache.clear()
            if len(self.distance_cache) > max_size:
                self.distance_cache.clear()
            if len(self.nearest_node_cache) > max_size // 10:
                self.nearest_node_cache.clear()
                
            # Reset LRU cache sizes
            self.get_cached_path = lru_cache(10000)(self._calculate_path)
            self._get_nearest_nodes = lru_cache(1000)(self._find_nearest_node_internal)
            
    def get_network_info(self, network_type: str) -> NetworkInfo:
        """Get information about loaded network"""
        if network_type not in self._info:
            raise ValueError(f"No info available for {network_type} network")
        return self._info[network_type]