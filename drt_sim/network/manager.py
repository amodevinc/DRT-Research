import logging
import threading
import asyncio
import math
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any
import networkx as nx
import geopandas as gpd
from pyproj import Transformer
from scipy.spatial import cKDTree
from functools import lru_cache
from concurrent.futures import ThreadPoolExecutor
from shapely.geometry import Point, Polygon
from drt_sim.config.config import NetworkConfig, NetworkInfo
from drt_sim.models.location import Location

logger = logging.getLogger(__name__)

class NetworkManager:
    """
    Enhanced NetworkManager class with optimized path finding, spatial indexing,
    service area support, caching, robust fallback strategies, and waypoint optimization.
    """
    def __init__(self, config: NetworkConfig):
        # Store configuration and initialize network-related variables
        self.config = config
        self.drive_network: Optional[nx.Graph] = None
        self.walk_network: Optional[nx.Graph] = None
        self._transformer: Optional[Transformer] = None
        self._info: Dict[str, NetworkInfo] = {}
        self.spatial_indexes: Dict[str, cKDTree] = {}
        self.node_coords: Dict[str, Dict[Any, Tuple[float, float]]] = {}
        self.node_lists: Dict[str, List[Any]] = {}
        self.service_area: Optional[Polygon] = None

        # Initialize executor and caches
        self._init_executor()
        self._init_caches()

        # Load networks and initialize spatial indexes and service area
        self.initialize_networks()
        self._init_spatial_indexes()
        self._init_service_area()

        # Define mode-specific path thresholds (drive and walk)
        self.path_thresholds = {
            'drive': {
                'distance_ratio': 2.5,
                'max_detour': 3000,       # in meters
                'alternate_nodes': 3,
                'search_radius': 500      # in meters
            },
            'walk': {
                'distance_ratio': 1.4,
                'max_detour': 500,
                'alternate_nodes': 5,
                'search_radius': 200
            }
        }

        logger.info("NetworkManager initialized successfully")

    # -----------------------------------------------------------
    # Initialization Methods
    # -----------------------------------------------------------
    def _init_executor(self):
        """Initialize the thread pool executor for background tasks."""
        self.executor = ThreadPoolExecutor(max_workers=10)

    def _init_caches(self):
        """Initialize thread-safe caches and LRU wrappers."""
        self.cache_lock = threading.Lock()
        self.path_cache: Dict = {}
        self.distance_cache: Dict = {}
        self.nearest_node_cache: Dict = {}
        self.cache_hits = 0
        self.cache_misses = 0

        # LRU caches for path calculation and nearest node lookup
        self.get_cached_path = lru_cache(maxsize=10000)(self._calculate_path)
        self._get_nearest_node = lru_cache(maxsize=1000)(self._find_nearest_node_internal)

    def initialize_networks(self):
        """
        Load and initialize drive and walk networks from configuration files.
        """
        logger.info("Initializing networks...")
        if self.config.network_file:
            self.drive_network = self._load_network(self.config.network_file, "drive")
            logger.info(f"Drive network loaded with {self.drive_network.number_of_nodes()} nodes")
        if self.config.walk_network_file:
            self.walk_network = self._load_network(self.config.walk_network_file, "walk")
            logger.info(f"Walk network loaded with {self.walk_network.number_of_nodes()} nodes")

    def _init_spatial_indexes(self):
        """
        Build spatial indexes (KD-trees) for drive and walk networks for fast coordinate lookups.
        """
        logger.info("Initializing spatial indexes...")
        for network_type in ['drive', 'walk']:
            network = getattr(self, f"{network_type}_network")
            if network is not None:
                # Create coordinate array (using (y, x) order)
                coords = np.array([
                    (data.get('y'), data.get('x'))
                    for _, data in network.nodes(data=True)
                ])
                # Build KD-tree
                self.spatial_indexes[network_type] = cKDTree(coords)
                # Store node coordinates and node list
                self.node_coords[network_type] = {
                    node: (data.get('y'), data.get('x'))
                    for node, data in network.nodes(data=True)
                }
                self.node_lists[network_type] = list(network.nodes())
                logger.info(f"Spatial index created for {network_type} network")

    def _init_service_area(self):
        """
        Initialize the service area polygon if provided in the configuration.
        """
        if self.config.service_area_polygon:
            self.service_area = Polygon(self.config.service_area_polygon)
        else:
            self.service_area = None

    def _load_network(self, file_path: Union[str, Path], network_type: str) -> nx.Graph:
        """
        Load and process a network from a file.
        Supports .graphml and .geojson file formats.
        """
        file_path = Path(file_path)
        logger.info(f"Loading {network_type} network from {file_path}")
        if not file_path.exists():
            raise FileNotFoundError(f"Network file not found: {file_path}")
        try:
            if file_path.suffix == '.graphml':
                G = nx.read_graphml(file_path)
            elif file_path.suffix == '.geojson':
                G = self._load_from_geojson(file_path)
            else:
                raise ValueError(f"Unsupported file format: {file_path.suffix}")
            G = self._process_network(G, network_type)
            self._info[network_type] = self._create_network_info(G, file_path.stem)
            return G
        except Exception as e:
            logger.error(f"Error loading network: {str(e)}")
            raise

    def _load_from_geojson(self, file_path: Path) -> nx.Graph:
        """Load network from a GeoJSON file."""
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
        """
        Ensure network edges have required attributes like length and travel_time.
        """
        logger.info(f"Processing {network_type} network...")
        speed = self.config.walking_speed if network_type == 'walk' else self.config.driving_speed
        for u, v, data in G.edges(data=True):
            if 'length' not in data:
                data['length'] = self._calculate_length(G, u, v)
            data['travel_time'] = data['length'] / speed
        return G

    def _calculate_length(self, G: nx.Graph, u: Any, v: Any) -> float:
        """
        Calculate Euclidean length between two nodes.
        Uses coordinate transformation if needed.
        """
        if self.config.coordinate_system == "EPSG:4326":
            if not self._transformer:
                self._transformer = Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True)
            start = self._transformer.transform(G.nodes[u]['y'], G.nodes[u]['x'])
            end = self._transformer.transform(G.nodes[v]['y'], G.nodes[v]['x'])
            return math.sqrt((end[0] - start[0])**2 + (end[1] - start[1])**2)
        else:
            return math.sqrt((G.nodes[v]['x'] - G.nodes[u]['x'])**2 + (G.nodes[v]['y'] - G.nodes[u]['y'])**2)

    def _create_network_info(self, G: nx.Graph, name: str) -> NetworkInfo:
        """
        Create and return a NetworkInfo object with basic details of the network.
        """
        nodes = G.nodes(data=True)
        lats = [data.get('y') for _, data in nodes if 'y' in data]
        lons = [data.get('x') for _, data in nodes if 'x' in data]
        return NetworkInfo(
            name=name,
            node_count=G.number_of_nodes(),
            edge_count=G.number_of_edges(),
            bbox=(min(lons), min(lats), max(lons), max(lats)),
            crs=self.config.coordinate_system
        )

    # -----------------------------------------------------------
    # Service Area Functions
    # -----------------------------------------------------------
    def is_point_in_service_area(self, lat: float, lon: float) -> bool:
        """
        Check if a given point (lat, lon) lies within the defined service area.
        """
        if self.service_area is None:
            return True
        return self.service_area.contains(Point(lon, lat))

    def filter_nodes_by_service_area(self, nodes: List[Any], network_type: str) -> List[Any]:
        """
        Filter and return only those nodes that fall within the service area.
        """
        if self.service_area is None:
            return nodes
        return [
            node for node in nodes
            if self.is_point_in_service_area(*self.get_node_coordinates(node, network_type))
        ]

    def get_node_coordinates(self, node: Any, network_type: str) -> Tuple[float, float]:
        """
        Get coordinates of a node from the specified network.
        """
        try:
            return self.node_coords[network_type][node]
        except KeyError:
            raise ValueError(f"Node {node} not found in {network_type} network")

    # -----------------------------------------------------------
    # Utility Functions
    # -----------------------------------------------------------
    def _haversine_distance(self, lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        """
        Calculate the great-circle distance between two points on the earth (in meters).
        """
        R = 6371000  # Earth's radius in meters
        lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        a = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
        c = 2 * math.asin(math.sqrt(a))
        return R * c

    def _find_nearest_node_internal(self, lat: float, lon: float, network_type: str) -> Any:
        """
        Internal function to find the nearest node to a given (lat, lon) using KD-tree.
        """
        tree = self.spatial_indexes[network_type]
        nodes = self.node_lists[network_type]
        coords = np.array([[lat, lon]])
        _, index = tree.query(coords, k=1)
        return nodes[index[0]]

    async def get_nearest_node(self, location: Location, network_type: str = 'drive') -> Any:
        """
        Asynchronous wrapper to get nearest node for a given location with caching.
        """
        cache_key = (location.lat, location.lon, network_type)
        with self.cache_lock:
            if cache_key in self.nearest_node_cache:
                self.cache_hits += 1
                return self.nearest_node_cache[cache_key]
            self.cache_misses += 1
        node = await asyncio.get_event_loop().run_in_executor(
            self.executor,
            self._get_nearest_node,
            location.lat,
            location.lon,
            network_type
        )
        with self.cache_lock:
            self.nearest_node_cache[cache_key] = node
        return node

    async def _find_alternate_nodes(self, lat: float, lon: float, network_type: str) -> List[Any]:
        """
        Find alternate nearest nodes based on mode-specific thresholds.
        """
        thresholds = self.path_thresholds[network_type]
        tree = self.spatial_indexes[network_type]
        nodes = self.node_lists[network_type]
        coords = np.array([[lat, lon]])
        k = thresholds['alternate_nodes'] * 2
        distances, indices = tree.query(coords, k=k)
        valid_indices = [
            idx for idx, dist in zip(indices[0], distances[0])
            if dist * 111000 <= thresholds['search_radius']
        ]
        return [nodes[idx] for idx in valid_indices[:thresholds['alternate_nodes']]]

    # -----------------------------------------------------------
    # Path Calculation Functions
    # -----------------------------------------------------------
    async def get_shortest_path(self, 
                            start_location: Location, 
                            end_location: Location, 
                            network_type: str = 'drive', 
                            weight: str = 'distance',
                            depth: int = 0,
                            max_depth: int = 5) -> Tuple[List[Any], float]:
        """
        Asynchronously calculate the shortest path between two locations using the specified network.
        Includes a recursion depth limit for fallback strategies.
        Returns a tuple (path, total_distance).
        """
        # Quick return if locations are nearly identical
        if abs(start_location.lat - end_location.lat) < 1e-7 and abs(start_location.lon - end_location.lon) < 1e-7:
            return [], 0.0
        cache_key = (
            start_location.lat, start_location.lon,
            end_location.lat, end_location.lon,
            network_type, weight
        )
        with self.cache_lock:
            if cache_key in self.path_cache:
                self.cache_hits += 1
                return self.path_cache[cache_key]
            self.cache_misses += 1

        start_alts = await self._find_alternate_nodes(start_location.lat, start_location.lon, network_type)
        end_alts = await self._find_alternate_nodes(end_location.lat, end_location.lon, network_type)

        best_path = None
        best_distance = float('inf')
        # Try all alternate combinations
        for start_node in start_alts:
            for end_node in end_alts:
                path, distance = await asyncio.get_event_loop().run_in_executor(
                    self.executor,
                    self.get_cached_path,
                    start_node,
                    end_node,
                    network_type,
                    weight
                )
                if path and distance < best_distance:
                    best_path = path
                    best_distance = distance

        # Fallback: if no valid path found and recursion depth not exceeded, try using an intermediate midpoint
        if best_distance == float('inf') and depth < max_depth:
            logger.debug("No valid path found using alternate nodes. Trying intermediate midpoint fallback.")
            mid_location = Location(
                (start_location.lat + end_location.lat) / 2,
                (start_location.lon + end_location.lon) / 2
            )
            path1, dist1 = await self.get_shortest_path(start_location, mid_location, network_type, weight, depth + 1, max_depth)
            path2, dist2 = await self.get_shortest_path(mid_location, end_location, network_type, weight, depth + 1, max_depth)
            if path1 and path2 and dist1 != float('inf') and dist2 != float('inf'):
                best_path = path1[:-1] + path2
                best_distance = dist1 + dist2

        result = (best_path or [], best_distance)
        with self.cache_lock:
            self.path_cache[cache_key] = result
        return result


    def _calculate_path(self, start_node: Any, end_node: Any, network_type: str, weight: str = 'distance') -> Tuple[List[Any], float]:
        """
        Calculate path between two nodes using NetworkX shortest_path.
        Includes fallback and error handling.
        """
        network = self.drive_network if network_type == 'drive' else self.walk_network
        if network is None:
            logger.error(f"{network_type} network is not loaded")
            return [], float('inf')
        if start_node not in network or end_node not in network:
            logger.error(f"Nodes {start_node} or {end_node} not in {network_type} network")
            return [], float('inf')
        if not nx.has_path(network, start_node, end_node):
            logger.error(f"No path exists between {start_node} and {end_node} in {network_type} network")
            return [], float('inf')
        try:
            path = nx.shortest_path(network, start_node, end_node, weight=weight)
            distance = nx.shortest_path_length(network, start_node, end_node, weight=weight)
            if distance == 0:
                return path, 50
            return path, distance
        except nx.NetworkXNoPath:
            logger.error(f"NetworkXNoPath: No path found between {start_node} and {end_node}")
            return [], float('inf')
        except Exception as e:
            logger.error(f"Error calculating path: {str(e)}")
            return [], float('inf')

    async def calculate_distance(self, loc1: Location, loc2: Location, network_type: str = 'drive', weight: str = 'distance') -> float:
        """
        Calculate the network distance between two locations.
        Falls back to direct haversine distance if no path is found.
        """
        cache_key = (loc1.lat, loc1.lon, loc2.lat, loc2.lon, network_type, weight)
        with self.cache_lock:
            if cache_key in self.distance_cache:
                self.cache_hits += 1
                return self.distance_cache[cache_key]
            self.cache_misses += 1

        path, distance = await self.get_shortest_path(loc1, loc2, network_type, weight)
        if not path or distance == float('inf'):
            direct_distance = self._haversine_distance(loc1.lat, loc1.lon, loc2.lat, loc2.lon)
            with self.cache_lock:
                self.distance_cache[cache_key] = direct_distance
            return direct_distance

        with self.cache_lock:
            self.distance_cache[cache_key] = distance
        return distance

    async def calculate_travel_time(self, loc1: Location, loc2: Location, network_type: str = 'drive') -> float:
        """
        Calculate travel time between two locations based on network distance and configured speeds.
        """
        distance = await self.calculate_distance(loc1, loc2, network_type, weight='distance')
        speed = self.config.walking_speed if network_type == 'walk' else self.config.driving_speed
        return distance / speed if speed > 0 else float('inf')
    
    def interpolate_waypoints(
        self,
        origin: Location,
        destination: Location,
        num_points: int
    ) -> List[Location]:
        """
        Create interpolated waypoints between origin and destination locations.
        
        Args:
            origin: Starting location
            destination: Ending location
            num_points: Number of intermediate points to generate (excluding origin and destination)
            
        Returns:
            List of Location objects representing waypoints along the path
        """
        try:
            if num_points < 1:
                return []
                
            # Convert to radians for spherical calculations
            lat1, lon1 = math.radians(origin.lat), math.radians(origin.lon)
            lat2, lon2 = math.radians(destination.lat), math.radians(destination.lon)
            
            # Calculate great circle distance
            d = math.acos(
                math.sin(lat1) * math.sin(lat2) + 
                math.cos(lat1) * math.cos(lat2) * math.cos(lon2 - lon1)
            )
            
            # If points are too close, return minimal interpolation
            if abs(d) < 1e-7:
                return []
                
            waypoints = []
            
            # Create interpolated points
            for i in range(1, num_points + 1):
                # Calculate fraction of distance
                f = i / (num_points + 1)
                
                # Calculate intermediate point using spherical interpolation
                A = math.sin((1 - f) * d) / math.sin(d)
                B = math.sin(f * d) / math.sin(d)
                
                x = A * math.cos(lat1) * math.cos(lon1) + B * math.cos(lat2) * math.cos(lon2)
                y = A * math.cos(lat1) * math.sin(lon1) + B * math.cos(lat2) * math.sin(lon2)
                z = A * math.sin(lat1) + B * math.sin(lat2)
                
                # Convert back to lat/lon
                lat = math.atan2(z, math.sqrt(x * x + y * y))
                lon = math.atan2(y, x)
                
                # Convert back to degrees and create Location
                waypoint = Location(
                    lat=math.degrees(lat),
                    lon=math.degrees(lon)
                )
                
                # Only add point if it's sufficiently far from previous point
                if not waypoints or self._haversine_distance(
                    waypoints[-1].lat, waypoints[-1].lon,
                    waypoint.lat, waypoint.lon
                ) >= 10:  # Minimum 10 meters between points
                    waypoints.append(waypoint)
            
            # Filter out any points too close to origin or destination
            filtered_waypoints = []
            for waypoint in waypoints:
                # Check distance from origin and destination
                dist_to_origin = self._haversine_distance(
                    origin.lat, origin.lon,
                    waypoint.lat, waypoint.lon
                )
                dist_to_dest = self._haversine_distance(
                    destination.lat, destination.lon,
                    waypoint.lat, waypoint.lon
                )
                
                # Only keep points that are at least 10m from both endpoints
                if dist_to_origin >= 10 and dist_to_dest >= 10:
                    filtered_waypoints.append(waypoint)
            
            return filtered_waypoints
            
        except Exception as e:
            logger.error(f"Error interpolating waypoints: {str(e)}")
            return []

    async def get_path_info(self, origin_location: Location, destination_location: Location, network_type: str = 'drive') -> Dict[str, Any]:
        """
        Get comprehensive path information including distance, duration, waypoints, and raw path.
        Ensures consistent waypoint spacing for smooth vehicle movement.
        """
        try:
            path, total_distance = await self.get_shortest_path(origin_location, destination_location, network_type)
            speed = self.config.walking_speed if network_type == 'walk' else self.config.driving_speed
            total_duration = total_distance / speed if speed > 0 else float('inf')
            
            # Return early with direct path if locations are too close
            if total_distance < 10:  # Less than 10 meters
                return {
                    'distance': max(50, total_distance),  # Minimum 50m to prevent zero-distance
                    'duration': max(50/speed, total_duration),
                    'waypoints': [{
                        'location': origin_location,
                        'distance': 0,
                        'duration': 0
                    }, {
                        'location': destination_location,
                        'distance': max(50, total_distance),
                        'duration': max(50/speed, total_duration)
                    }],
                    'path': path or []
                }

            # Generate waypoints with consistent spacing
            waypoints = []
            target_spacing = 50  # meters between waypoints
            cumulative_distance = 0
            cumulative_duration = 0
            
            # Always add origin
            waypoints.append({
                'location': origin_location,
                'distance': 0,
                'duration': 0
            })
            
            if path:
                last_added_coords = (origin_location.lat, origin_location.lon)
                current_segment_distance = 0
                
                for i in range(len(path) - 1):
                    node1, node2 = path[i], path[i + 1]
                    coords1 = self.get_node_coordinates(node1, network_type)
                    coords2 = self.get_node_coordinates(node2, network_type)
                    
                    seg_distance = self._haversine_distance(coords1[0], coords1[1], coords2[0], coords2[1])
                    if seg_distance < 1e-6:  # Skip effectively zero-distance segments
                        continue
                        
                    current_segment_distance += seg_distance
                    
                    # Add waypoint if we've covered enough distance
                    if current_segment_distance >= target_spacing:
                        cumulative_distance += current_segment_distance
                        cumulative_duration = cumulative_distance / speed
                        
                        waypoint_location = Location(lat=coords2[0], lon=coords2[1])
                        waypoints.append({
                            'location': waypoint_location,
                            'distance': cumulative_distance,
                            'duration': cumulative_duration
                        })
                        
                        last_added_coords = (coords2[0], coords2[1])
                        current_segment_distance = 0
            
            # Always add destination if it's not the last waypoint
            if (not waypoints or 
                abs(waypoints[-1]['location'].lat - destination_location.lat) > 1e-7 or 
                abs(waypoints[-1]['location'].lon - destination_location.lon) > 1e-7):
                waypoints.append({
                    'location': destination_location,
                    'distance': total_distance,
                    'duration': total_duration
                })

            # Ensure minimum distance between consecutive waypoints
            filtered_waypoints = [waypoints[0]]  # Always keep origin
            for i in range(1, len(waypoints)):
                prev = filtered_waypoints[-1]['location']
                curr = waypoints[i]['location']
                dist = self._haversine_distance(prev.lat, prev.lon, curr.lat, curr.lon)
                if dist >= 10:  # Minimum 10 meters between waypoints
                    filtered_waypoints.append(waypoints[i])
            
            # Ensure we always have the destination
            if filtered_waypoints[-1]['location'] != destination_location:
                filtered_waypoints.append(waypoints[-1])

            return {
                'distance': total_distance,
                'duration': total_duration,
                'waypoints': filtered_waypoints,
                'path': path
            }
        except Exception as e:
            logger.error(f"Error getting path info: {str(e)}")
            direct_distance = self._haversine_distance(
                origin_location.lat, origin_location.lon,
                destination_location.lat, destination_location.lon
            )
            speed = self.config.walking_speed if network_type == 'walk' else self.config.driving_speed
            direct_duration = direct_distance / speed if speed > 0 else float('inf')
            return {
                'distance': direct_distance,
                'duration': direct_duration,
                'waypoints': [{
                    'location': origin_location,
                    'distance': 0,
                    'duration': 0
                }, {
                    'location': destination_location,
                    'distance': direct_distance,
                    'duration': direct_duration
                }],
                'path': []
            }

    # -----------------------------------------------------------
    # Waypoint Optimization Function
    # -----------------------------------------------------------
    def optimize_path(self, locations: List[Location], tolerance: float = 1e-5) -> List[Location]:
        """
        Optimize/simplify a list of locations using the Douglas-Peucker algorithm.
        """
        if len(locations) <= 2:
            return locations

        def point_line_distance(point: Location, start: Location, end: Location) -> float:
            # Handle degenerate case
            if start.lat == end.lat and start.lon == end.lon:
                return self._haversine_distance(point.lat, point.lon, start.lat, start.lon)
            # Calculate perpendicular distance
            numerator = abs(
                (end.lon - start.lon) * (start.lat - point.lat) -
                (start.lon - point.lon) * (end.lat - start.lat)
            )
            denominator = math.sqrt((end.lon - start.lon)**2 + (end.lat - start.lat)**2)
            return numerator / denominator if denominator != 0 else 0

        def douglas_peucker(points: List[Location], epsilon: float) -> List[Location]:
            if len(points) < 3:
                return points
            dmax = 0.0
            index = 0
            for i in range(1, len(points) - 1):
                d = point_line_distance(points[i], points[0], points[-1])
                if d > dmax:
                    index = i
                    dmax = d
            if dmax > epsilon:
                rec_results1 = douglas_peucker(points[:index+1], epsilon)
                rec_results2 = douglas_peucker(points[index:], epsilon)
                return rec_results1[:-1] + rec_results2
            else:
                return [points[0], points[-1]]

        return douglas_peucker(locations, tolerance)

    # -----------------------------------------------------------
    # Cache Management Functions
    # -----------------------------------------------------------
    def get_cache_stats(self) -> Dict[str, Union[int, float]]:
        """Return statistics about cache usage."""
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
        """Clear all caches and reset statistics."""
        with self.cache_lock:
            self.path_cache.clear()
            self.distance_cache.clear()
            self.nearest_node_cache.clear()
            self.get_cached_path.cache_clear()
            self._get_nearest_node.cache_clear()
            self.cache_hits = 0
            self.cache_misses = 0

    def optimize_memory_usage(self, max_cache_size: Optional[int] = None):
        """
        Optimize memory usage by clearing caches if they exceed limits and resetting LRU caches.
        """
        max_size = max_cache_size or self.config.cache_size
        with self.cache_lock:
            if len(self.path_cache) > max_size:
                self.path_cache.clear()
            if len(self.distance_cache) > max_size:
                self.distance_cache.clear()
            if len(self.nearest_node_cache) > max_size // 10:
                self.nearest_node_cache.clear()
            self.get_cached_path = lru_cache(maxsize=10000)(self._calculate_path)
            self._get_nearest_node = lru_cache(maxsize=1000)(self._find_nearest_node_internal)
