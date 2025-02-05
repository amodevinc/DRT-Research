'''Under Construction'''
from typing import List, Tuple
import numpy as np
from shapely.geometry import Point, Polygon, LineString
from rtree import index
from drt_sim.models.location import Location, haversine_distance

class GeometryManager:
    """Manages geometric operations for DRT simulation"""
    
    def __init__(self):
        self.location_index = index.Index()
        self.locations: List[Location] = []
    
    def add_location(self, location: Location) -> int:
        """Add a location to the spatial index"""
        idx = len(self.locations)
        point = location.to_point()
        self.location_index.insert(idx, point.bounds)
        self.locations.append(location)
        return idx
    
    def nearest_locations(self, 
                         location: Location, 
                         k: int = 1) -> List[Tuple[Location, float]]:
        """Find k nearest locations to a point"""
        point = location.to_point()
        nearest = []
        
        for idx in self.location_index.nearest(point.bounds, k):
            near_loc = self.locations[idx]
            distance = self.haversine_distance(location, near_loc)
            nearest.append((near_loc, distance))
        
        return sorted(nearest, key=lambda x: x[1])
    
    def create_service_area(self, locations: List[Location]) -> Polygon:
        """Create a service area polygon from a set of locations"""
        points = [loc.to_point() for loc in locations]
        return Polygon([[p.x, p.y] for p in points])
    
    def is_in_service_area(self, location: Location, service_area: Polygon) -> bool:
        """Check if a location is within a service area"""
        return service_area.contains(location.to_point())
    
    def calculate_route_length(self, route: List[Location]) -> float:
        """Calculate the total length of a route in meters"""
        if len(route) < 2:
            return 0.0
            
        total_distance = 0.0
        for i in range(len(route) - 1):
            total_distance += self.haversine_distance(route[i], route[i + 1])
            
        return total_distance
    
    def simplify_route(self, 
                      route: List[Location], 
                      tolerance: float = 10.0) -> List[Location]:
        """Simplify a route while preserving essential points"""
        if len(route) < 3:
            return route
            
        points = [loc.to_point() for loc in route]
        line = LineString(points)
        simplified = line.simplify(tolerance)
        
        return [Location.from_point(Point(coord)) 
                for coord in simplified.coords]
    
    def buffer_stops(self, 
                    stops: List[Location], 
                    radius: float) -> List[Polygon]:
        """Create buffer zones around stops"""
        return [loc.to_point().buffer(radius) for loc in stops]
    
    def find_centroid(self, locations: List[Location]) -> Location:
        """Find the centroid of a set of locations"""
        if not locations:
            raise ValueError("Cannot find centroid of empty location set")
            
        points = [loc.to_point() for loc in locations]
        centroid = Polygon([[p.x, p.y] for p in points]).centroid
        
        return Location.from_point(Point(centroid.x, centroid.y))