# drt_sim/models/location.py
from dataclasses import dataclass
from typing import Optional, Dict, Any
from shapely.geometry import Point
import numpy as np

@dataclass
class Location:
    """Represents a location in the DRT system"""
    lat: float
    lon: float
    elevation: Optional[float] = None
    
    
    def to_point(self) -> Point:
        """Convert to Shapely Point"""
        return Point(self.lon, self.lat)
    
    @classmethod
    def from_point(cls, point: Point) -> 'Location':
        """Create Location from Shapely Point"""
        return cls(lat=point.y, lon=point.x)
    
    def distance_to(self, other: 'Location') -> float:
        """Calculate the distance between two locations in meters"""
        return haversine_distance(self, other)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert Location to dictionary"""
        return {
            'lat': self.lat,
            'lon': self.lon,
            'elevation': self.elevation
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Location':
        """Create Location from dictionary"""
        return cls(lat=data['lat'], lon=data['lon'], elevation=data.get('elevation'))
    
def haversine_distance(loc1: Location, loc2: Location) -> float:
    """Calculate Haversine distance between two locations in meters"""
    R = 6371000  # Earth's radius in meters
    
    lat1, lon1 = np.radians(loc1.lat), np.radians(loc1.lon)
    lat2, lon2 = np.radians(loc2.lat), np.radians(loc2.lon)
    
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    c = 2 * np.arcsin(np.sqrt(a))
    
    return R * c