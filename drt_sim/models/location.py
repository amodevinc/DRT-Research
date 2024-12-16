# drt_sim/models/location.py
from dataclasses import dataclass
from typing import Optional
from shapely.geometry import Point

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
