from dataclasses import dataclass, field
from typing import Optional, Dict, Any
from shapely.geometry import Point
import numpy as np
from drt_sim.models.base import ModelBase
import uuid

@dataclass
class Location(ModelBase):
    """Represents a location in the DRT system"""
    lat: float
    lon: float
    elevation: Optional[float] = None
    id: str = field(default_factory=lambda: str(uuid.uuid4()))

    def __str__(self) -> str:
        """Provides a concise string representation of the location"""
        elev = f"|elev={self.elevation:.1f}m" if self.elevation is not None else ""
        return f"Loc[{self.id[:8]}|{self.lat:.5f},{self.lon:.5f}{elev}]"

    def __post_init__(self):
        """Initialize ModelBase attributes after dataclass initialization"""
        super().__init__()
        # Override id if it was provided in initialization
        if not isinstance(self.id, str):
            self.id = str(uuid.uuid4())

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
        location_dict = {
            'id': self.id,
            'lat': self.lat,
            'lon': self.lon,
        }
        return {**location_dict}
    
    def to_json(self) -> str:
        """Convert Location to JSON string representation."""
        import json
        return json.dumps(self.to_dict())

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Location':
        """Create Location from dictionary"""
        return cls(
            id=data.get('id'),
            lat=data['lat'],
            lon=data['lon'],
        )

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