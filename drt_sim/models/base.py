# drt_sim/models/base.py
from datetime import datetime
from enum import Enum
import uuid
import json
from numpy import int64
from typing import Optional

class SimulationEncoder(json.JSONEncoder):
    """Custom JSON encoder for simulation state classes"""
    def default(self, obj):
        # Get the class name for better error messages
        class_name = obj.__class__.__name__

        if isinstance(obj, datetime):
            return obj.isoformat()
        if isinstance(obj, Enum):
            return obj.value
        if isinstance(obj, ModelBase):
            return obj.to_dict()
        if isinstance(obj, int64):
            return int(obj)
        if isinstance(obj, dict):
            # Convert dictionary with non-string keys to use string keys
            return {str(key): value for key, value in obj.items()}
        # Add handling for objects with to_dict method
        if hasattr(obj, 'to_dict') and callable(getattr(obj, 'to_dict')):
            return obj.to_dict()
            
        raise TypeError(f'Object of type {class_name} is not JSON serializable. '
                       f'Consider implementing to_dict() method or adding specific handling to SimulationEncoder.')
    
class ModelBase:
    """Base class for all models with common functionality"""
    def __init__(self, id: Optional[str] = None):
        self.id: str = id or str(uuid.uuid4())
        self.created_at: datetime = datetime.now()
        self.updated_at: datetime = datetime.now()

    def to_dict(self) -> dict:
        """Convert the object to a dictionary with JSON-serializable values"""
        return {
            'id': self.id,
            'created_at': self.created_at,
            'updated_at': self.updated_at
        }

    def to_json(self) -> str:
        """Convert the object to a JSON string"""
        return json.dumps(self, cls=SimulationEncoder)
    
    @classmethod
    def from_json(cls, json_str: str) -> 'ModelBase':
        """Create an object from a JSON string"""
        data = json.loads(json_str)
        return cls.from_dict(data)

    def _convert_to_serializable(self, value):
        """Helper method to convert values to JSON-serializable format"""
        if isinstance(value, dict):
            return {str(k): self._convert_to_serializable(v) for k, v in value.items()}
        elif isinstance(value, list):
            return [self._convert_to_serializable(item) for item in value]
        elif isinstance(value, Enum):
            return value.value
        elif hasattr(value, 'to_dict') and callable(getattr(value, 'to_dict')):
            return value.to_dict()
        return value