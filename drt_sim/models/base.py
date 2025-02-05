# drt_sim/models/base.py
from datetime import datetime
from enum import Enum
import uuid
import json

class SimulationEncoder(json.JSONEncoder):
    """Custom JSON encoder for simulation state classes"""
    def default(self, obj):
        if isinstance(obj, datetime):
            return obj.isoformat()
        if isinstance(obj, Enum):
            return obj.value
        if isinstance(obj, ModelBase):
            return obj.to_dict()
        return super().default(obj)
    
class ModelBase:
    """Base class for all models with common functionality"""
    def __init__(self):
        self.id: str = str(uuid.uuid4())
        self.created_at: datetime = datetime.now()
        self.updated_at: datetime = datetime.now()
    def to_json(self) -> str:
        """Convert the object to a JSON string"""
        return json.dumps(self, cls=SimulationEncoder)
    
    @classmethod
    def from_json(cls, json_str: str) -> 'ModelBase':
        """Create an object from a JSON string"""
        data = json.loads(json_str)
        return cls.from_dict(data)