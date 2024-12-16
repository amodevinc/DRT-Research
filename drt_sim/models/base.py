# drt_sim/models/base.py
from datetime import datetime
from enum import Enum
import uuid

class ModelBase:
    """Base class for all models with common functionality"""
    def __init__(self):
        self.id: str = str(uuid.uuid4())
        self.created_at: datetime = datetime.now()
        self.updated_at: datetime = datetime.now()