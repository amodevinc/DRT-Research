# drt_sim/core/state/base.py
from typing import Dict, Any, Optional, TypeVar, Generic
from datetime import datetime
from dataclasses import dataclass, field
from collections import defaultdict
from abc import ABC, abstractmethod

T = TypeVar('T')

@dataclass
class StateContainer(Generic[T]):
    """Generic container for state items with history tracking"""
    _items: Dict[str, T] = field(default_factory=dict)
    _history: Dict[datetime, Dict[str, T]] = field(default_factory=lambda: defaultdict(dict))
    _snapshot_buffer: Dict[str, T] = field(default_factory=dict)
    
    def add(self, id: str, item: T) -> None:
        """Add new item to container"""
        if id in self._items:
            raise ValueError(f"Item with id {id} already exists")
        self._items[id] = item
        
    def get(self, id: str) -> Optional[T]:
        """Get item by ID"""
        return self._items.get(id)
        
    def update(self, id: str, item: T) -> None:
        """Update existing item"""
        if id not in self._items:
            raise KeyError(f"Item {id} not found")
        self._items[id] = item
            
    def remove(self, id: str) -> None:
        """Remove item from container"""
        if id not in self._items:
            raise KeyError(f"Item {id} not found")
        self._items.pop(id)
        
    def take_snapshot(self, timestamp: datetime) -> None:
        """Take snapshot of current state"""
        self._history[timestamp] = self._items.copy()
        
    def get_snapshot(self, timestamp: datetime) -> Dict[str, T]:
        """Get historical snapshot"""
        if timestamp not in self._history:
            raise KeyError(f"No snapshot exists for timestamp {timestamp}")
        return self._history[timestamp]
        
    def clear_history(self) -> None:
        """Clear historical data"""
        self._history.clear()
        
    def begin_transaction(self) -> None:
        """Begin state transaction"""
        self._snapshot_buffer = self._items.copy()
        
    def commit_transaction(self) -> None:
        """Commit current transaction"""
        self._snapshot_buffer.clear()
        
    def rollback_transaction(self) -> None:
        """Rollback current transaction"""
        self._items = self._snapshot_buffer.copy()
        self._snapshot_buffer.clear()
        
    @property
    def items(self) -> Dict[str, T]:
        """Get all items (read-only)"""
        return self._items.copy()

class StateWorker(ABC):
    """Abstract base class defining interface for state workers"""
    
    @abstractmethod
    def initialize(self, config: Optional[Any] = None) -> None:
        """Initialize worker with config"""
        pass
        
    @abstractmethod
    def take_snapshot(self, timestamp: datetime) -> None:
        """Take state snapshot"""
        pass
        
    @abstractmethod
    def get_state(self) -> Dict[str, Any]:
        """Get current worker state"""
        pass
        
    @abstractmethod
    def update_state(self, state: Dict[str, Any]) -> None:
        """Update worker state"""
        pass
        
    @abstractmethod
    def restore_state(self, saved_state: Dict[str, Any]) -> None:
        """Restore worker state from saved state"""
        pass
        
    @abstractmethod
    def begin_transaction(self) -> None:
        """Begin state transaction"""
        pass
        
    @abstractmethod
    def commit_transaction(self) -> None:
        """Commit current transaction"""
        pass
        
    @abstractmethod
    def rollback_transaction(self) -> None:
        """Rollback current transaction"""
        pass
        
    @abstractmethod
    def cleanup(self) -> None:
        """Clean up worker resources"""
        pass