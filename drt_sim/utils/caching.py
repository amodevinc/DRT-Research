# drt_sim/utils/caching.py
from functools import lru_cache
import pickle
from pathlib import Path
from typing import Any, Callable, Optional, TypeVar, Dict
from datetime import datetime, timedelta
import hashlib
import json

T = TypeVar('T')

class DRTCache:
    """Caching system for DRT simulation components"""
    
    def __init__(self, cache_dir: str = "cache",
                 max_size: int = 1000,
                 ttl: Optional[timedelta] = timedelta(hours=24)):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.max_size = max_size
        self.ttl = ttl
        self.metadata: Dict[str, Dict] = self._load_metadata()
        
    def _load_metadata(self) -> Dict:
        """Load cache metadata"""
        metadata_file = self.cache_dir / "metadata.json"
        if metadata_file.exists():
            return json.loads(metadata_file.read_text())
        return {}
    
    def _save_metadata(self) -> None:
        """Save cache metadata"""
        metadata_file = self.cache_dir / "metadata.json"
        metadata_file.write_text(json.dumps(self.metadata, indent=2))
    
    def _generate_key(self, *args, **kwargs) -> str:
        """Generate a unique cache key"""
        key_data = {
            'args': args,
            'kwargs': kwargs
        }
        key_str = json.dumps(key_data, sort_keys=True)
        return hashlib.sha256(key_str.encode()).hexdigest()
    
    def cache_result(self, ttl: Optional[timedelta] = None) -> Callable:
        """Decorator for caching function results"""
        def decorator(func: Callable[..., T]) -> Callable[..., T]:
            @lru_cache(maxsize=self.max_size)
            def wrapper(*args, **kwargs) -> T:
                cache_key = self._generate_key(func.__name__, *args, **kwargs)
                
                # Check if cached result exists and is valid
                if cache_key in self.metadata:
                    entry = self.metadata[cache_key]
                    cache_time = datetime.fromisoformat(entry['timestamp'])
                    current_ttl = ttl or self.ttl
                    
                    if (not current_ttl or 
                        datetime.now() - cache_time < current_ttl):
                        cache_file = self.cache_dir / f"{cache_key}.pickle"
                        if cache_file.exists():
                            return pickle.loads(cache_file.read_bytes())
                
                # Calculate and cache result
                result = func(*args, **kwargs)
                self._cache_value(cache_key, result)
                return result
            
            return wrapper
        return decorator
    
    def _cache_value(self, key: str, value: Any) -> None:
        """Cache a value with metadata"""
        cache_file = self.cache_dir / f"{key}.pickle"
        cache_file.write_bytes(pickle.dumps(value))
        
        self.metadata[key] = {
            'timestamp': datetime.now().isoformat(),
            'size': cache_file.stat().st_size
        }
        self._save_metadata()
        
    def clear_expired(self) -> None:
        """Clear expired cache entries"""
        current_time = datetime.now()
        expired_keys = []
        
        for key, entry in self.metadata.items():
            cache_time = datetime.fromisoformat(entry['timestamp'])
            if self.ttl and current_time - cache_time > self.ttl:
                expired_keys.append(key)
                
        for key in expired_keys:
            self._remove_cache_entry(key)
    
    def _remove_cache_entry(self, key: str) -> None:
        """Remove a cache entry and its metadata"""
        cache_file = self.cache_dir / f"{key}.pickle"
        if cache_file.exists():
            cache_file.unlink()
        
        if key in self.metadata:
            del self.metadata[key]
            
        self._save_metadata()