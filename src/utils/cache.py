"""
Caching utilities for embeddings and intermediate results
"""
import hashlib
import pickle
from pathlib import Path
from typing import Any, Optional
import json


class Cache:
    """Simple file-based cache"""
    
    def __init__(self, cache_dir: str = ".cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True, parents=True)
    
    def _get_hash(self, key: str) -> str:
        """Generate hash for cache key"""
        return hashlib.md5(key.encode()).hexdigest()
    
    def get(self, key: str) -> Optional[Any]:
        """Get cached value"""
        cache_file = self.cache_dir / f"{self._get_hash(key)}.pkl"
        
        if not cache_file.exists():
            return None
        
        try:
            with open(cache_file, 'rb') as f:
                return pickle.load(f)
        except:
            return None
    
    def set(self, key: str, value: Any):
        """Set cache value"""
        cache_file = self.cache_dir / f"{self._get_hash(key)}.pkl"
        
        try:
            with open(cache_file, 'wb') as f:
                pickle.dump(value, f)
        except Exception as e:
            print(f"Warning: Failed to cache value: {e}")
    
    def clear(self):
        """Clear all cache"""
        for cache_file in self.cache_dir.glob("*.pkl"):
            cache_file.unlink()


def get_text_hash(text: str) -> str:
    """Get hash of text for caching/comparison"""
    return hashlib.sha256(text.encode()).hexdigest()[:16]

