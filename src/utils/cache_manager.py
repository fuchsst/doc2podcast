"""Cache management utilities"""
import os
import json
import time
import hashlib
from pathlib import Path
from typing import Any, Optional, Callable
from functools import wraps
import pickle

class CacheManager:
    """Manages caching of function results"""
    
    def __init__(self, cache_dir: str = ".cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        
    def _get_cache_key(self, *args, **kwargs) -> str:
        """Generate cache key from function arguments"""
        # Convert args and kwargs to string and hash
        args_str = str(args) + str(sorted(kwargs.items()))
        return hashlib.md5(args_str.encode()).hexdigest()
        
    def _get_cache_path(self, func_name: str, cache_key: str) -> Path:
        """Get path for cached result"""
        return self.cache_dir / f"{func_name}_{cache_key}.pkl"
        
    def cache(self, ttl: Optional[int] = None) -> Callable:
        """Cache decorator
        
        Args:
            ttl: Time to live in seconds (optional)
            
        Returns:
            Decorated function that caches results
        """
        def decorator(func: Callable) -> Callable:
            @wraps(func)
            def wrapper(*args, **kwargs):
                # Generate cache key
                cache_key = self._get_cache_key(*args, **kwargs)
                cache_path = self._get_cache_path(func.__name__, cache_key)
                
                # Check if cached result exists and is valid
                if cache_path.exists():
                    # Check TTL if specified
                    if ttl is not None:
                        age = time.time() - cache_path.stat().st_mtime
                        if age > ttl:
                            cache_path.unlink()
                        else:
                            with open(cache_path, 'rb') as f:
                                return pickle.load(f)
                    else:
                        with open(cache_path, 'rb') as f:
                            return pickle.load(f)
                            
                # Generate and cache result
                result = func(*args, **kwargs)
                with open(cache_path, 'wb') as f:
                    pickle.dump(result, f)
                    
                return result
                
            return wrapper
        return decorator
        
    def clear_cache(self, func_name: Optional[str] = None):
        """Clear cached results
        
        Args:
            func_name: Optional function name to clear specific cache
        """
        if func_name:
            pattern = f"{func_name}_*.pkl"
        else:
            pattern = "*.pkl"
            
        for cache_file in self.cache_dir.glob(pattern):
            cache_file.unlink()
            
    def get_cache_size(self) -> int:
        """Get total size of cache in bytes"""
        total_size = 0
        for cache_file in self.cache_dir.glob("*.pkl"):
            total_size += cache_file.stat().st_size
        return total_size
        
    def get_cache_stats(self) -> dict:
        """Get cache statistics"""
        stats = {
            "total_size": self.get_cache_size(),
            "num_files": len(list(self.cache_dir.glob("*.pkl"))),
            "functions": {}
        }
        
        # Group by function name
        for cache_file in self.cache_dir.glob("*.pkl"):
            func_name = cache_file.stem.split("_")[0]
            if func_name not in stats["functions"]:
                stats["functions"][func_name] = {
                    "count": 0,
                    "size": 0
                }
            stats["functions"][func_name]["count"] += 1
            stats["functions"][func_name]["size"] += cache_file.stat().st_size
            
        return stats

cache_manager = CacheManager()
