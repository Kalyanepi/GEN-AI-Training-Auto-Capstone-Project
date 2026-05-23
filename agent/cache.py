"""Thread-safe LRU cache with optional TTL.

WHY a custom class instead of functools.lru_cache:
- Need TTL semantics (router decisions can go stale after model upgrades).
- Need explicit size + hit/miss telemetry for the structured logger.
- Need thread-safety (uvicorn workers handle concurrent requests).
"""
from __future__ import annotations

import threading
import time
from collections import OrderedDict
from typing import Any, Callable, Generic, Hashable, Optional, TypeVar

from observability.logger import get_logger

logger = get_logger(__name__)

K = TypeVar("K", bound=Hashable)
V = TypeVar("V")


class LRUCache(Generic[K, V]):
    """Bounded LRU cache with optional per-entry TTL.

    Not async-aware: callers wrap their own async fetches around it.
    """

    def __init__(self, name: str, max_size: int = 256, ttl_seconds: Optional[float] = None) -> None:
        if max_size <= 0:
            raise ValueError("max_size must be > 0")
        self.name = name
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds
        self._store: "OrderedDict[K, tuple[V, float]]" = OrderedDict()
        self._lock = threading.Lock()
        self._hits = 0
        self._misses = 0

    def get(self, key: K) -> Optional[V]:
        with self._lock:
            entry = self._store.get(key)
            if entry is None:
                self._misses += 1
                return None
            value, ts = entry
            if self.ttl_seconds is not None and (time.time() - ts) > self.ttl_seconds:
                # Expired — evict.
                self._store.pop(key, None)
                self._misses += 1
                return None
            # Mark MRU.
            self._store.move_to_end(key)
            self._hits += 1
            return value

    def set(self, key: K, value: V) -> None:
        with self._lock:
            if key in self._store:
                self._store.move_to_end(key)
            self._store[key] = (value, time.time())
            while len(self._store) > self.max_size:
                evicted_key, _ = self._store.popitem(last=False)
                logger.debug("lru_evict", cache=self.name, key=str(evicted_key)[:64])

    def get_or_compute(self, key: K, compute: Callable[[], V]) -> V:
        """Sync helper. For async callers, use get/set explicitly."""
        cached = self.get(key)
        if cached is not None:
            return cached
        value = compute()
        self.set(key, value)
        return value

    def stats(self) -> dict[str, Any]:
        with self._lock:
            total = self._hits + self._misses
            return {
                "name": self.name,
                "size": len(self._store),
                "max_size": self.max_size,
                "hits": self._hits,
                "misses": self._misses,
                "hit_rate": round(self._hits / total, 3) if total else 0.0,
            }

    def clear(self) -> None:
        with self._lock:
            self._store.clear()
            self._hits = 0
            self._misses = 0


def normalize_text(text: str) -> str:
    """Lowercase + collapse whitespace for cache-key stability."""
    return " ".join((text or "").lower().split())
