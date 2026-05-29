"""Token bucket rate limiter.

WHY in-process token bucket: simple, dependency-free, sufficient for single-
instance Phase 2. Architecture plan §16 documents API Gateway + WAF as the
production scale-out path.
"""
from __future__ import annotations

import time
from threading import Lock
from typing import Dict

from fastapi import Request
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

from api.config import settings
from observability.logger import get_logger

logger = get_logger(__name__)


class _Bucket:
    __slots__ = ("tokens", "last_refill")

    def __init__(self, capacity: float) -> None:
        self.tokens = capacity
        self.last_refill = time.monotonic()


class TokenBucketRateLimiter(BaseHTTPMiddleware):
    """Per-IP token bucket. capacity = settings.rate_limit_per_minute."""

    def __init__(self, app) -> None:
        super().__init__(app)
        self.capacity = float(settings.rate_limit_per_minute)
        self.refill_per_sec = self.capacity / 60.0
        self._buckets: Dict[str, _Bucket] = {}
        self._lock = Lock()

    def _take(self, key: str) -> tuple[bool, float]:
        with self._lock:
            now = time.monotonic()
            bucket = self._buckets.get(key)
            if bucket is None:
                bucket = _Bucket(capacity=self.capacity)
                self._buckets[key] = bucket
            elapsed = now - bucket.last_refill
            bucket.tokens = min(self.capacity, bucket.tokens + elapsed * self.refill_per_sec)
            bucket.last_refill = now
            if bucket.tokens >= 1.0:
                bucket.tokens -= 1.0
                return True, 0.0
            # Not enough tokens — compute retry_after.
            deficit = 1.0 - bucket.tokens
            retry_after = deficit / self.refill_per_sec
            return False, retry_after

    async def dispatch(self, request: Request, call_next):
        # WHY skip rate limit on health/ready: deployment health checks should
        # never be throttled or they'll trigger spurious unhealthy states.
        if request.url.path in {"/health", "/ready", "/docs", "/openapi.json"}:
            return await call_next(request)

        client_ip = request.client.host if request.client else "unknown"
        allowed, retry_after = self._take(client_ip)
        if not allowed:
            logger.warning("rate_limited", ip=client_ip, retry_after=retry_after)
            return JSONResponse(
                status_code=429,
                content={
                    "error": "Rate limit exceeded",
                    "retry_after": int(retry_after) + 1,
                },
                headers={"Retry-After": str(int(retry_after) + 1)},
            )
        return await call_next(request)
