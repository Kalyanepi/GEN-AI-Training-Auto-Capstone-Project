"""Structured request/response logging middleware.

WHY middleware (not per-route): centralizes timing + correlation-id binding.
Every request gets a `request_id` automatically attached to all log records
made during its lifetime via structlog contextvars.
"""
from __future__ import annotations

import time
import uuid

from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware
from structlog.contextvars import bind_contextvars, clear_contextvars

from observability.logger import get_logger

logger = get_logger(__name__)


class StructuredLoggingMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        request_id = request.headers.get("x-request-id") or str(uuid.uuid4())
        clear_contextvars()
        bind_contextvars(request_id=request_id, path=request.url.path, method=request.method)

        start = time.perf_counter()
        logger.info("request_received")
        try:
            response = await call_next(request)
        except Exception as e:
            logger.error("request_failed", error=str(e), exc_info=True)
            raise
        latency_ms = int((time.perf_counter() - start) * 1000)
        response.headers["x-request-id"] = request_id
        response.headers["x-latency-ms"] = str(latency_ms)
        logger.info(
            "request_completed",
            status=response.status_code,
            latency_ms=latency_ms,
        )
        return response
