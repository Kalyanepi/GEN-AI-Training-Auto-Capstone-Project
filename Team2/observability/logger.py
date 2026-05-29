"""Structured JSON logger used by every module.

WHY: Production debugging on EC2 requires machine-parseable logs. Plain text
logs make it impossible to filter by session_id, tool name, or latency..
"""
from __future__ import annotations

import logging
import sys

import structlog

from api.config import settings


def _configure_stdlib_logging() -> None:
    """Route stdlib logs through structlog at the configured level."""
    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=getattr(logging, settings.log_level.upper(), logging.INFO),
    )


def configure_logging() -> None:
    """Configure structlog once at process start.

    WHY: Idempotent configure prevents duplicate handlers when reload happens.
    """
    _configure_stdlib_logging()

    processors = [
        structlog.contextvars.merge_contextvars,
        structlog.processors.add_log_level,
        structlog.processors.TimeStamper(fmt="iso", utc=True),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
    ]
    if settings.log_format == "json":
        processors.append(structlog.processors.JSONRenderer())
    else:
        processors.append(structlog.dev.ConsoleRenderer())

    structlog.configure(
        processors=processors,
        wrapper_class=structlog.make_filtering_bound_logger(
            getattr(logging, settings.log_level.upper(), logging.INFO)
        ),
        logger_factory=structlog.PrintLoggerFactory(),
        cache_logger_on_first_use=True,
    )


def get_logger(name: str) -> structlog.stdlib.BoundLogger:
    """Get a named structured logger.

    Usage:
        logger = get_logger(__name__)
        logger.info("tool_executed", tool="repair_cost", latency_ms=12)
    """
    return structlog.get_logger(name)


configure_logging()
