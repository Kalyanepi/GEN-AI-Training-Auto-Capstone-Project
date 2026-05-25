"""LangSmith tracing wrapper.

WHY: Centralizing the trace setup means every node/tool gets the same project,
tags, and run metadata — and the API can always return a clickable trace URL
in its response..
"""
from __future__ import annotations

import os
from typing import Optional

from api.config import settings
from observability.logger import get_logger

logger = get_logger(__name__)


def configure_langsmith() -> None:
    """Set LangSmith env vars from settings so LangChain/LangGraph auto-trace.

    WHY: LangChain reads these globals at runtime; setting them once at startup
    avoids per-call configuration drift.
    """
    if not settings.langchain_tracing_v2:
        logger.info("langsmith_disabled")
        return
    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ["LANGCHAIN_API_KEY"] = settings.langchain_api_key
    os.environ["LANGCHAIN_PROJECT"] = settings.langchain_project
    os.environ["LANGCHAIN_ENDPOINT"] = settings.langchain_endpoint
    logger.info("langsmith_configured", project=settings.langchain_project)


def build_trace_url(run_id: Optional[str]) -> Optional[str]:
    """Return a UI URL to view a LangSmith run, or None if tracing is off.

    WHY: The API contract guarantees a `trace_url` field; UI uses this to render
    a one-click link for demo transparency.
    """
    if not run_id or not settings.langchain_tracing_v2:
        return None
    return f"https://smith.langchain.com/o/-/projects/{settings.langchain_project}/r/{run_id}"
