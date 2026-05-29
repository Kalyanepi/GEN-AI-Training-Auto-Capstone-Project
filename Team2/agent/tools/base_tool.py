"""Abstract BaseTool — every tool extends this for consistent behavior.

WHY a base class: uniform logging, error handling, and ToolResult shape mean
the orchestrator can treat every tool identically. New tools "just work" with
the LangGraph node + output guardrail without per-tool branching.
"""
from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from observability.logger import get_logger
from rag.citation_tracker import Citation


@dataclass
class ToolResult:
    """Uniform tool return shape consumed by the LangGraph orchestrator."""
    tool_name: str
    success: bool
    data: Optional[Dict[str, Any]] = None
    citations: List[Citation] = field(default_factory=list)
    dollar_values: List[float] = field(default_factory=list)  # for fabrication guardrail
    error: Optional[str] = None
    fallback_message: Optional[str] = None
    latency_ms: int = 0


class DataNotFoundError(Exception):
    """Raised when a tool's lookup yields no matching data — handled, not fatal."""


class BaseTool(ABC):
    """Async tool base class with structured logging + error handling."""

    name: str = "base_tool"
    description: str = ""

    def __init__(self) -> None:
        self.logger = get_logger(f"agent.tools.{self.name}")

    @abstractmethod
    async def _execute(self, **kwargs: Any) -> ToolResult:
        """Subclass-specific core logic. MUST return a ToolResult."""

    def _no_data_message(self) -> str:
        """Override per tool for tailored fallback wording."""
        return (
            "I don't have the data needed to answer that. "
            "Please contact your RoadGuard adjuster for an official answer."
        )

    async def run(self, **kwargs: Any) -> ToolResult:
        """Public entry point — wraps _execute with timing + uniform errors.

        WHY structured ToolResult on every path: the orchestrator can decide
        whether to retry, fall back, or surface a safe message — instead of
        letting exceptions bubble up to the user as 500 errors.
        """
        start = time.perf_counter()
        try:
            self.logger.info("tool_start", tool=self.name, kwargs=self._safe_kwargs(kwargs))
            result = await self._execute(**kwargs)
            result.latency_ms = int((time.perf_counter() - start) * 1000)
            result.tool_name = self.name
            self.logger.info(
                "tool_success",
                tool=self.name,
                latency_ms=result.latency_ms,
                citations=len(result.citations),
            )
            return result
        except DataNotFoundError as e:
            latency = int((time.perf_counter() - start) * 1000)
            self.logger.warning("tool_no_data", tool=self.name, reason=str(e), latency_ms=latency)
            return ToolResult(
                tool_name=self.name,
                success=False,
                error=str(e),
                fallback_message=self._no_data_message(),
                latency_ms=latency,
            )
        except Exception as e:
            latency = int((time.perf_counter() - start) * 1000)
            self.logger.error(
                "tool_error", tool=self.name, error=str(e), latency_ms=latency, exc_info=True
            )
            return ToolResult(
                tool_name=self.name,
                success=False,
                error="Internal error",
                fallback_message=(
                    "Something went wrong on our side. Please try again or "
                    "contact your adjuster directly."
                ),
                latency_ms=latency,
            )

    @staticmethod
    def _safe_kwargs(kwargs: Dict[str, Any]) -> Dict[str, Any]:
        """Strip noisy or sensitive kwargs from logs."""
        return {k: v for k, v in kwargs.items() if k not in {"history", "session_context"}}
