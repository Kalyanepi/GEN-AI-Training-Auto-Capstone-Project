"""Response schemas — public-facing API contract."""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class CitationOut(BaseModel):
    document: str
    section: Optional[str] = None
    page: Optional[int] = None
    excerpt: str
    relevance_score: float
    chunk_id: str
    source_type: str  # "pdf" | "csv"


class ToolResultOut(BaseModel):
    tool_name: str
    success: bool
    latency_ms: int = 0
    error: Optional[str] = None


class ChatResponse(BaseModel):
    session_id: str
    answer: str
    intent_detected: Optional[str] = None
    tools_used: List[str] = Field(default_factory=list)
    tool_results: List[ToolResultOut] = Field(default_factory=list)
    citations: List[CitationOut] = Field(default_factory=list)
    guardrail_triggered: bool = False
    guardrail_reason: Optional[str] = None
    disclaimer: Optional[str] = None
    trace_url: Optional[str] = None
    trace_id: Optional[str] = None
    latency_ms: int = 0
    calculation_breakdown: Optional[str] = None  # surfaced when total_loss_tool fires
    confidence_score: Optional[float] = None     # 0.0 - 1.0 heuristic groundedness score


class HealthResponse(BaseModel):
    status: str
    faiss_loaded: bool
    chunk_count: int = 0
    repair_cost_rows: int = 0
    total_loss_rows: int = 0


class ReadyResponse(BaseModel):
    status: str
    sessions_active: int
    chunk_count: int


class ErrorResponse(BaseModel):
    error: str
    detail: Optional[str] = None
    retry_after: Optional[int] = None
