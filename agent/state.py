"""AgentState — typed LangGraph state schema flowing through every node."""
from __future__ import annotations

from typing import Any, Dict, List, Optional, TypedDict

from rag.citation_tracker import Citation


class ToolInvocation(TypedDict, total=False):
    """One tool invocation result captured for the orchestrator and observability."""
    tool_name: str
    success: bool
    data: Optional[Dict[str, Any]]
    error: Optional[str]
    fallback_message: Optional[str]
    latency_ms: int


class AgentState(TypedDict, total=False):
    # --- Input ---
    session_id: str
    user_message: str
    policy_tier: Optional[str]
    coverage_type: Optional[str]
    vehicle_category: Optional[str]
    state_code: Optional[str]
    vehicle_year: Optional[int]
    acv: Optional[float]
    repair_cost: Optional[float]

    # --- Derived during execution ---
    detected_intent: Optional[str]
    router_reasoning: Optional[str]
    tools_to_invoke: List[str]
    tool_results: List[ToolInvocation]
    citations: List[Citation]
    allowed_dollar_values: List[float]

    # --- Guardrail flags ---
    input_guardrail_triggered: bool
    output_guardrail_triggered: bool
    guardrail_reason: Optional[str]
    guardrail_message: Optional[str]

    # --- Output ---
    final_answer: Optional[str]
    disclaimer: Optional[str]

    # --- Memory + observability ---
    conversation_history: List[Dict[str, str]]   # [{role, content}, ...]
    trace_id: Optional[str]
    latency_ms: Optional[int]
