"""Core chat business logic — invokes the orchestrator and assembles the response.

WHY a dedicated service layer: the route handler stays a thin HTTP wrapper.
Business logic (state hydration, response shaping, trace URL building) lives
here and is independently testable.
"""
from __future__ import annotations

from typing import Any, Dict, List

from agent.orchestrator import Orchestrator
from agent.param_extractor import extract_params
from agent.state import AgentState
from api.schemas.request import ChatRequest
from api.schemas.response import ChatResponse, CitationOut, ToolResultOut
from api.services.session_service import SessionService
from observability.langsmith_tracer import build_trace_url
from observability.logger import get_logger
from rag.citation_tracker import Citation

logger = get_logger(__name__)


class ChatService:
    def __init__(self, orchestrator: Orchestrator, session_service: SessionService) -> None:
        self.orchestrator = orchestrator
        self.session_service = session_service

    async def chat(self, request: ChatRequest) -> ChatResponse:
        # 1a. Extract structured params from the natural-language message
        # so users who type "ACV is $8,000 in Illinois" don't have to also
        # fill API/sidebar fields. Explicit request params still take precedence.
        extracted = extract_params(request.message)

        # 1b. Hydrate session context — request fields win over extraction.
        ctx = self.session_service.hydrate_context(
            session_id=request.session_id,
            policy_tier=request.policy_tier,
            coverage_type=request.coverage_type,
            vehicle_category=request.vehicle_category,
            state_code=request.state_code or extracted.state_code,
            vehicle_year=request.vehicle_year,
            acv=request.acv if request.acv is not None else extracted.acv,
            repair_cost=request.repair_cost if request.repair_cost is not None else extracted.repair_cost,
        )

        # 2. Build initial AgentState (use session-resolved facts so follow-up
        #    questions inherit prior tier/state/vehicle without re-asking).
        initial: AgentState = {
            "session_id": request.session_id,
            "user_message": request.message,
            "policy_tier": ctx.policy_tier,
            "coverage_type": ctx.coverage_type,
            "vehicle_category": ctx.vehicle_category,
            "state_code": ctx.state_code,
            "vehicle_year": ctx.vehicle_year,
            "acv": ctx.acv,
            "repair_cost": ctx.repair_cost,
        }

        # 3. Invoke graph.
        final = await self.orchestrator.invoke(initial)

        # 4. Shape response.
        citations_out = self._citations_to_out(final.get("citations") or [])
        tool_results_out = [
            ToolResultOut(
                tool_name=t.get("tool_name", "unknown"),
                success=bool(t.get("success")),
                latency_ms=int(t.get("latency_ms") or 0),
                error=t.get("error"),
            )
            for t in (final.get("tool_results") or [])
        ]
        tools_used = [t.tool_name for t in tool_results_out if t.success] or [
            t.tool_name for t in tool_results_out
        ]

        calculation_breakdown = self._extract_breakdown(final.get("tool_results") or [])

        return ChatResponse(
            session_id=request.session_id,
            answer=final.get("final_answer") or "",
            intent_detected=final.get("detected_intent"),
            tools_used=tools_used,
            tool_results=tool_results_out,
            citations=citations_out,
            guardrail_triggered=bool(
                final.get("input_guardrail_triggered") or final.get("output_guardrail_triggered")
            ),
            guardrail_reason=final.get("guardrail_reason"),
            disclaimer=final.get("disclaimer"),
            trace_id=final.get("trace_id"),
            trace_url=build_trace_url(final.get("trace_id")),
            latency_ms=int(final.get("latency_ms") or 0),
            calculation_breakdown=calculation_breakdown,
            confidence_score=final.get("confidence_score"),
        )

    @staticmethod
    def _citations_to_out(citations: List[Citation]) -> List[CitationOut]:
        return [
            CitationOut(
                document=c.document,
                section=c.section,
                page=c.page,
                excerpt=c.excerpt,
                relevance_score=c.relevance_score,
                chunk_id=c.chunk_id,
                source_type=c.source_type,
            )
            for c in citations
        ]

    @staticmethod
    def _extract_breakdown(tool_results: List[Dict[str, Any]]) -> str | None:
        """Surface total_loss_tool's calculation breakdown to the UI."""
        for t in tool_results:
            if t.get("tool_name") == "total_loss_tool" and t.get("success"):
                return (t.get("data") or {}).get("calculation_breakdown")
        return None
