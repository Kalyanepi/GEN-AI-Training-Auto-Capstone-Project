"""LangGraph orchestrator — the typed stateful graph executing every query.

Node flow (per architecture plan §4.3):
  START -> input_guardrail -> [block? -> guardrail_response -> END]
                          \-> intent_router -> tool_execution
                                           \-> [OUT_OF_SCOPE -> guardrail_response -> END]
        -> llm_synthesis -> output_guardrail -> [block? -> guardrail_response -> END]
                                            \-> memory_update -> END

WHY LangGraph over chains: typed state passing + conditional edges + visualizable
graph make multi-tool flows debuggable. Every node mutates AgentState and the
final state contains everything needed to build the API response.
"""
from __future__ import annotations

import time
import uuid
from typing import Any, Dict, List, Optional

from langgraph.graph import END, StateGraph
from openai import AsyncOpenAI

from agent.memory import get_session_store
from agent.prompts.system_prompt import SYSTEM_PROMPT
from agent.prompts.tool_prompts import build_synthesis_messages
from agent.router import IntentRouter
from agent.state import AgentState
from agent.tools.base_tool import BaseTool, ToolResult
from agent.tools.coverage_identifier_tool import CoverageIdentifierTool
from agent.tools.fnol_guide_tool import FnolGuideTool
from agent.tools.policy_rag_tool import PolicyRagTool
from agent.tools.rental_lookup_tool import RentalLookupTool
from agent.tools.repair_cost_tool import RepairCostTool
from agent.tools.roadside_tool import RoadsideTool
from agent.tools.total_loss_tool import TotalLossTool
from agent.tools.um_uim_tool import UmUimTool
from api.config import settings
from guardrails.input_guardrails import check_input
from guardrails.output_guardrails import check_output
from observability.logger import get_logger
from rag.citation_tracker import Citation

logger = get_logger(__name__)


def _build_tool_registry() -> Dict[str, BaseTool]:
    """Instantiate all tools once. WHY: tools hold loaded CSV/FAISS handles."""
    return {
        "policy_rag_tool": PolicyRagTool(),
        "repair_cost_tool": RepairCostTool(),
        "total_loss_tool": TotalLossTool(),
        "fnol_guide_tool": FnolGuideTool(),
        "rental_lookup_tool": RentalLookupTool(),
        "roadside_tool": RoadsideTool(),
        "um_uim_tool": UmUimTool(),
        "coverage_identifier_tool": CoverageIdentifierTool(),
    }


class Orchestrator:
    """Compiled LangGraph wrapper exposing an async `invoke` method."""

    def __init__(self) -> None:
        self.client = AsyncOpenAI(api_key=settings.openai_api_key)
        self.router = IntentRouter(client=self.client)
        self.tools = _build_tool_registry()
        self.session_store = get_session_store()
        self.graph = self._build_graph()

    # ----------------- Graph nodes -----------------

    async def _node_input_guardrail(self, state: AgentState) -> AgentState:
        decision = await check_input(state["user_message"], self.client)
        if decision.blocked:
            state["input_guardrail_triggered"] = True
            state["guardrail_reason"] = decision.reason
            state["guardrail_message"] = decision.message
            state["final_answer"] = decision.message
        else:
            state["input_guardrail_triggered"] = False
        return state

    async def _node_intent_router(self, state: AgentState) -> AgentState:
        decision = await self.router.classify(state["user_message"])
        state["detected_intent"] = decision.intent
        state["router_reasoning"] = decision.reasoning
        state["tools_to_invoke"] = list(decision.tools)
        return state

    async def _node_tool_execution(self, state: AgentState) -> AgentState:
        """Run each requested tool sequentially.

        WHY sequential (not parallel): tool outputs may inform later tools'
        kwargs in a future iteration; sequential keeps the graph predictable.
        For Phase 2's tool set, parallel would shave only ~100ms.
        """
        tool_results: List[Dict[str, Any]] = state.get("tool_results") or []
        all_citations: List[Citation] = list(state.get("citations") or [])
        allowed_dollars: List[float] = list(state.get("allowed_dollar_values") or [])

        kwargs_template = self._build_tool_kwargs(state)

        for tool_name in state.get("tools_to_invoke") or []:
            tool = self.tools.get(tool_name)
            if tool is None:
                logger.warning("unknown_tool_skipped", tool=tool_name)
                continue
            result: ToolResult = await tool.run(**kwargs_template)
            tool_results.append({
                "tool_name": result.tool_name,
                "success": result.success,
                "data": result.data,
                "error": result.error,
                "fallback_message": result.fallback_message,
                "latency_ms": result.latency_ms,
            })
            all_citations.extend(result.citations)
            allowed_dollars.extend(result.dollar_values)

        state["tool_results"] = tool_results
        state["citations"] = all_citations
        state["allowed_dollar_values"] = allowed_dollars
        return state

    async def _node_llm_synthesis(self, state: AgentState) -> AgentState:
        """Synthesize the final grounded answer from tool results + chunks."""
        tool_results = state.get("tool_results") or []
        any_data = any(t.get("success") for t in tool_results)

        # WHY anti-hallucination short-circuit: if no tool produced data and
        # we have no retrieved chunks, the LLM has nothing grounded to say.
        # Return the canonical "no data" message instead of risking fabrication.
        if not any_data:
            state["final_answer"] = (
                "I don't have specific policy language on that in my documents. "
                f"Please contact your RoadGuard adjuster at {settings.adjuster_phone} "
                "for an official answer."
            )
            return state

        session_context = {
            "policy_tier": state.get("policy_tier"),
            "vehicle_category": state.get("vehicle_category"),
            "state_code": state.get("state_code"),
            "coverage_type": state.get("coverage_type"),
            "vehicle_year": state.get("vehicle_year"),
            "acv": state.get("acv"),
            "repair_cost": state.get("repair_cost"),
        }
        messages = build_synthesis_messages(
            system_prompt=SYSTEM_PROMPT,
            user_message=state["user_message"],
            session_context=session_context,
            tool_results=tool_results,
            history=state.get("conversation_history"),
        )

        try:
            resp = await self.client.chat.completions.create(
                model=settings.openai_synthesis_model,
                messages=messages,
                temperature=0.2,
                max_tokens=900,
            )
            answer = (resp.choices[0].message.content or "").strip()
            state["final_answer"] = answer or self._no_answer_fallback()
        except Exception as e:
            logger.error("synthesis_failed", error=str(e), exc_info=True)
            state["final_answer"] = self._no_answer_fallback()
        return state

    def _no_answer_fallback(self) -> str:
        return (
            "I'm having trouble producing an answer right now. "
            f"Please try again in a moment or contact your adjuster at {settings.adjuster_phone}."
        )

    async def _node_output_guardrail(self, state: AgentState) -> AgentState:
        answer = state.get("final_answer") or ""
        citations = state.get("citations") or []
        allowed = state.get("allowed_dollar_values") or []
        decision = check_output(answer, citations, allowed)
        if decision.blocked:
            state["output_guardrail_triggered"] = True
            state["guardrail_reason"] = decision.reason
            state["guardrail_message"] = decision.message
            state["final_answer"] = decision.message
            # WHY clear citations on block: a refused answer should not be
            # accompanied by citations as if it were a substantive response.
            state["citations"] = []
        else:
            state["output_guardrail_triggered"] = False
            state["disclaimer"] = (
                "Official determination is made by your assigned RoadGuard adjuster."
            )
        return state

    async def _node_guardrail_response(self, state: AgentState) -> AgentState:
        """Terminal node for blocked queries — final_answer already set."""
        if not state.get("final_answer"):
            state["final_answer"] = (
                "I can only help with auto insurance topics. "
                f"Contact RoadGuard Claims: {settings.adjuster_phone}."
            )
        state["citations"] = []
        return state

    async def _node_memory_update(self, state: AgentState) -> AgentState:
        sid = state["session_id"]
        # Update structured facts.
        self.session_store.update_facts(
            sid,
            policy_tier=state.get("policy_tier"),
            vehicle_category=state.get("vehicle_category"),
            vehicle_year=state.get("vehicle_year"),
            state_code=state.get("state_code"),
            coverage_type=state.get("coverage_type"),
            acv=state.get("acv"),
            repair_cost=state.get("repair_cost"),
            last_intent=state.get("detected_intent"),
        )
        # Append turn.
        self.session_store.append_turn(sid, "user", state["user_message"])
        if state.get("final_answer"):
            self.session_store.append_turn(sid, "assistant", state["final_answer"])
        return state

    # ----------------- Routing helpers -----------------

    def _route_after_input_guardrail(self, state: AgentState) -> str:
        return "guardrail_response" if state.get("input_guardrail_triggered") else "intent_router"

    def _route_after_intent(self, state: AgentState) -> str:
        if state.get("detected_intent") == "OUT_OF_SCOPE" or not state.get("tools_to_invoke"):
            if state.get("detected_intent") == "OUT_OF_SCOPE":
                state["guardrail_reason"] = "OUT_OF_SCOPE"
                state["final_answer"] = (
                    "I can only help with auto insurance topics: coverage, claims, "
                    "repair estimates, total loss, FNOL, rental, and roadside."
                )
                return "guardrail_response"
        return "tool_execution"

    def _route_after_output_guardrail(self, state: AgentState) -> str:
        return "guardrail_response" if state.get("output_guardrail_triggered") else "memory_update"

    # ----------------- Tool kwarg builder -----------------

    @staticmethod
    def _build_tool_kwargs(state: AgentState) -> Dict[str, Any]:
        """Common kwargs every tool may consume. Extra kwargs are ignored by **_."""
        return {
            "query": state["user_message"],
            "incident_description": state["user_message"],
            "damage_type": state["user_message"],   # repair_cost_tool fuzzy-matches
            "policy_tier": state.get("policy_tier"),
            "coverage_type": state.get("coverage_type"),
            "vehicle_category": state.get("vehicle_category"),
            "vehicle_year": state.get("vehicle_year"),
            "state_code": state.get("state_code"),
            "acv": state.get("acv"),
            "repair_cost": state.get("repair_cost"),
        }

    # ----------------- Graph construction -----------------

    def _build_graph(self):
        graph = StateGraph(AgentState)
        graph.add_node("input_guardrail", self._node_input_guardrail)
        graph.add_node("intent_router", self._node_intent_router)
        graph.add_node("tool_execution", self._node_tool_execution)
        graph.add_node("llm_synthesis", self._node_llm_synthesis)
        graph.add_node("output_guardrail", self._node_output_guardrail)
        graph.add_node("guardrail_response", self._node_guardrail_response)
        graph.add_node("memory_update", self._node_memory_update)

        graph.set_entry_point("input_guardrail")
        graph.add_conditional_edges(
            "input_guardrail",
            self._route_after_input_guardrail,
            {"guardrail_response": "guardrail_response", "intent_router": "intent_router"},
        )
        graph.add_conditional_edges(
            "intent_router",
            self._route_after_intent,
            {"guardrail_response": "guardrail_response", "tool_execution": "tool_execution"},
        )
        graph.add_edge("tool_execution", "llm_synthesis")
        graph.add_edge("llm_synthesis", "output_guardrail")
        graph.add_conditional_edges(
            "output_guardrail",
            self._route_after_output_guardrail,
            {"guardrail_response": "guardrail_response", "memory_update": "memory_update"},
        )
        graph.add_edge("guardrail_response", END)
        graph.add_edge("memory_update", END)
        return graph.compile()

    # ----------------- Public API -----------------

    async def invoke(self, initial_state: AgentState) -> AgentState:
        """Execute the graph and return the final state."""
        start = time.perf_counter()
        trace_id = str(uuid.uuid4())
        initial_state.setdefault("trace_id", trace_id)
        initial_state.setdefault("tool_results", [])
        initial_state.setdefault("citations", [])
        initial_state.setdefault("allowed_dollar_values", [])
        # Hydrate conversation history from session store.
        sid = initial_state.get("session_id")
        if sid:
            initial_state["conversation_history"] = self.session_store.get_history(sid)
        try:
            final = await self.graph.ainvoke(initial_state)
        except Exception as e:
            logger.error("graph_invoke_failed", error=str(e), exc_info=True)
            initial_state["final_answer"] = self._no_answer_fallback()
            initial_state["citations"] = []
            final = initial_state
        final["latency_ms"] = int((time.perf_counter() - start) * 1000)
        return final


# Module-level singleton (built lazily).
_orchestrator: Optional[Orchestrator] = None


def get_orchestrator() -> Orchestrator:
    global _orchestrator
    if _orchestrator is None:
        _orchestrator = Orchestrator()
    return _orchestrator
