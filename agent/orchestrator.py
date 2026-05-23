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

import asyncio
import time
import uuid
from typing import Any, Dict, List, Optional

from langgraph.graph import END, StateGraph
from openai import AsyncOpenAI

from agent.memory import get_session_store
from agent.param_extractor import (
    extract_damage_type,
    extract_damage_types,
    extract_vehicle_category,
)
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


# WHY deterministic tool order: the synthesis LLM reads tool results in the
# order they were appended. Putting grounded RAG context BEFORE structured
# CSV results reduces hallucination — the model anchors on policy language
# first, then fits numeric facts (deductibles, thresholds) on top. Without a
# fixed order, the router's free-form `tools` array introduces variance and
# occasional fabrication on multi-tool queries.
TOOL_EXECUTION_ORDER: List[str] = [
    "coverage_identifier_tool",   # cheap classifier — informs downstream context
    "policy_rag_tool",            # grounded policy language first
    "fnol_guide_tool",            # procedural grounding
    "um_uim_tool",                # tier-aware UM/UIM
    "rental_lookup_tool",         # tier-aware structured lookup
    "roadside_tool",              # tier-aware structured lookup
    "repair_cost_tool",           # CSV numeric lookup
    "total_loss_tool",            # CSV numeric calc (often depends on repair)
]


def _sort_tools_deterministically(tool_names: List[str]) -> List[str]:
    """Return tool_names sorted by TOOL_EXECUTION_ORDER; unknown tools go last."""
    priority = {name: idx for idx, name in enumerate(TOOL_EXECUTION_ORDER)}
    return sorted(tool_names, key=lambda t: priority.get(t, len(priority)))


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

    async def _node_preflight(self, state: AgentState) -> AgentState:
        """Run input guardrail AND intent router CONCURRENTLY.

        WHY merged: both nodes read the raw user_message and don't depend on
        each other. Running them in parallel via asyncio.gather saves the
        wall-time of the slower of the two (~400ms typical).

        WHY we still apply guardrail first in conditional logic: if input is
        blocked the router decision is discarded, but starting the router
        early costs nothing extra — we already paid for the parallel run.
        """
        guard_task = asyncio.create_task(check_input(state["user_message"], self.client))
        route_task = asyncio.create_task(self.router.classify(state["user_message"]))
        guard_decision, route_decision = await asyncio.gather(guard_task, route_task)

        # 1. Apply input guardrail first — if blocked, router result is ignored.
        if guard_decision.blocked:
            state["input_guardrail_triggered"] = True
            state["guardrail_reason"] = guard_decision.reason
            state["guardrail_message"] = guard_decision.message
            state["final_answer"] = guard_decision.message
            # Still capture router output for observability/logging.
            state["detected_intent"] = route_decision.intent
            state["router_reasoning"] = route_decision.reasoning
            state["tools_to_invoke"] = []
            return state

        state["input_guardrail_triggered"] = False
        state["detected_intent"] = route_decision.intent
        state["router_reasoning"] = route_decision.reasoning
        state["tools_to_invoke"] = list(route_decision.tools)

        # 2. Handle terminal intents inline (set final_answer before routing).
        if route_decision.intent == "OUT_OF_SCOPE":
            state["guardrail_reason"] = "OUT_OF_SCOPE"
            state["final_answer"] = (
                "I can only help with auto insurance topics: coverage, claims, "
                "repair estimates, total loss, FNOL, rental, and roadside. "
                f"For anything else, contact RoadGuard Claims: {settings.adjuster_phone}."
            )
        elif route_decision.intent == "CLARIFICATION_NEEDED":
            # WHY not a guardrail: clarification is a normal conversational
            # turn, not a block. We skip tools + synthesis and ask back.
            state["final_answer"] = route_decision.clarification_question or (
                "Could you give me a bit more detail about what you'd like to know?"
            )

        return state

    async def _node_tool_execution(self, state: AgentState) -> AgentState:
        """Run all requested tools CONCURRENTLY via asyncio.gather.

        WHY parallel is safe here: tools today are independent — none consumes
        another tool's output. Each reads from FAISS or CSV (both are loaded
        in memory; FAISS is thread-safe for reads, pandas DataFrames are
        copy-on-write). Parallel execution saves ~150-400ms on multi-tool
        intents (MULTI_INTENT, TOTAL_LOSS+policy_rag, etc.).

        WHY we still sort tools first: the synthesis LLM reads results in
        list order, so we preserve deterministic ordering for the final
        prompt even though network/CPU dispatch order is arbitrary.
        """
        tool_results: List[Dict[str, Any]] = state.get("tool_results") or []
        all_citations: List[Citation] = list(state.get("citations") or [])
        allowed_dollars: List[float] = list(state.get("allowed_dollar_values") or [])

        kwargs_template = self._build_tool_kwargs(state)

        # Enforce deterministic execution order in the *result list*. Dispatch
        # is parallel; results are collected in this fixed order regardless of
        # which task finished first.
        ordered_tools = _sort_tools_deterministically(state.get("tools_to_invoke") or [])
        state["tools_to_invoke"] = ordered_tools

        # Build the parallel task list, skipping unknown tools.
        tasks = []
        names = []
        for tool_name in ordered_tools:
            tool = self.tools.get(tool_name)
            if tool is None:
                logger.warning("unknown_tool_skipped", tool=tool_name)
                continue
            tasks.append(asyncio.create_task(tool.run(**kwargs_template)))
            names.append(tool_name)

        results: List[ToolResult] = await asyncio.gather(*tasks) if tasks else []

        # Collect in deterministic (ordered_tools) order.
        for result in results:
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
            state["citations"] = []
        else:
            state["output_guardrail_triggered"] = False
            state["disclaimer"] = (
                "Official determination is made by your assigned RoadGuard adjuster."
            )
        state["confidence_score"] = self._compute_confidence(state)
        return state

    @staticmethod
    def _compute_confidence(state: AgentState) -> float:
        """Heuristic 0.0-1.0 confidence score.

        WHY a heuristic instead of an LLM self-rating: deterministic, free,
        and grounded in objective signals (tool success, citation strength,
        guardrail outcome). Self-rated confidence from LLMs is famously
        unreliable for this purpose.
        """
        # If any guardrail fired the response isn't a real answer.
        if state.get("input_guardrail_triggered") or state.get("output_guardrail_triggered"):
            return 0.0
        if state.get("detected_intent") == "OUT_OF_SCOPE":
            return 0.0

        score = 0.0
        tool_results = state.get("tool_results") or []
        if tool_results:
            success_ratio = sum(1 for t in tool_results if t.get("success")) / len(tool_results)
            score += 0.35 * success_ratio   # tools succeeded
        else:
            # No tool data at all — we have nothing to ground on.
            return 0.15

        citations = state.get("citations") or []
        if citations:
            score += 0.25                   # at least one citation
            top = max((c.relevance_score for c in citations), default=0.0)
            # Reward strong top-citation relevance (above the similarity floor).
            floor = settings.similarity_threshold
            if top >= floor:
                # Linearly scale 0.0..0.30 between floor and 1.0.
                normalized = min(1.0, (top - floor) / max(1e-6, 1.0 - floor))
                score += 0.30 * normalized

        # Synthesis actually produced a non-fallback answer.
        if state.get("final_answer") and "having trouble producing an answer" not in (
            state.get("final_answer") or ""
        ):
            score += 0.10

        return round(min(1.0, max(0.0, score)), 2)

    async def _node_guardrail_response(self, state: AgentState) -> AgentState:
        """Terminal node for blocked queries — final_answer already set by the
        node that triggered the block (input_guardrail, intent_router, or
        output_guardrail). This node is a no-op pass-through that exists so
        the graph has a named terminal node for blocked paths."""
        if not state.get("final_answer"):
            state["final_answer"] = (
                "I can only help with auto insurance topics. "
                f"Contact RoadGuard Claims: {settings.adjuster_phone}."
            )
        state["citations"] = []
        return state

    async def _node_memory_update(self, state: AgentState) -> AgentState:
        sid = state["session_id"]
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
        self.session_store.append_turn(sid, "user", state["user_message"])
        if state.get("final_answer"):
            self.session_store.append_turn(sid, "assistant", state["final_answer"])
        return state

    # ----------------- Routing helpers (pure — no state mutation) -----------------

    def _route_after_preflight(self, state: AgentState) -> str:
        """Decide where to go after the merged guardrail+router preflight."""
        if state.get("input_guardrail_triggered"):
            return "guardrail_response"
        intent = state.get("detected_intent")
        tools = state.get("tools_to_invoke") or []
        if intent == "CLARIFICATION_NEEDED":
            return "memory_update"
        if intent == "OUT_OF_SCOPE" or not tools:
            return "guardrail_response"
        return "tool_execution"

    def _route_after_output_guardrail(self, state: AgentState) -> str:
        return "guardrail_response" if state.get("output_guardrail_triggered") else "memory_update"

    # ----------------- Tool kwarg builder -----------------

    @staticmethod
    def _build_tool_kwargs(state: AgentState) -> Dict[str, Any]:
        """Common kwargs every tool may consume. Extra kwargs are ignored by **_.

        WHY extract_damage_type: repair_cost_tool's fuzzy matcher expects a
        short phrase like "rear bumper replacement", not a full sentence.
        Passing the raw user_message gives SequenceMatcher a ~5% ratio (well
        below the 0.60 threshold), causing every repair estimate to fail.
        extract_damage_type pulls out just the relevant damage phrase.

        WHY extract_vehicle_category: users say "Honda Civic" but the CSV
        uses categories like "Economy/Compact". The mapper bridges the gap
        without requiring the user to pick from a dropdown first. It only
        fires when the sidebar hasn't already set a category.

        WHY extract_damage_types (plural): real users describe multiple
        damages in one turn ("cracked headlight and hood damage"). Returning
        a list lets repair_cost_tool fan out per-damage lookups and the
        synthesis LLM aggregates the results into one coherent estimate.
        """
        user_msg = state["user_message"]

        # Damage extraction: prefer multi-damage list; fall back to single.
        damages = extract_damage_types(user_msg)
        if not damages:
            single = extract_damage_type(user_msg)
            damages = [single] if single else [user_msg]

        # Vehicle category: sidebar value wins; if unset, try to infer from
        # make/model in the user's message.
        vehicle_category = state.get("vehicle_category")
        if not vehicle_category:
            vehicle_category = extract_vehicle_category(user_msg)

        return {
            "query": user_msg,
            "incident_description": user_msg,
            "damage_type": damages[0] if damages else user_msg,
            "damage_types": damages,
            "policy_tier": state.get("policy_tier"),
            "coverage_type": state.get("coverage_type"),
            "vehicle_category": vehicle_category,
            "vehicle_year": state.get("vehicle_year"),
            "state_code": state.get("state_code"),
            "acv": state.get("acv"),
            "repair_cost": state.get("repair_cost"),
        }

    # ----------------- Graph construction -----------------

    def _build_graph(self):
        graph = StateGraph(AgentState)
        # Merged preflight: input guardrail + intent router run in parallel.
        graph.add_node("preflight", self._node_preflight)
        graph.add_node("tool_execution", self._node_tool_execution)
        graph.add_node("llm_synthesis", self._node_llm_synthesis)
        graph.add_node("output_guardrail", self._node_output_guardrail)
        graph.add_node("guardrail_response", self._node_guardrail_response)
        graph.add_node("memory_update", self._node_memory_update)

        graph.set_entry_point("preflight")
        graph.add_conditional_edges(
            "preflight",
            self._route_after_preflight,
            {
                "guardrail_response": "guardrail_response",
                "tool_execution": "tool_execution",
                "memory_update": "memory_update",   # CLARIFICATION_NEEDED bypass
            },
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
