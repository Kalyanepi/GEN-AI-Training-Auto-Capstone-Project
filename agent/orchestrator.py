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
import json
import time
import uuid
from typing import Any, AsyncGenerator, Dict, List, Optional

from langgraph.graph import END, StateGraph
from openai import AsyncOpenAI

from agent.memory import get_session_store
from agent.param_extractor import (
    extract_all_state_codes,
    extract_damage_type,
    extract_damage_types,
    extract_params,
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
from guardrails.input_guardrails import check_input, _check_greeting
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
        t0 = time.perf_counter()
        msg = state["user_message"]
        logger.info("preflight_start", message=msg[:50])

        # Fast path: greetings bypass both LLM calls entirely — zero latency.
        if _check_greeting(msg):
            state["input_guardrail_triggered"] = False
            state["detected_intent"] = "GREETING"
            state["router_reasoning"] = "greeting bypass"
            state["tools_to_invoke"] = []
            state["final_answer"] = (
                "Hi there! I'm **Auto Insurance AI Copilot**. "
                "I can help you with policy coverage questions, repair estimates, "
                "total loss calculations, filing a claim (FNOL), rental limits, "
                "roadside assistance, and more.\n\n"
                "What can I help you with today?"
            )
            return state

        # Total-loss clarification fast path: when the previous turn asked
        # for a missing total-loss param, handle the follow-up here instead
        # of letting the router re-classify a bare answer like "4000$".
        #
        # WHY bypass router here: the router only sees the raw message and
        # session memory is invisible to it. "4000$" with no context looks
        # like CLARIFICATION_NEEDED to the router even if repair_cost is
        # already known. We resolve the next missing param directly.
        last_intent = state.get("last_intent")
        if last_intent == "CLARIFICATION_NEEDED":
            # Also extract from current message to catch values the session
            # may have missed (e.g. after a server reload wiped memory).
            _ex = extract_params(msg,
                missing_acv=not bool(state.get("acv")),
                missing_repair=not bool(state.get("repair_cost")),
            )
            has_acv = bool(state.get("acv") or _ex.acv)
            has_repair = bool(state.get("repair_cost") or _ex.repair_cost)
            has_state = bool(state.get("state_code") or _ex.state_code)
            # Merge any newly extracted values into state so tool_execution uses them.
            if _ex.acv and not state.get("acv"):
                state["acv"] = _ex.acv
            if _ex.repair_cost and not state.get("repair_cost"):
                state["repair_cost"] = _ex.repair_cost
            if _ex.state_code and not state.get("state_code"):
                state["state_code"] = _ex.state_code
            if has_acv and has_repair and has_state:
                # All params present — run the calculation.
                state["input_guardrail_triggered"] = False
                state["detected_intent"] = "TOTAL_LOSS"
                state["router_reasoning"] = "clarification complete — all total-loss params present"
                state["tools_to_invoke"] = ["total_loss_tool", "policy_rag_tool"]
                return state
            if has_acv or has_repair:
                # At least one numeric param known — we're in a total-loss
                # clarification chain. Ask for exactly what's still missing.
                if not has_state and has_acv and has_repair:
                    q = "Which state is the vehicle registered in?"
                elif not has_acv:
                    q = "What is the actual cash value (ACV) of your vehicle?"
                elif not has_repair:
                    q = "What is the estimated repair cost for your vehicle?"
                else:
                    q = "Which state is the vehicle registered in?"
                state["input_guardrail_triggered"] = False
                state["detected_intent"] = "CLARIFICATION_NEEDED"
                state["router_reasoning"] = "total-loss clarification chain — asking for next missing param"
                state["tools_to_invoke"] = []
                state["final_answer"] = q
                return state

        # Normal path: run guardrail + router in parallel.
        history = state.get("conversation_history") or []
        guard_task = asyncio.create_task(check_input(msg, self.client, last_intent))
        route_task = asyncio.create_task(self.router.classify(msg, history))
        guard_decision, route_decision = await asyncio.gather(guard_task, route_task)

        # 1. Apply input guardrail first — if blocked, router result is ignored.
        if guard_decision.blocked:
            state["input_guardrail_triggered"] = True
            state["guardrail_reason"] = guard_decision.reason
            state["guardrail_message"] = guard_decision.message
            state["final_answer"] = guard_decision.message
            state["detected_intent"] = route_decision.intent
            state["router_reasoning"] = route_decision.reasoning
            state["tools_to_invoke"] = []
            return state

        state["input_guardrail_triggered"] = False
        state["detected_intent"] = route_decision.intent
        state["router_reasoning"] = route_decision.reasoning
        state["tools_to_invoke"] = list(route_decision.tools)

        # 2. Handle terminal intents inline (set final_answer before routing).
        if state["detected_intent"] == "GREETING":
            state["final_answer"] = (
                "Hi there! I'm **Auto Insurance AI Copilot**. "
                "I can help you with policy coverage questions, repair estimates, "
                "total loss calculations, filing a claim (FNOL), rental limits, "
                "roadside assistance, and more.\n\n"
                "What can I help you with today?"
            )
        elif state["detected_intent"] == "OUT_OF_SCOPE":
            state["guardrail_reason"] = "OUT_OF_SCOPE"
            state["final_answer"] = (
                "I can only help with auto insurance topics: coverage, claims, "
                "repair estimates, total loss, FNOL, rental, and roadside."
            )
        elif state["detected_intent"] == "CLARIFICATION_NEEDED":
            # WHY not a guardrail: clarification is a normal conversational
            # turn, not a block. We skip tools + synthesis and ask back.
            state["final_answer"] = route_decision.clarification_question or (
                "Could you give me a bit more detail about what you'd like to know?"
            )

        logger.info("preflight_end", latency_ms=int((time.perf_counter() - t0) * 1000))
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
        t0 = time.perf_counter()
        tools_to_run = state.get("tools_to_invoke") or []
        logger.info("tools_start", tools=tools_to_run)

        tool_results: List[Dict[str, Any]] = state.get("tool_results") or []
        all_citations: List[Citation] = list(state.get("citations") or [])
        allowed_dollars: List[float] = list(state.get("allowed_dollar_values") or [])

        kwargs_template = self._build_tool_kwargs(state)

        # Enforce deterministic execution order in the *result list*. Dispatch
        # is parallel; results are collected in this fixed order regardless of
        # which task finished first.
        ordered_tools = _sort_tools_deterministically(state.get("tools_to_invoke") or [])
        state["tools_to_invoke"] = ordered_tools

        # For total_loss_tool, fan out one task per state found in the message.
        # WHY: multi-state comparison queries ("Florida vs Texas threshold") need
        # a separate CSV lookup per state — a single call only returns one state.
        user_msg = state.get("user_message", "")
        all_states = extract_all_state_codes(user_msg)

        # Build the parallel task list, skipping unknown tools.
        tasks = []
        names = []
        for tool_name in ordered_tools:
            tool = self.tools.get(tool_name)
            if tool is None:
                logger.warning("unknown_tool_skipped", tool=tool_name)
                continue
            if tool_name == "total_loss_tool" and len(all_states) > 1:
                # Fan out one task per state, each with its own state_code.
                for sc in all_states:
                    kw = {**kwargs_template, "state_code": sc}
                    tasks.append(asyncio.create_task(tool.run(**kw)))
                    names.append(tool_name)
            else:
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

        # WHY: user-provided ACV and repair cost get echoed back in the synthesis
        # answer ("Repair Costs: $3,800 / ACV: $4,500"). The fabricated-cost
        # guardrail checks every dollar figure against allowed_dollar_values, so
        # if those inputs aren't in the list the answer gets blocked as fabricated.
        acv_val = kwargs_template.get("acv")
        repair_val = kwargs_template.get("repair_cost")
        if acv_val and acv_val > 0:
            allowed_dollars.append(float(acv_val))
        if repair_val and repair_val > 0:
            allowed_dollars.append(float(repair_val))

        state["tool_results"] = tool_results
        state["citations"] = all_citations
        state["allowed_dollar_values"] = allowed_dollars
        logger.info("tools_end", latency_ms=int((time.perf_counter() - t0) * 1000))
        return state

    async def _node_llm_synthesis(self, state: AgentState) -> AgentState:
        """Synthesize the final grounded answer from tool results + chunks."""
        # Skip synthesis if preflight already produced a final answer
        # (GREETING, CLARIFICATION_NEEDED, OUT_OF_SCOPE all set it directly).
        if state.get("final_answer"):
            return state

        t0 = time.perf_counter()
        logger.info("synthesis_start", intent=state.get("detected_intent"), model=settings.openai_synthesis_model)

        tool_results = state.get("tool_results") or []
        any_data = any(t.get("success") for t in tool_results)

        if not any_data:
            state["final_answer"] = (
                "I don't have specific policy language on that in my documents. "
                "Please try rephrasing your question or ask about a specific coverage topic."
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
                max_tokens=500,  # Reduced from 900 for faster responses (~1-2s savings)
            )
            answer = (resp.choices[0].message.content or "").strip()
            state["final_answer"] = answer or self._no_answer_fallback()
        except Exception as e:
            logger.error("synthesis_failed", error=str(e), exc_info=True)
            state["final_answer"] = self._no_answer_fallback()
        logger.info("synthesis_end", latency_ms=int((time.perf_counter() - t0) * 1000))
        return state

    def _no_answer_fallback(self) -> str:
        return (
            "I'm having trouble producing an answer right now. "
            f"Please try again in a moment or contact your adjuster at {settings.adjuster_phone}."
        )

    async def _node_output_guardrail(self, state: AgentState) -> AgentState:
        # Terminal intents (GREETING, CLARIFICATION_NEEDED, OUT_OF_SCOPE) produce
        # hard-coded answers that don't contain LLM-generated content — skip all
        # output guardrail checks to avoid false positives (e.g. "limits" in a
        # greeting triggering the citation-required check).
        terminal_intents = {"GREETING", "CLARIFICATION_NEEDED", "OUT_OF_SCOPE"}
        if state.get("detected_intent") in terminal_intents:
            state["output_guardrail_triggered"] = False
            state["disclaimer"] = None
            state["confidence_score"] = self._compute_confidence(state)
            return state

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
            # WHY: check_output may have sanitized the answer (e.g. stripped
            # hallucinated phone numbers). Use the cleaned text if returned.
            if decision.message:
                state["final_answer"] = decision.message
            state["disclaimer"] = (
                "Official determination is made by your assigned adjuster."
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
                "I can only help with auto insurance topics: coverage, claims, "
                "repair estimates, total loss, FNOL, rental, and roadside."
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
        if intent in ("CLARIFICATION_NEEDED", "GREETING"):
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

        # WHY extract_params here: users naturally embed state/ACV/repair in
        # free text ("ACV $8k, repair $6.5k in Illinois"). Session state only
        # has these if the sidebar set them or a prior turn captured them.
        # Regex extraction costs zero latency and fills the gap.
        extracted = extract_params(user_msg)
        state_code = state.get("state_code") or extracted.state_code
        acv = state.get("acv") or extracted.acv
        repair_cost = state.get("repair_cost") or extracted.repair_cost

        return {
            "query": user_msg,
            "incident_description": user_msg,
            "damage_type": damages[0] if damages else user_msg,
            "damage_types": damages,
            "policy_tier": state.get("policy_tier"),
            "coverage_type": state.get("coverage_type"),
            "vehicle_category": vehicle_category,
            "vehicle_year": state.get("vehicle_year"),
            "state_code": state_code,
            "acv": acv,
            "repair_cost": repair_cost,
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

    def _prepare_state(self, initial_state: AgentState) -> AgentState:
        """Seed defaults before graph/stream execution."""
        trace_id = str(uuid.uuid4())
        initial_state.setdefault("trace_id", trace_id)
        initial_state.setdefault("tool_results", [])
        initial_state.setdefault("citations", [])
        initial_state.setdefault("allowed_dollar_values", [])
        sid = initial_state.get("session_id")
        if sid:
            initial_state["conversation_history"] = self.session_store.get_history(sid)
        return initial_state

    async def invoke(self, initial_state: AgentState) -> AgentState:
        """Execute the graph and return the final state."""
        start = time.perf_counter()
        self._prepare_state(initial_state)
        try:
            final = await self.graph.ainvoke(initial_state)
        except Exception as e:
            logger.error("graph_invoke_failed", error=str(e), exc_info=True)
            initial_state["final_answer"] = self._no_answer_fallback()
            initial_state["citations"] = []
            final = initial_state
        final["latency_ms"] = int((time.perf_counter() - start) * 1000)
        return final

    async def stream_invoke(self, initial_state: AgentState) -> AsyncGenerator[str, None]:
        """Run full pipeline; stream LLM synthesis tokens as SSE.

        Yields SSE-formatted strings:
          - One 'meta' event with intent/tools/citations JSON (before tokens).
          - Multiple 'token' events, one per streamed LLM chunk.
          - One 'done' event with final metadata (latency, confidence, disclaimer).
        """
        start = time.perf_counter()
        self._prepare_state(initial_state)

        # Run all nodes up to (but not including) llm_synthesis.
        try:
            state = await self._node_preflight(initial_state)
        except Exception as e:
            logger.error("stream_preflight_failed", error=str(e), exc_info=True)
            yield f"event: token\ndata: {json.dumps({'token': self._no_answer_fallback()})}\n\n"
            yield "event: done\ndata: {}\n\n"
            return

        # Guardrail block at input stage.
        if state.get("input_guardrail_triggered"):
            msg = state.get("guardrail_message") or state.get("final_answer") or ""
            meta = {
                "intent": state.get("detected_intent"),
                "tools": [],
                "citations": [],
                "guardrail_triggered": True,
                "guardrail_reason": state.get("guardrail_reason"),
            }
            yield f"event: meta\ndata: {json.dumps(meta)}\n\n"
            yield f"event: token\ndata: {json.dumps({'token': msg})}\n\n"
            yield "event: done\ndata: {}\n\n"
            return

        # Tool execution (guardrail passed or CLARIFICATION_NEEDED bypass).
        route = self._route_after_preflight(state)
        if route == "tool_execution":
            try:
                state = await self._node_tool_execution(state)
            except Exception as e:
                logger.error("stream_tool_exec_failed", error=str(e), exc_info=True)

        # Emit meta event now — client can render intent/citations immediately.
        from rag.citation_tracker import Citation
        raw_citations = state.get("citations") or []
        citations_json = [
            {
                "document": c.document if isinstance(c, Citation) else c.get("document", ""),
                "section": c.section if isinstance(c, Citation) else c.get("section"),
                "page": c.page if isinstance(c, Citation) else c.get("page"),
                "excerpt": c.excerpt if isinstance(c, Citation) else c.get("excerpt", ""),
                "relevance_score": c.relevance_score if isinstance(c, Citation) else c.get("relevance_score", 0),
                "chunk_id": c.chunk_id if isinstance(c, Citation) else c.get("chunk_id", ""),
                "source_type": c.source_type if isinstance(c, Citation) else c.get("source_type", "pdf"),
            }
            for c in raw_citations
        ]
        tool_results = state.get("tool_results") or []
        tools_used = [t["tool_name"] for t in tool_results if t.get("success")]
        meta = {
            "intent": state.get("detected_intent"),
            "tools": tools_used,
            "citations": citations_json,
            "guardrail_triggered": False,
            "session_id": state.get("session_id"),
            "trace_id": state.get("trace_id"),
        }
        yield f"event: meta\ndata: {json.dumps(meta)}\n\n"

        # Stream LLM synthesis.
        tool_results_list = state.get("tool_results") or []
        any_data = any(t.get("success") for t in tool_results_list)
        full_answer = ""

        # If preflight already set a final answer (GREETING, CLARIFICATION_NEEDED,
        # OUT_OF_SCOPE), emit it directly — no synthesis needed.
        preflight_answer = state.get("final_answer") or ""
        if preflight_answer:
            full_answer = preflight_answer
            yield f"event: token\ndata: {json.dumps({'token': full_answer})}\n\n"
        elif not any_data:
            full_answer = (
                "I don't have specific policy language on that in my documents. "
                "Please try rephrasing your question or ask about a specific coverage topic."
            )
            yield f"event: token\ndata: {json.dumps({'token': full_answer})}\n\n"
        else:
            session_context = {
                k: state.get(k) for k in
                ("policy_tier", "vehicle_category", "state_code", "coverage_type", "vehicle_year", "acv", "repair_cost")
            }
            messages = build_synthesis_messages(
                system_prompt=SYSTEM_PROMPT,
                user_message=state["user_message"],
                session_context=session_context,
                tool_results=tool_results_list,
                history=state.get("conversation_history"),
            )
            try:
                stream = await self.client.chat.completions.create(
                    model=settings.openai_synthesis_model,
                    messages=messages,
                    temperature=0.2,
                    max_tokens=500,  # Reduced from 900 for faster responses
                    stream=True,
                )
                async for chunk in stream:
                    delta = chunk.choices[0].delta.content if chunk.choices else None
                    if delta:
                        full_answer += delta
                        yield f"event: token\ndata: {json.dumps({'token': delta})}\n\n"
            except Exception as e:
                logger.error("stream_synthesis_failed", error=str(e), exc_info=True)
                full_answer = self._no_answer_fallback()
                yield f"event: token\ndata: {json.dumps({'token': full_answer})}\n\n"

        # Run output guardrail on the fully assembled answer.
        state["final_answer"] = full_answer
        state = await self._node_output_guardrail(state)
        if state.get("output_guardrail_triggered"):
            blocked_msg = state.get("final_answer") or ""
            yield f"event: guardrail\ndata: {json.dumps({'reason': state.get('guardrail_reason'), 'message': blocked_msg})}\n\n"

        # Memory update.
        await self._node_memory_update(state)

        latency_ms = int((time.perf_counter() - start) * 1000)
        state["latency_ms"] = latency_ms
        confidence = self._compute_confidence(state)

        done_payload = {
            "latency_ms": latency_ms,
            "confidence_score": confidence,
            "disclaimer": state.get("disclaimer"),
            "guardrail_triggered": bool(state.get("output_guardrail_triggered")),
            "guardrail_reason": state.get("guardrail_reason"),
            "calculation_breakdown": self._extract_breakdown_from_state(state),
        }
        yield f"event: done\ndata: {json.dumps(done_payload)}\n\n"

    @staticmethod
    def _extract_breakdown_from_state(state: AgentState) -> str | None:
        for t in (state.get("tool_results") or []):
            if t.get("tool_name") == "total_loss_tool" and t.get("success"):
                return (t.get("data") or {}).get("calculation_breakdown")
        return None


# Module-level singleton (built lazily).
_orchestrator: Optional[Orchestrator] = None


def get_orchestrator() -> Orchestrator:
    global _orchestrator
    if _orchestrator is None:
        _orchestrator = Orchestrator()
    return _orchestrator
