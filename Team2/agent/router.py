"""Intent router — gpt-4o-mini structured-JSON classifier.

WHY structured JSON (not function calling): JSON-mode is supported widely,
deterministic with temperature=0, and easy to parse + validate. Function
calling adds a layer that the simpler JSON approach doesn't need here.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import List, Optional

from openai import AsyncOpenAI

from agent.cache import LRUCache, normalize_text
from agent.prompts.router_prompt import ROUTER_SYSTEM_PROMPT
from api.config import settings
from observability.logger import get_logger

logger = get_logger(__name__)

# Module-level cache shared across requests. WHY module-level: same router
# decision for the same prompt is identical; one cache per process avoids
# duplicate gpt-4o-mini calls in burst traffic.
_router_cache: LRUCache[str, "RouterDecision"] = LRUCache(
    name="router_decisions",
    max_size=settings.router_cache_size,
    ttl_seconds=settings.router_cache_ttl_seconds,
)


VALID_INTENTS = {
    "COVERAGE_QA", "REPAIR_ESTIMATE", "TOTAL_LOSS", "FNOL_GUIDANCE",
    "RENTAL_LOOKUP", "ROADSIDE", "UM_UIM", "MULTI_INTENT",
    "CLARIFICATION_NEEDED", "GREETING", "OUT_OF_SCOPE",
}

VALID_TOOLS = {
    "policy_rag_tool", "repair_cost_tool", "total_loss_tool", "fnol_guide_tool",
    "rental_lookup_tool", "roadside_tool", "um_uim_tool", "coverage_identifier_tool",
}


@dataclass
class RouterDecision:
    intent: str
    tools: List[str]
    reasoning: str
    clarification_question: Optional[str] = None


class IntentRouter:
    def __init__(self, client: Optional[AsyncOpenAI] = None) -> None:
        self.client = client or AsyncOpenAI(api_key=settings.openai_api_key)
        self.model = settings.openai_router_model

    async def classify(
        self,
        user_message: str,
        conversation_history: Optional[List[dict]] = None,
    ) -> RouterDecision:
        """Classify intent. Falls back to COVERAGE_QA + policy_rag_tool on error.

        WHY this fallback: COVERAGE_QA + RAG is the safest default — it grounds
        in retrieved chunks and never fabricates. Better than refusing to answer.

        WHY conversation_history: bare clarification answers like "4000$" or
        "Texas" look off-topic in isolation. Providing the last assistant
        message (the clarification question) gives the router context to
        correctly classify the answer as a TOTAL_LOSS follow-up.
        """
        # Only cache context-free messages — context-dependent answers must
        # always be re-evaluated with their conversation history.
        has_history = bool(conversation_history)
        cache_key = normalize_text(user_message) if not has_history else None
        if cache_key:
            cached = _router_cache.get(cache_key)
            if cached is not None:
                logger.info("router_cache_hit", intent=cached.intent, tools=cached.tools)
                return cached

        # Build message list: system + up to last 2 history turns + current user msg.
        messages: List[dict] = [{"role": "system", "content": ROUTER_SYSTEM_PROMPT}]
        if has_history:
            # Include only the last assistant message for context (keeps prompt short).
            recent = [m for m in (conversation_history or []) if m.get("role") == "assistant"]
            if recent:
                messages.append({"role": "assistant", "content": recent[-1]["content"]})
        messages.append({"role": "user", "content": user_message})

        try:
            resp = await self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=0.0,
                response_format={"type": "json_object"},
                max_tokens=300,
            )
            raw = resp.choices[0].message.content or "{}"
            parsed = json.loads(raw)
            intent = str(parsed.get("intent", "")).strip().upper()
            tools = parsed.get("tools") or []
            reasoning = str(parsed.get("reasoning", "")).strip()
            clarification = str(parsed.get("clarification_question") or "").strip() or None

            if intent not in VALID_INTENTS:
                logger.warning("router_invalid_intent", raw=intent)
                intent = "COVERAGE_QA"
            tools = [t for t in tools if t in VALID_TOOLS]

            # Terminal intents must not invoke tools.
            if intent in {"OUT_OF_SCOPE", "CLARIFICATION_NEEDED", "GREETING"}:
                tools = []
            elif not tools:
                # Always provide at least one grounded tool for non-terminal queries.
                tools = ["policy_rag_tool"]

            # Safety net: if the model picked CLARIFICATION_NEEDED but forgot the
            # question, supply a generic one rather than returning an empty answer.
            if intent == "CLARIFICATION_NEEDED" and not clarification:
                clarification = (
                    "Could you give me a bit more detail — for example, the coverage "
                    "type, the incident, or what you'd like me to look up?"
                )

            decision = RouterDecision(
                intent=intent,
                tools=tools,
                reasoning=reasoning,
                clarification_question=clarification,
            )
            logger.info("router_decision", intent=intent, tools=tools, reasoning=reasoning)
            if cache_key:
                _router_cache.set(cache_key, decision)
            return decision
        except Exception as e:
            logger.error("router_failed_fallback", error=str(e), exc_info=True)
            # WHY: don't cache fallback decisions — the underlying error may be transient
            # (rate limit, network), and we want the next request to retry the real call.
            return RouterDecision(
                intent="COVERAGE_QA",
                tools=["policy_rag_tool"],
                reasoning="Router error; defaulting to grounded RAG.",
            )
