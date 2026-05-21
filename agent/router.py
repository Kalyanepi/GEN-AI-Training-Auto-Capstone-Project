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

from agent.prompts.router_prompt import ROUTER_SYSTEM_PROMPT
from api.config import settings
from observability.logger import get_logger

logger = get_logger(__name__)


VALID_INTENTS = {
    "COVERAGE_QA", "REPAIR_ESTIMATE", "TOTAL_LOSS", "FNOL_GUIDANCE",
    "RENTAL_LOOKUP", "ROADSIDE", "UM_UIM", "MULTI_INTENT", "OUT_OF_SCOPE",
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


class IntentRouter:
    def __init__(self, client: Optional[AsyncOpenAI] = None) -> None:
        self.client = client or AsyncOpenAI(api_key=settings.openai_api_key)
        self.model = settings.openai_router_model

    async def classify(self, user_message: str) -> RouterDecision:
        """Classify intent. Falls back to COVERAGE_QA + policy_rag_tool on error.

        WHY this fallback: COVERAGE_QA + RAG is the safest default — it grounds
        in retrieved chunks and never fabricates. Better than refusing to answer.
        """
        try:
            resp = await self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": ROUTER_SYSTEM_PROMPT},
                    {"role": "user", "content": user_message},
                ],
                temperature=0.0,
                response_format={"type": "json_object"},
                max_tokens=300,
            )
            raw = resp.choices[0].message.content or "{}"
            parsed = json.loads(raw)
            intent = str(parsed.get("intent", "")).strip().upper()
            tools = parsed.get("tools") or []
            reasoning = str(parsed.get("reasoning", "")).strip()

            if intent not in VALID_INTENTS:
                logger.warning("router_invalid_intent", raw=intent)
                intent = "COVERAGE_QA"
            tools = [t for t in tools if t in VALID_TOOLS]
            if intent != "OUT_OF_SCOPE" and not tools:
                # Always provide at least one grounded tool for non-out-of-scope queries.
                tools = ["policy_rag_tool"]

            decision = RouterDecision(intent=intent, tools=tools, reasoning=reasoning)
            logger.info("router_decision", intent=intent, tools=tools, reasoning=reasoning)
            return decision
        except Exception as e:
            logger.error("router_failed_fallback", error=str(e), exc_info=True)
            return RouterDecision(
                intent="COVERAGE_QA",
                tools=["policy_rag_tool"],
                reasoning="Router error; defaulting to grounded RAG.",
            )
