"""coverage_identifier_tool (FR-01, FR-07) — classify incident -> coverage type.

WHY keyword-based with conservative confidence: deterministic and auditable.
LLM classification here would risk hallucinating "yes you're covered" when
the tool's job is only to point the router at the right coverage_type.

WHY 'never definitive on ambiguous': the architecture plan explicitly forbids
this tool from making coverage decisions. It's a routing aid, not an oracle.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from agent.tools.base_tool import BaseTool, ToolResult


_COVERAGE_KEYWORDS: Dict[str, List[str]] = {
    "collision": ["hit", "rear-ended", "rear ended", "crash", "collided", "impact", "rollover", "pothole", "at fault", "at-fault", "single car"],
    "comprehensive": ["theft", "stolen", "vandalism", "vandalized", "hail", "flood", "fire", "deer", "animal", "tree fell", "weather", "windshield", "glass"],
    "liability": ["other driver", "third party", "third-party", "their car", "they hit me", "property damage to other"],
    "um_uim": ["uninsured", "underinsured", "no insurance", "hit and run", "hit-and-run"],
    "gap": ["loan", "lease", "owe more", "upside down", "negative equity"],
    "medpay": ["medical", "hospital", "ambulance", "injuries", "passengers hurt"],
    "roadside": ["tow", "towing", "jump start", "lockout", "flat tire", "stranded", "out of fuel", "ran out of gas"],
    "rental": ["rental car", "loaner", "substitute vehicle", "while my car is in the shop"],
}

# Tier deductibles (mirror repair_cost_tool table) — used when the
# identified coverage maps to a known deductible.
_TIER_DEDUCTIBLES: Dict[str, Dict[str, int]] = {
    "standard": {"collision": 1000, "comprehensive": 500},
    "premium": {"collision": 500, "comprehensive": 250},
    "elite": {"collision": 250, "comprehensive": 0},
}


def _score_coverages(text: str) -> List[Tuple[str, int]]:
    lower = text.lower()
    scores: List[Tuple[str, int]] = []
    for cov, kws in _COVERAGE_KEYWORDS.items():
        hits = sum(1 for k in kws if re.search(rf"\b{re.escape(k)}\b", lower))
        if hits:
            scores.append((cov, hits))
    return sorted(scores, key=lambda x: x[1], reverse=True)


class CoverageIdentifierTool(BaseTool):
    name = "coverage_identifier_tool"
    description = (
        "Identify which coverage type(s) likely apply to an incident "
        "description. Never definitive on ambiguous cases — confirms via adjuster."
    )

    async def _execute(
        self,
        incident_description: str,
        policy_tier: Optional[str] = None,
        **_: Any,
    ) -> ToolResult:
        if not incident_description:
            return ToolResult(
                tool_name=self.name,
                success=False,
                data={"reasoning": "No incident description provided"},
                fallback_message=self._no_data_message(),
            )

        ranked = _score_coverages(incident_description)
        primary = ranked[0][0] if ranked else None
        secondary = ranked[1][0] if len(ranked) >= 2 else None

        # Confidence heuristic.
        if not ranked:
            confidence = "low"
        elif len(ranked) == 1 or ranked[0][1] >= ranked[1][1] + 2:
            confidence = "high"
        elif ranked[0][1] > ranked[1][1]:
            confidence = "medium"
        else:
            confidence = "low"

        deductible = None
        if primary in {"collision", "comprehensive"} and policy_tier:
            deductible = _TIER_DEDUCTIBLES.get(policy_tier.lower(), {}).get(primary)

        # WHY explicit ambiguity caveat in reasoning: the orchestrator's LLM
        # synthesis prompt reads this and is instructed to mirror the caveat.
        reasoning_parts = [f"Keyword analysis ranked coverages: {ranked}."]
        if confidence != "high" and secondary:
            reasoning_parts.append(
                f"Both {primary} and {secondary} may apply — your adjuster will confirm."
            )

        return ToolResult(
            tool_name=self.name,
            success=primary is not None,
            data={
                "primary_coverage": primary,
                "secondary_coverage": secondary,
                "confidence": confidence,
                "scores": ranked,
                "applicable_deductible_usd": deductible,
                "reasoning": " ".join(reasoning_parts),
                "policy_tier": policy_tier,
            },
            dollar_values=[float(deductible)] if deductible is not None else [],
        )

    def _no_data_message(self) -> str:
        return (
            "Please describe what happened (e.g., 'rear-ended', 'hail damage', "
            "'theft') so I can identify the relevant coverage."
        )
