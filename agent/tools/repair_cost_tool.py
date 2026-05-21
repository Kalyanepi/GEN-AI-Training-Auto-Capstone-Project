"""repair_cost_tool (FR-03) — fuzzy damage match + vehicle category lookup.

WHY fuzzy match damage_type: users say "bumper hit" or "my hood got dented",
not the exact CSV label "Hood dent repair". difflib at threshold 0.60 handles
common paraphrases without false positives on unrelated damage types.
"""
from __future__ import annotations

from difflib import SequenceMatcher
from threading import Lock
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from agent.tools.base_tool import BaseTool, DataNotFoundError, ToolResult
from api.config import settings
from ingestion.csv_loader import load_repair_cost_df
from rag.citation_tracker import csv_citation


# Tier -> deductible map per architecture plan §2.1.
_TIER_DEDUCTIBLES: Dict[str, Dict[str, int]] = {
    "standard": {"collision": 1000, "comprehensive": 500},
    "premium": {"collision": 500, "comprehensive": 250},
    "elite": {"collision": 250, "comprehensive": 0},
}


_df_cache: Optional[pd.DataFrame] = None
_df_lock = Lock()


def _get_df() -> pd.DataFrame:
    global _df_cache
    with _df_lock:
        if _df_cache is None:
            _df_cache = load_repair_cost_df()
        return _df_cache


def _best_damage_match(query: str, choices: List[str], threshold: float) -> Optional[Tuple[str, float]]:
    """Return (best_label, score) above threshold, else None."""
    q = query.lower().strip()
    best: Tuple[Optional[str], float] = (None, 0.0)
    for label in choices:
        score = SequenceMatcher(None, q, label.lower()).ratio()
        # WHY substring boost: "hood" vs "Hood dent repair" scores low under
        # SequenceMatcher; explicit substring presence is a stronger signal.
        if q in label.lower() or label.lower() in q:
            score = max(score, 0.75)
        if score > best[1]:
            best = (label, score)
    if best[0] is not None and best[1] >= threshold:
        return best  # type: ignore[return-value]
    return None


class RepairCostTool(BaseTool):
    name = "repair_cost_tool"
    description = (
        "Look up repair cost ranges by damage type and vehicle category. "
        "Applies the policyholder's deductible based on their tier."
    )

    async def _execute(
        self,
        damage_type: str,
        vehicle_category: Optional[str] = None,
        policy_tier: Optional[str] = None,
        coverage_type: Optional[str] = None,
        **_: Any,
    ) -> ToolResult:
        if not damage_type:
            raise DataNotFoundError("damage_type is required")

        df = _get_df()

        # 1. Fuzzy match the damage_type column.
        damage_choices = df["damage_type"].dropna().unique().tolist()
        match = _best_damage_match(damage_type, damage_choices, settings.fuzzy_damage_match_threshold)
        if not match:
            raise DataNotFoundError(f"No damage_type match for '{damage_type}'")
        matched_label, match_score = match

        subset = df[df["damage_type"] == matched_label]

        # 2. Filter by vehicle_category if provided (exact, case-insensitive).
        if vehicle_category:
            mask = subset["vehicle_category"].str.lower() == vehicle_category.lower()
            if mask.any():
                subset = subset[mask]
            # WHY: if no exact category match, we still return the damage-only
            # range — better partial answer than refusal.

        if subset.empty:
            raise DataNotFoundError(f"No rows for damage='{matched_label}' vehicle='{vehicle_category}'")

        # 3. Aggregate to a single answer.
        low = float(subset["repair_cost_low_usd"].min())
        high = float(subset["repair_cost_high_usd"].max())
        avg = float(subset["repair_cost_avg_usd"].mean())
        labor_hours = float(subset["typical_labor_hours"].mean())
        parts_availability = subset["parts_availability"].mode().iloc[0] if not subset.empty else None
        notes = "; ".join(sorted({n for n in subset["notes"].dropna().astype(str) if n.strip()}))

        # 4. Apply tier deductible (Collision vs Comprehensive).
        coverage_for_deductible = (coverage_type or subset["coverage_type"].iloc[0] or "").lower()
        deductible = 0
        if policy_tier and coverage_for_deductible in {"collision", "comprehensive"}:
            tier = policy_tier.lower()
            deductible = _TIER_DEDUCTIBLES.get(tier, {}).get(coverage_for_deductible, 0)

        net_low = max(0.0, low - deductible)
        net_high = max(0.0, high - deductible)
        net_avg = max(0.0, avg - deductible)

        # 5. Build citation referencing the CSV row(s).
        row_descriptor = f"{matched_label}"
        if vehicle_category:
            row_descriptor += f" / {vehicle_category}"
        excerpt = (
            f"{matched_label} ({coverage_for_deductible or 'N/A'}): "
            f"${low:,.0f} - ${high:,.0f} (avg ${avg:,.0f}); "
            f"labor ~{labor_hours:.1f}h; parts: {parts_availability}"
        )
        citation = csv_citation(
            csv_filename="RepairCost_ReferenceTable.csv",
            row_descriptor=row_descriptor,
            excerpt=excerpt,
            relevance_score=match_score,
        )

        return ToolResult(
            tool_name=self.name,
            success=True,
            data={
                "matched_damage_type": matched_label,
                "match_score": round(match_score, 3),
                "vehicle_category": vehicle_category,
                "coverage_type": coverage_for_deductible or None,
                "repair_cost_low_usd": round(low, 2),
                "repair_cost_high_usd": round(high, 2),
                "repair_cost_avg_usd": round(avg, 2),
                "typical_labor_hours": round(labor_hours, 1),
                "parts_availability": parts_availability,
                "notes": notes,
                "policy_tier": policy_tier,
                "deductible_applied": deductible,
                "net_cost_low_usd": round(net_low, 2),
                "net_cost_high_usd": round(net_high, 2),
                "net_cost_avg_usd": round(net_avg, 2),
            },
            citations=[citation],
            dollar_values=[low, high, avg, deductible, net_low, net_high, net_avg],
        )

    def _no_data_message(self) -> str:
        return (
            "I couldn't find that exact damage type or vehicle category in our "
            "repair cost reference table. Please describe the damage in different "
            "terms or contact your adjuster for a custom estimate."
        )
