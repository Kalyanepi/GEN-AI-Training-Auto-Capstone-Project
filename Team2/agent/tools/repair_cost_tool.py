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


# WHY a curated stop-set: tokens like "damage", "repair", "the", "my" carry
# no part-identifying signal. Removing them before token-overlap scoring
# prevents "hood damage" + "windshield damage" from sharing the spurious
# "damage" token and inflating the wrong row's score.
_DAMAGE_STOPWORDS = {
    "the", "a", "an", "my", "some", "any", "of", "for", "on", "in", "to",
    "and", "or", "with", "got", "have", "had", "is", "was", "were",
    "damage", "damaged", "repair", "repaired", "fix", "fixed", "fixing",
    "broken", "cracked", "dented", "scratched", "scratch", "dent", "crack",
    "issue", "issues", "problem", "problems", "needs", "need",
    # Deployment / filler words that add no part-identifying signal.
    "deployed", "all", "went", "off", "triggered", "activated", "inflate", "inflated",
}

# WHY synonym map: users say "airbags" (plural) but the CSV labels use
# "Airbag" (singular). Normalizing before key-token scoring prevents the
# plural form from failing the Jaccard overlap check.
_DAMAGE_SYNONYMS: dict[str, str] = {
    "airbags": "airbag",
    "bags": "bag",
    "bumpers": "bumper",
    "fenders": "fender",
    "doors": "door",
    "windows": "window",
    "tires": "tire",
    "wheels": "wheel",
    "mirrors": "mirror",
    "headlights": "headlight",
    "taillights": "taillight",
    # quantity / count synonyms
    "both": "dual",
    "two": "dual",
    "double": "dual",
    "one": "1",
    "single": "1",
}


def _key_tokens(text: str) -> set:
    """Lowercase content tokens minus stopwords, with synonym normalization."""
    import re as _re
    raw = _re.findall(r"[a-z0-9]+", text.lower())
    result = set()
    for t in raw:
        if t in _DAMAGE_STOPWORDS or len(t) <= 1:
            continue
        result.add(_DAMAGE_SYNONYMS.get(t, t))
    return result


def _best_damage_match(query: str, choices: List[str], threshold: float) -> Optional[Tuple[str, float]]:
    """Return (best_label, score) above threshold, else None.

    Scoring blends three signals:
      1. Plain SequenceMatcher ratio (catches typos, word reordering).
      2. Substring containment (full phrase in label or vice-versa).
      3. Key-noun token overlap (Jaccard on content words).

    WHY blended: pure SequenceMatcher rates "hood damage" vs "Hood dent
    repair" at ~0.40 — below threshold — even though they obviously refer
    to the same part. The token-overlap signal recognizes the shared
    "hood" noun and rescues the match. Conversely, "windshield damage"
    won't accidentally hit "Hood replacement" because their key-token
    sets are disjoint after stopword removal.
    """
    q = query.lower().strip()
    q_tokens = _key_tokens(q)
    best: Tuple[Optional[str], float] = (None, 0.0)
    for label in choices:
        ll = label.lower()
        score = SequenceMatcher(None, q, ll).ratio()
        # Exact case-insensitive match — always wins.
        if q == ll:
            return (label, 1.0)
        # Substring containment boost.
        if q in ll or ll in q:
            score = max(score, 0.80)
        # Key-noun token overlap.
        l_tokens = _key_tokens(label)
        if q_tokens and l_tokens:
            shared = q_tokens & l_tokens
            if shared:
                jaccard = len(shared) / len(q_tokens | l_tokens)
                # All label tokens matched = very strong signal.
                if l_tokens.issubset(q_tokens):
                    score = max(score, 0.85)
                else:
                    score = max(score, max(0.70, jaccard))
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
        damage_type: Optional[str] = None,
        damage_types: Optional[List[str]] = None,
        vehicle_category: Optional[str] = None,
        policy_tier: Optional[str] = None,
        coverage_type: Optional[str] = None,
        **_: Any,
    ) -> ToolResult:
        # Resolve a unified list of damages. damage_types (plural) wins; if
        # only damage_type is provided we still iterate for consistency.
        damages: List[str] = []
        if damage_types:
            damages = [d for d in damage_types if d and d.strip()]
        elif damage_type:
            damages = [damage_type]
        if not damages:
            raise DataNotFoundError("damage_type is required")

        # If multiple damages, fan out per-damage lookups and aggregate.
        if len(damages) > 1:
            return await self._execute_multi(
                damages, vehicle_category, policy_tier, coverage_type,
            )

        damage_type = damages[0]
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

    async def _execute_multi(
        self,
        damages: List[str],
        vehicle_category: Optional[str],
        policy_tier: Optional[str],
        coverage_type: Optional[str],
    ) -> ToolResult:
        """Aggregate per-damage CSV lookups into a single ToolResult.

        WHY: real users say "cracked headlight and hood damage on my Civic".
        Looking up only the first damage hides costs the user clearly asked
        about. We run the single-damage path for each, then sum low/high/avg
        ranges into a combined estimate. Citations and dollar values are
        unioned so the guardrail's allowed-dollar set covers all components.
        """
        per_damage: List[Dict[str, Any]] = []
        all_citations = []
        all_dollars: List[float] = []
        unmatched: List[str] = []

        # Reuse single-damage path for each, ignoring failures (we still
        # want partial answers for the damages that DID match).
        for d in damages:
            try:
                r = await self._execute(
                    damage_type=d,
                    vehicle_category=vehicle_category,
                    policy_tier=policy_tier,
                    coverage_type=coverage_type,
                )
            except DataNotFoundError:
                unmatched.append(d)
                continue
            if not r.success or not r.data:
                unmatched.append(d)
                continue
            per_damage.append(r.data)
            all_citations.extend(r.citations)
            all_dollars.extend(r.dollar_values)

        if not per_damage:
            raise DataNotFoundError(
                f"No damage_type matches for any of: {damages}"
            )

        # Aggregate: sum the gross ranges; deductible is applied ONCE per
        # claim, not per damage line (the user has one deductible for the
        # whole claim event).
        gross_low = sum(d["repair_cost_low_usd"] for d in per_damage)
        gross_high = sum(d["repair_cost_high_usd"] for d in per_damage)
        gross_avg = sum(d["repair_cost_avg_usd"] for d in per_damage)
        labor_total = sum(d["typical_labor_hours"] for d in per_damage)

        # Use the deductible from the first row (all rows share tier+coverage).
        deductible = per_damage[0].get("deductible_applied", 0)
        net_low = max(0.0, gross_low - deductible)
        net_high = max(0.0, gross_high - deductible)
        net_avg = max(0.0, gross_avg - deductible)
        all_dollars.extend([gross_low, gross_high, gross_avg, net_low, net_high, net_avg])

        return ToolResult(
            tool_name=self.name,
            success=True,
            data={
                "multi_damage": True,
                "damages_matched": [d["matched_damage_type"] for d in per_damage],
                "damages_unmatched": unmatched,
                "vehicle_category": vehicle_category,
                "per_damage": per_damage,
                "repair_cost_low_usd": round(gross_low, 2),
                "repair_cost_high_usd": round(gross_high, 2),
                "repair_cost_avg_usd": round(gross_avg, 2),
                "typical_labor_hours": round(labor_total, 1),
                "policy_tier": policy_tier,
                "deductible_applied": deductible,
                "net_cost_low_usd": round(net_low, 2),
                "net_cost_high_usd": round(net_high, 2),
                "net_cost_avg_usd": round(net_avg, 2),
            },
            citations=all_citations,
            dollar_values=all_dollars,
        )

    def _no_data_message(self) -> str:
        return (
            "I couldn't find that exact damage type or vehicle category in our "
            "repair cost reference table. Please describe the damage in different "
            "terms or contact your adjuster for a custom estimate."
        )
