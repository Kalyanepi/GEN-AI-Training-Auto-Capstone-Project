"""total_loss_tool (FR-04) — state threshold + ACV calculation engine.

WHY a dedicated tool with full audit breakdown: total loss decisions are the
highest-stakes calculation in the system. The output exposes step-by-step math
so adjusters and demo audiences can verify every number.

State special rules (per architecture plan §2.3):
  - Most states: standard percentage threshold
  - TX, CO (100%): ACTUAL total loss — repair + storage must equal/exceed ACV
  - PA (110%): COMBINED rule — repair cost AND salvage value both considered
"""
from __future__ import annotations

from datetime import datetime
from threading import Lock
from typing import Any, Dict, Optional

import pandas as pd

from agent.tools.base_tool import BaseTool, DataNotFoundError, ToolResult
from agent.tools.repair_cost_tool import _TIER_DEDUCTIBLES
from ingestion.csv_loader import load_total_loss_df
from rag.citation_tracker import csv_citation


_df_cache: Optional[pd.DataFrame] = None
_df_lock = Lock()

_SPECIAL_RULES = {
    "TX": "Texas applies the ACTUAL total loss rule — repair cost plus storage must equal or exceed ACV before total loss is declared.",
    "CO": "Colorado applies the ACTUAL total loss rule — repair cost plus storage must equal or exceed ACV before total loss is declared.",
    "PA": "Pennsylvania applies a COMBINED rule — both repair cost AND salvage value are considered in the total loss decision.",
}


def _get_df() -> pd.DataFrame:
    global _df_cache
    with _df_lock:
        if _df_cache is None:
            _df_cache = load_total_loss_df()
        return _df_cache


def _vehicle_age_from_year(vehicle_year: Optional[int]) -> Optional[int]:
    if not vehicle_year or vehicle_year <= 1900:
        return None
    return max(0, datetime.utcnow().year - int(vehicle_year))


def _select_age_row(df: pd.DataFrame, vehicle_age: Optional[int]) -> pd.Series:
    """Pick the correct age-bucket row for a state.

    WHY default to mid-age bucket when vehicle_age unknown: most policyholder
    queries omit vehicle year. Mid-age (4-6 yrs) gives a defensible default
    rather than refusing to answer.
    """
    if vehicle_age is not None:
        mask = (df["vehicle_age_min_yrs"] <= vehicle_age) & (df["vehicle_age_max_yrs"] >= vehicle_age)
        if mask.any():
            return df[mask].iloc[0]
    # Fallback: pick the row whose age bucket label contains 'mid', else first row.
    mid = df[df["vehicle_age_category"].str.contains("mid", case=False, na=False)]
    if not mid.empty:
        return mid.iloc[0]
    return df.iloc[0]


class TotalLossTool(BaseTool):
    name = "total_loss_tool"
    description = (
        "Determine whether a vehicle is a total loss given ACV, repair cost, "
        "state, and optional vehicle year. Returns full calculation breakdown."
    )

    async def _execute(
        self,
        state_code: str,
        acv: Optional[float] = None,
        repair_cost: Optional[float] = None,
        vehicle_year: Optional[int] = None,
        policy_tier: Optional[str] = None,
        coverage_type: Optional[str] = "collision",
        deductible_override: Optional[float] = None,
        **_: Any,
    ) -> ToolResult:
        if not state_code:
            raise DataNotFoundError("state_code is required")

        # Threshold-lookup mode: no ACV/repair provided — just return state thresholds.
        lookup_only = (acv is None or acv <= 0) and (repair_cost is None or repair_cost <= 0)

        df = _get_df()
        state_code = state_code.strip().upper()
        state_rows = df[df["state_code"] == state_code]
        if state_rows.empty:
            raise DataNotFoundError(f"No threshold data for state '{state_code}'")

        vehicle_age = _vehicle_age_from_year(vehicle_year)
        row = _select_age_row(state_rows, vehicle_age)

        threshold_pct = float(row["total_loss_threshold_pct"])
        salvage_pct = float(row["salvage_value_typical_pct_acv"])
        state_name = str(row["state_name"])
        age_category = str(row["vehicle_age_category"])
        settlement_basis = str(row["roadguard_settlement_basis"])
        notes = str(row["notes"]) if pd.notna(row["notes"]) else ""
        special_rule = _SPECIAL_RULES.get(state_code)

        if lookup_only:
            # Threshold-lookup mode: return all age-bucket rows for the state.
            all_rows = []
            for _, r in state_rows.iterrows():
                all_rows.append({
                    "age_category": str(r["vehicle_age_category"]),
                    "threshold_pct": float(r["total_loss_threshold_pct"]),
                    "salvage_pct": float(r["salvage_value_typical_pct_acv"]),
                    "notes": str(r["notes"]) if pd.notna(r["notes"]) else "",
                })
            excerpt = (
                f"{state_name}: total loss threshold {threshold_pct:.0f}% of ACV. "
                f"Salvage ~{salvage_pct:.0f}% ACV. Settlement basis: {settlement_basis}."
            )
            citation = csv_citation(
                csv_filename="TotalLoss_Threshold_Table.csv",
                row_descriptor=f"{state_name} thresholds",
                excerpt=excerpt,
                relevance_score=1.0,
            )
            return ToolResult(
                tool_name=self.name,
                success=True,
                data={
                    "lookup_only": True,
                    "state_code": state_code,
                    "state_name": state_name,
                    "threshold_pct": threshold_pct,
                    "salvage_pct": salvage_pct,
                    "settlement_basis": settlement_basis,
                    "special_rule": special_rule,
                    "notes": notes,
                    "all_age_buckets": all_rows,
                },
                citations=[citation],
                dollar_values=[],
            )

        repair_ratio_pct = (repair_cost / acv) * 100.0
        threshold_amount = acv * (threshold_pct / 100.0)
        marginal_amount = threshold_amount - repair_cost  # positive => below threshold
        is_total_loss = repair_ratio_pct >= threshold_pct

        salvage_value = acv * (salvage_pct / 100.0)

        # Apply deductible to settlement when total loss declared.
        # WHY override: if the user explicitly states their deductible in the
        # query, that takes precedence over the tier default — their actual
        # policy may differ from the tier standard.
        if deductible_override is not None and deductible_override >= 0:
            deductible = float(deductible_override)
        elif policy_tier and coverage_type:
            deductible = float(_TIER_DEDUCTIBLES.get(policy_tier.lower(), {}).get(coverage_type.lower(), 0))
        else:
            deductible = 0.0
        settlement_amount = max(0.0, acv - deductible) if is_total_loss else 0.0

        breakdown_lines = [
            f"State: {state_name} ({state_code})  |  Threshold: {threshold_pct:.0f}%",
            f"Vehicle age bucket: {age_category}",
            f"Repair ratio = ${repair_cost:,.0f} / ${acv:,.0f} = {repair_ratio_pct:.1f}%",
            f"Threshold trigger amount = ${acv:,.0f} x {threshold_pct:.0f}% = ${threshold_amount:,.0f}",
            f"Margin to threshold = ${marginal_amount:,.0f} ({'below' if marginal_amount > 0 else 'at/above'} threshold)",
            f"Salvage value (~{salvage_pct:.0f}% of ACV) = ${salvage_value:,.0f}",
        ]
        deductible_source = "user-provided" if deductible_override is not None and deductible_override >= 0 else "tier default"
        if is_total_loss:
            breakdown_lines.append(
                f"Settlement = ACV - deductible ({deductible_source}) = ${acv:,.0f} - ${deductible:,.0f} = ${settlement_amount:,.0f}"
            )
        if special_rule:
            breakdown_lines.append(f"Special rule: {special_rule}")

        breakdown = "\n".join(breakdown_lines)

        excerpt = (
            f"{state_name}: {threshold_pct:.0f}% threshold, "
            f"salvage ~{salvage_pct:.0f}% ACV, age {age_category}. "
            f"Settlement basis: {settlement_basis}."
        )
        citation = csv_citation(
            csv_filename="TotalLoss_Threshold_Table.csv",
            row_descriptor=f"{state_name} / {age_category}",
            excerpt=excerpt,
            relevance_score=1.0,
        )

        return ToolResult(
            tool_name=self.name,
            success=True,
            data={
                "is_total_loss": is_total_loss,
                "state_code": state_code,
                "state_name": state_name,
                "threshold_pct": threshold_pct,
                "repair_ratio_pct": round(repair_ratio_pct, 2),
                "threshold_trigger_amount_usd": round(threshold_amount, 2),
                "marginal_amount_usd": round(marginal_amount, 2),
                "acv_usd": round(float(acv), 2),
                "repair_cost_usd": round(float(repair_cost), 2),
                "salvage_pct": salvage_pct,
                "salvage_value_usd": round(salvage_value, 2),
                "deductible_applied": deductible,
                "settlement_amount_usd": round(settlement_amount, 2),
                "vehicle_age_category": age_category,
                "vehicle_age_years": vehicle_age,
                "settlement_basis": settlement_basis,
                "special_rule": special_rule,
                "notes": notes,
                "calculation_breakdown": breakdown,
            },
            citations=[citation],
            dollar_values=[
                float(acv),
                float(repair_cost),
                threshold_amount,
                abs(marginal_amount),
                salvage_value,
                float(deductible),
                settlement_amount,
            ],
        )

    def _no_data_message(self) -> str:
        return (
            "I need ACV, repair cost, and state code to determine total loss. "
            "Please provide those, or contact your adjuster for an official "
            "determination."
        )
