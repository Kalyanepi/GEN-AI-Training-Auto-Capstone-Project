"""CSV schema validation and DataFrame loaders.

WHY explicit schema validation: silent column-name drift between datasets and
code is the most common cause of "tool returned wrong number" bugs. Fail loud
at ingestion time, not at user-query time.
"""
from __future__ import annotations

from pathlib import Path
from typing import List

import pandas as pd

from api.config import settings
from observability.logger import get_logger

logger = get_logger(__name__)


REPAIR_COST_REQUIRED_COLUMNS: List[str] = [
    "damage_type",
    "coverage_type",
    "vehicle_category",
    "repair_cost_low_usd",
    "repair_cost_high_usd",
    "repair_cost_avg_usd",
    "typical_labor_hours",
    "parts_availability",
    "notes",
]

TOTAL_LOSS_REQUIRED_COLUMNS: List[str] = [
    "state_code",
    "state_name",
    "total_loss_threshold_pct",
    "vehicle_age_min_yrs",
    "vehicle_age_max_yrs",
    "vehicle_age_category",
    "salvage_value_typical_pct_acv",
    "roadguard_settlement_basis",
    "notes",
]


class SchemaValidationError(ValueError):
    """Raised when a CSV is missing required columns."""


def _validate_columns(df: pd.DataFrame, required: List[str], source: str) -> None:
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise SchemaValidationError(
            f"{source}: missing required columns {missing}. Found: {list(df.columns)}"
        )


def load_repair_cost_df(path: Path | None = None) -> pd.DataFrame:
    """Load and validate the repair cost reference table."""
    path = path or settings.repair_cost_csv
    df = pd.read_csv(path)
    _validate_columns(df, REPAIR_COST_REQUIRED_COLUMNS, str(path))

    # WHY: coerce numerics so downstream arithmetic doesn't silently string-concat.
    for col in ["repair_cost_low_usd", "repair_cost_high_usd", "repair_cost_avg_usd", "typical_labor_hours"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # WHY: standardize whitespace/case so fuzzy matching is consistent.
    for col in ["damage_type", "coverage_type", "vehicle_category", "parts_availability"]:
        df[col] = df[col].astype(str).str.strip()

    logger.info("repair_cost_csv_loaded", rows=len(df), columns=list(df.columns))
    return df


def load_total_loss_df(path: Path | None = None) -> pd.DataFrame:
    """Load and validate the total loss threshold table."""
    path = path or settings.total_loss_csv
    df = pd.read_csv(path)
    _validate_columns(df, TOTAL_LOSS_REQUIRED_COLUMNS, str(path))

    # WHY strip '%' first: the source CSV stores threshold as "75%" (string).
    # pd.to_numeric("75%") returns NaN silently, which propagates as nan%
    # in the calculation breakdown and breaks the is_total_loss check.
    for col in [
        "total_loss_threshold_pct",
        "vehicle_age_min_yrs",
        "vehicle_age_max_yrs",
        "salvage_value_typical_pct_acv",
    ]:
        df[col] = (
            df[col].astype(str).str.replace("%", "", regex=False).str.strip()
        )
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df["state_code"] = df["state_code"].astype(str).str.strip().str.upper()
    df["state_name"] = df["state_name"].astype(str).str.strip()

    logger.info("total_loss_csv_loaded", rows=len(df), columns=list(df.columns))
    return df
