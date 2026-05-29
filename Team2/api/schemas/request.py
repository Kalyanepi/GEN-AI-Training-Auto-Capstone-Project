"""Request schemas — fully validated by Pydantic before reaching the agent."""
from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field, field_validator


VALID_TIERS = {"standard", "premium", "elite"}
VALID_COVERAGE_TYPES = {
    "collision", "comprehensive", "liability", "um_uim", "gap",
    "medpay", "rental", "roadside", "fnol", "general",
}


class ChatRequest(BaseModel):
    """Inbound chat request."""
    session_id: str = Field(..., min_length=8, max_length=64)
    message: str = Field(..., min_length=1, max_length=2000)
    policy_tier: Optional[str] = None
    coverage_type: Optional[str] = None
    vehicle_category: Optional[str] = None
    state_code: Optional[str] = Field(default=None, max_length=2)
    vehicle_year: Optional[int] = Field(default=None, ge=1950, le=2100)
    acv: Optional[float] = Field(default=None, ge=0)
    repair_cost: Optional[float] = Field(default=None, ge=0)

    @field_validator("policy_tier")
    @classmethod
    def _validate_tier(cls, v: Optional[str]) -> Optional[str]:
        if v is None:
            return v
        v = v.lower().strip()
        if v not in VALID_TIERS:
            raise ValueError(f"policy_tier must be one of {sorted(VALID_TIERS)}")
        return v

    @field_validator("coverage_type")
    @classmethod
    def _validate_coverage(cls, v: Optional[str]) -> Optional[str]:
        if v is None:
            return v
        v = v.lower().strip()
        if v not in VALID_COVERAGE_TYPES:
            raise ValueError(f"coverage_type must be one of {sorted(VALID_COVERAGE_TYPES)}")
        return v

    @field_validator("state_code")
    @classmethod
    def _validate_state(cls, v: Optional[str]) -> Optional[str]:
        return v.upper().strip() if v else v
