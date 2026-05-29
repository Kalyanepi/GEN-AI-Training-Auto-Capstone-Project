"""Guardrail-specific prompt fragments referenced by the orchestrator."""
from __future__ import annotations

GUARDRAIL_BLOCKED_PREFIX = (
    "[Guardrail engaged] "
)

OUT_OF_SCOPE_RESPONSE_TEMPLATE = (
    "I can only help with auto insurance topics: policy coverage, repair "
    "estimates, total loss thresholds, FNOL guidance, rental limits, and "
    "roadside coverage. Please rephrase your question within these areas."
)
