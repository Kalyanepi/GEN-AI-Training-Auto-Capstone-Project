"""Thin service over SessionMemoryStore — used by the chat + session routes."""
from __future__ import annotations

from typing import Optional

from agent.memory import SessionContext, SessionMemoryStore


class SessionService:
    def __init__(self, store: SessionMemoryStore) -> None:
        self.store = store

    def hydrate_context(
        self,
        session_id: str,
        policy_tier: Optional[str] = None,
        coverage_type: Optional[str] = None,
        vehicle_category: Optional[str] = None,
        state_code: Optional[str] = None,
        vehicle_year: Optional[int] = None,
        acv: Optional[float] = None,
        repair_cost: Optional[float] = None,
    ) -> SessionContext:
        """Merge request-level facts into the session context."""
        return self.store.update_facts(
            session_id,
            policy_tier=policy_tier,
            coverage_type=coverage_type,
            vehicle_category=vehicle_category,
            state_code=state_code,
            vehicle_year=vehicle_year,
            acv=acv,
            repair_cost=repair_cost,
        )

    def clear(self, session_id: str) -> bool:
        return self.store.clear(session_id)

    def active_count(self) -> int:
        return self.store.session_count()
