"""Two-tier session memory: conversation buffer + structured SessionContext.

Tier 1: Last K conversation turns (user + assistant messages).
Tier 2: Extracted structured facts (tier, vehicle, state, ACV, etc.) that
        persist across turns so the user never has to repeat themselves.

WHY in-memory dict (not Redis) for Phase 2: the architecture explicitly states
sessions are ephemeral — no PII persisted to disk. Single-instance deployment
in Phase 2; Redis is the documented Phase 3 path.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from threading import RLock
from typing import Dict, List, Optional

from pydantic import BaseModel, Field

from api.config import settings
from observability.logger import get_logger

logger = get_logger(__name__)


class SessionContext(BaseModel):
    """Structured facts extracted from the conversation (Tier 2)."""
    session_id: str
    policy_tier: Optional[str] = None
    vehicle_category: Optional[str] = None
    vehicle_year: Optional[int] = None
    state_code: Optional[str] = None
    coverage_type: Optional[str] = None
    coverage_types_discussed: List[str] = Field(default_factory=list)
    incident_type: Optional[str] = None
    acv: Optional[float] = None
    repair_cost: Optional[float] = None
    last_intent: Optional[str] = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    last_active_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


@dataclass
class _SessionEntry:
    context: SessionContext
    history: List[Dict[str, str]] = field(default_factory=list)  # [{role, content}]


class SessionMemoryStore:
    """In-memory session store with TTL-based eviction."""

    def __init__(self) -> None:
        self._sessions: Dict[str, _SessionEntry] = {}
        self._lock = RLock()
        self.window_k = settings.conversation_window_k
        self.ttl = timedelta(minutes=settings.session_ttl_minutes)

    def _evict_expired(self) -> None:
        """Drop sessions inactive longer than TTL.

        WHY: prevents unbounded memory growth on a long-running API.
        """
        now = datetime.now(timezone.utc)
        expired = [sid for sid, e in self._sessions.items() if now - e.context.last_active_at > self.ttl]
        for sid in expired:
            self._sessions.pop(sid, None)
        if expired:
            logger.info("sessions_evicted", count=len(expired))

    def get_or_create(self, session_id: str, **initial_facts) -> _SessionEntry:
        with self._lock:
            self._evict_expired()
            entry = self._sessions.get(session_id)
            if entry is None:
                entry = _SessionEntry(context=SessionContext(session_id=session_id, **initial_facts))
                self._sessions[session_id] = entry
                logger.info("session_created", session_id=session_id)
            return entry

    def update_facts(self, session_id: str, **facts) -> SessionContext:
        """Merge non-null facts into the session context."""
        with self._lock:
            entry = self.get_or_create(session_id)
            ctx = entry.context
            for k, v in facts.items():
                if v in (None, ""):
                    continue
                if k == "coverage_type" and v:
                    if v not in ctx.coverage_types_discussed:
                        ctx.coverage_types_discussed.append(v)
                if hasattr(ctx, k):
                    setattr(ctx, k, v)
            ctx.last_active_at = datetime.now(timezone.utc)
            return ctx

    def append_turn(self, session_id: str, role: str, content: str) -> None:
        with self._lock:
            entry = self.get_or_create(session_id)
            entry.history.append({"role": role, "content": content})
            # WHY: keep window k pairs (= 2k messages) to bound prompt size.
            max_messages = self.window_k * 2
            if len(entry.history) > max_messages:
                entry.history = entry.history[-max_messages:]
            entry.context.last_active_at = datetime.now(timezone.utc)

    def get_history(self, session_id: str) -> List[Dict[str, str]]:
        with self._lock:
            entry = self._sessions.get(session_id)
            return list(entry.history) if entry else []

    def get_context(self, session_id: str) -> Optional[SessionContext]:
        with self._lock:
            entry = self._sessions.get(session_id)
            return entry.context if entry else None

    def clear(self, session_id: str) -> bool:
        with self._lock:
            return self._sessions.pop(session_id, None) is not None

    def session_count(self) -> int:
        with self._lock:
            self._evict_expired()
            return len(self._sessions)


# Module-level singleton — used by api.dependencies.
_store: Optional[SessionMemoryStore] = None


def get_session_store() -> SessionMemoryStore:
    global _store
    if _store is None:
        _store = SessionMemoryStore()
    return _store
