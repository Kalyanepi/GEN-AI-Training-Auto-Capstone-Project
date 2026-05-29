"""FastAPI dependency injection — single place to wire singletons.

WHY: routes/services should not import singletons directly; passing them as
Depends() args keeps the code testable (override in pytest with .dependency_overrides).
"""
from __future__ import annotations

from agent.memory import SessionMemoryStore, get_session_store
from agent.orchestrator import Orchestrator, get_orchestrator


def orchestrator_dep() -> Orchestrator:
    return get_orchestrator()


def session_store_dep() -> SessionMemoryStore:
    return get_session_store()
