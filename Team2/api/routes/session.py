"""Session management endpoint — explicit clear."""
from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException

from agent.memory import SessionMemoryStore
from api.dependencies import session_store_dep

router = APIRouter(prefix="/api/v1", tags=["session"])


@router.delete("/session/{session_id}")
async def clear_session(
    session_id: str,
    store: SessionMemoryStore = Depends(session_store_dep),
) -> dict:
    cleared = store.clear(session_id)
    if not cleared:
        raise HTTPException(status_code=404, detail="Session not found")
    return {"session_id": session_id, "cleared": True}
