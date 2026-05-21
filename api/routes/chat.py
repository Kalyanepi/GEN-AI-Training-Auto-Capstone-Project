"""POST /api/v1/chat — main chat endpoint."""
from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException

from agent.memory import SessionMemoryStore
from agent.orchestrator import Orchestrator
from api.dependencies import orchestrator_dep, session_store_dep
from api.schemas.request import ChatRequest
from api.schemas.response import ChatResponse
from api.services.chat_service import ChatService
from api.services.session_service import SessionService
from observability.logger import get_logger

logger = get_logger(__name__)

router = APIRouter(prefix="/api/v1", tags=["chat"])


@router.post("/chat", response_model=ChatResponse)
async def chat(
    request: ChatRequest,
    orchestrator: Orchestrator = Depends(orchestrator_dep),
    store: SessionMemoryStore = Depends(session_store_dep),
) -> ChatResponse:
    try:
        service = ChatService(orchestrator=orchestrator, session_service=SessionService(store))
        return await service.chat(request)
    except Exception as e:
        # WHY: never leak stack traces to clients; structured log keeps it.
        logger.error("chat_endpoint_failed", error=str(e), exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error")
