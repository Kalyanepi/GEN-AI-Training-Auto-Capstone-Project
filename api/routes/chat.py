"""POST /api/v1/chat — main chat endpoint.
POST /api/v1/chat/stream — SSE streaming endpoint.
"""
from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse

from agent.memory import SessionMemoryStore
from agent.orchestrator import Orchestrator
from agent.param_extractor import extract_params
from agent.state import AgentState
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
        logger.error("chat_endpoint_failed", error=str(e), exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error")


@router.post("/chat/stream")
async def chat_stream(
    request: ChatRequest,
    orchestrator: Orchestrator = Depends(orchestrator_dep),
    store: SessionMemoryStore = Depends(session_store_dep),
) -> StreamingResponse:
    """SSE streaming endpoint — yields meta/token/guardrail/done events.

    WHY a separate endpoint: the non-streaming /chat endpoint is kept intact
    for the dev console, evaluation suite, and other API consumers that need
    the full JSON response. The streaming endpoint adds word-by-word UX
    without breaking existing callers.
    """
    session_service = SessionService(store)
    # Get current session to know which fields are missing (for bare-number extraction).
    prior_ctx = store.get_context(request.session_id)
    missing_acv = prior_ctx is None or prior_ctx.acv is None
    missing_repair = prior_ctx is None or prior_ctx.repair_cost is None
    extracted = extract_params(request.message, missing_acv=missing_acv, missing_repair=missing_repair)
    ctx = session_service.hydrate_context(
        session_id=request.session_id,
        policy_tier=request.policy_tier,
        coverage_type=request.coverage_type,
        vehicle_category=request.vehicle_category,
        state_code=request.state_code or extracted.state_code,
        vehicle_year=request.vehicle_year,
        acv=request.acv if request.acv is not None else extracted.acv,
        repair_cost=request.repair_cost if request.repair_cost is not None else extracted.repair_cost,
    )
    initial: AgentState = {
        "session_id": request.session_id,
        "user_message": request.message,
        "policy_tier": ctx.policy_tier,
        "coverage_type": ctx.coverage_type,
        "vehicle_category": ctx.vehicle_category,
        "state_code": ctx.state_code,
        "vehicle_year": ctx.vehicle_year,
        "acv": ctx.acv,
        "repair_cost": ctx.repair_cost,
        "last_intent": ctx.last_intent,
    }

    async def event_generator():
        try:
            async for chunk in orchestrator.stream_invoke(initial):
                yield chunk
        except Exception as e:
            import json as _json
            logger.error("chat_stream_failed", error=str(e), exc_info=True)
            yield f"event: token\ndata: {_json.dumps({'token': 'Sorry, an error occurred. Please try again.'})}\\n\\n"
            yield "event: done\ndata: {}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "Connection": "keep-alive",
        },
    )
