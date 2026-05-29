"""Health and readiness probes."""
from __future__ import annotations

from fastapi import APIRouter, Depends

from agent.memory import SessionMemoryStore
from api.dependencies import session_store_dep
from api.schemas.response import HealthResponse, ReadyResponse
from ingestion.csv_loader import load_repair_cost_df, load_total_loss_df
from observability.logger import get_logger
from rag.faiss_store import get_store

logger = get_logger(__name__)

router = APIRouter(tags=["health"])


@router.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    """Liveness — verifies FAISS index + CSVs load.

    WHY this depth: shallow "OK" probes pass even when the index is missing,
    causing first-request 503s. Probing critical resources at health-check
    time surfaces config issues during deploy.
    """
    chunk_count = 0
    faiss_loaded = False
    repair_rows = 0
    total_loss_rows = 0
    try:
        store = get_store()
        chunk_count = store.size
        faiss_loaded = True
    except Exception as e:
        logger.error("health_faiss_load_failed", error=str(e))

    try:
        repair_rows = len(load_repair_cost_df())
    except Exception as e:
        logger.error("health_repair_csv_failed", error=str(e))

    try:
        total_loss_rows = len(load_total_loss_df())
    except Exception as e:
        logger.error("health_total_loss_csv_failed", error=str(e))

    status = "ok" if (faiss_loaded and repair_rows and total_loss_rows) else "degraded"
    return HealthResponse(
        status=status,
        faiss_loaded=faiss_loaded,
        chunk_count=chunk_count,
        repair_cost_rows=repair_rows,
        total_loss_rows=total_loss_rows,
    )


@router.get("/ready", response_model=ReadyResponse)
async def ready(store: SessionMemoryStore = Depends(session_store_dep)) -> ReadyResponse:
    chunk_count = 0
    try:
        chunk_count = get_store().size
    except Exception:
        pass
    return ReadyResponse(
        status="ready" if chunk_count > 0 else "not_ready",
        sessions_active=store.session_count(),
        chunk_count=chunk_count,
    )
