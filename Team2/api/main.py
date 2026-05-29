"""FastAPI app factory + lifespan events..

WHY lifespan: load FAISS index and CSVs once at startup. Without lifespan
warmup, the first chat request would pay the ~150ms load cost.
"""
from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI

from api.config import settings
from api.middleware.logging_middleware import StructuredLoggingMiddleware
from api.middleware.rate_limit_middleware import TokenBucketRateLimiter
from api.routes import chat as chat_route
from api.routes import health as health_route
from api.routes import session as session_route
from ingestion.csv_loader import load_repair_cost_df, load_total_loss_df
from observability.langsmith_tracer import configure_langsmith
from observability.logger import get_logger
from rag.faiss_store import load_store

logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("api_startup_begin")
    configure_langsmith()
    # Warm up: load FAISS + CSVs.
    try:
        load_store()
    except FileNotFoundError as e:
        logger.warning("faiss_index_missing_at_startup", detail=str(e))
    try:
        load_repair_cost_df()
        load_total_loss_df()
    except Exception as e:
        logger.warning("csv_warmup_failed", error=str(e))
    # Build orchestrator (compiles graph + instantiates tools).
    from agent.orchestrator import get_orchestrator
    get_orchestrator()

    # Pre-warm hot paths so the first real chat doesn't pay first-load costs.
    # WHY each item: query embedding pays ~250ms TLS+API, reranker pays ~3s
    # torch import + model download/load. Doing them now means the first user
    # request feels as fast as steady-state.
    _prewarm_hot_paths()

    logger.info("api_startup_complete")
    yield
    logger.info("api_shutdown")


def _prewarm_hot_paths() -> None:
    """Fire one no-op embed + one no-op rerank so first-request latency drops.

    Failures here are non-fatal — startup must succeed even if OpenAI is down
    or the cross-encoder model can't be downloaded.
    """
    try:
        from ingestion.embedder import Embedder
        Embedder().embed_query("warmup")
        logger.info("embedder_prewarmed")
    except Exception as e:
        logger.warning("embedder_prewarm_failed", error=str(e))
    if settings.reranker_enabled:
        try:
            from rag.reranker import _get_model
            _get_model()
            logger.info("reranker_prewarmed")
        except Exception as e:
            logger.warning("reranker_prewarm_failed", error=str(e))
    # WHY: Presidio + spaCy model takes ~2s to initialize on first request.
    # Pre-warming here means first chat request doesn't pay this cost.
    try:
        from guardrails.input_guardrails import _get_presidio_engine
        _get_presidio_engine()
        logger.info("presidio_prewarmed")
    except Exception as e:
        logger.warning("presidio_prewarm_failed", error=str(e))


def create_app() -> FastAPI:
    app = FastAPI(
        title="RoadGuard AI Copilot API",
        description="Auto Insurance AI Copilot — Accurate. Grounded. Cited. Never Fabricated.",
        version="1.0.0",
        lifespan=lifespan,
    )
    # Middleware order matters: rate limit BEFORE logging so blocked requests
    # are logged with their 429 status.
    app.add_middleware(StructuredLoggingMiddleware)
    app.add_middleware(TokenBucketRateLimiter)

    app.include_router(health_route.router)
    app.include_router(chat_route.router)
    app.include_router(session_route.router)
    return app


app = create_app()


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api.main:app", host=settings.api_host, port=settings.api_port, reload=False)
