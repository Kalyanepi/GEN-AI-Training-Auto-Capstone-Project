"""FastAPI app factory + lifespan events.

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
    logger.info("api_startup_complete")
    yield
    logger.info("api_shutdown")


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
