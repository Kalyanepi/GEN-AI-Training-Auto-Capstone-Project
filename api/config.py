"""Central configuration via pydantic-settings.

WHY: Every threshold, model name, path, and limit is an env var with a sensible
default. Team members tune behavior by editing .env, never by editing code.
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # --- OpenAI ---
    openai_api_key: str = Field(default="sk-replace-me")
    openai_embedding_model: str = "text-embedding-3-small"
    openai_router_model: str = "gpt-4o-mini"
    openai_synthesis_model: str = "gpt-4o"
    openai_guardrail_model: str = "gpt-4o-mini"

    # --- LangSmith ---
    langchain_tracing_v2: bool = True
    langchain_api_key: str = Field(default="lsv2_replace-me")
    langchain_project: str = "roadguard-copilot-dev"
    langchain_endpoint: str = "https://api.smith.langchain.com"

    # --- Paths ---
    data_dir: Path = Path("./data")
    pdf_dir: Path = Path("./data/pdfs")
    faiss_index_dir: Path = Path("./data/faiss_index")
    repair_cost_csv: Path = Path("./data/RepairCost_ReferenceTable.csv")
    total_loss_csv: Path = Path("./data/TotalLoss_Threshold_Table.csv")

    # --- RAG ---
    chunk_size_tokens: int = 500
    chunk_overlap_tokens: int = 75
    embedding_batch_size: int = 100
    retrieval_top_k: int = 5
    retrieval_overfetch_multiplier: int = 4
    # WHY 0.35: with text-embedding-3-small + cosine similarity, typical
    # in-domain matches in our 60-chunk index score 0.35-0.55. The original
    # 0.65 was tuned for a much larger index and was filtering out legitimate
    # results. Tune via env var SIMILARITY_THRESHOLD if your index grows.
    similarity_threshold: float = 0.35
    reranker_enabled: bool = True
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"

    # --- Agent / Memory ---
    conversation_window_k: int = 10
    session_ttl_minutes: int = 30
    max_citations_per_response: int = 4
    citation_excerpt_max_chars: int = 150

    # --- Cache layer (LRU) ---
    # Router decisions and query embeddings are deterministic for a given input,
    # so caching them eliminates redundant OpenAI calls for repeated questions.
    router_cache_size: int = 256
    router_cache_ttl_seconds: int = 3600   # 1 hour; covers a typical session
    embed_cache_size: int = 512
    embed_cache_ttl_seconds: int = 3600
    retrieval_cache_size: int = 256
    retrieval_cache_ttl_seconds: int = 1800   # 30min; ingested chunks rarely change in-session

    # --- Guardrails ---
    fuzzy_damage_match_threshold: float = 0.60
    fabricated_cost_tolerance_pct: float = 10.0
    adjuster_phone: str = "1-800-555-0601"
    adjuster_url: str = "roadguardauto.com/claims"

    # --- API ---
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    api_base_url: str = "http://localhost:8000"
    rate_limit_per_minute: int = 30
    request_timeout_seconds: int = 30

    # --- UI ---
    ui_host: str = "0.0.0.0"
    ui_port: int = 8501

    # --- Logging ---
    log_level: str = "INFO"
    log_format: str = "json"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Singleton accessor. WHY: avoids re-reading .env on every call."""
    return Settings()


settings = get_settings()
