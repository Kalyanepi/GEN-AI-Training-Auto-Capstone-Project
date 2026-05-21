"""Async HTTP client wrapping the FastAPI backend."""
from __future__ import annotations

import os
from typing import Any, Dict, Optional

import httpx


def get_api_base_url() -> str:
    return os.getenv("API_BASE_URL", "http://localhost:8000")


class ApiClient:
    def __init__(self, base_url: Optional[str] = None, timeout: float = 60.0) -> None:
        self.base_url = (base_url or get_api_base_url()).rstrip("/")
        self.timeout = timeout

    async def chat(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            resp = await client.post(f"{self.base_url}/api/v1/chat", json=payload)
            resp.raise_for_status()
            return resp.json()

    async def clear_session(self, session_id: str) -> bool:
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            try:
                resp = await client.delete(f"{self.base_url}/api/v1/session/{session_id}")
                return resp.status_code == 200
            except httpx.HTTPError:
                return False

    async def health(self) -> Dict[str, Any]:
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.get(f"{self.base_url}/health")
            resp.raise_for_status()
            return resp.json()


def chat_sync(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Synchronous wrapper — Streamlit's runtime is sync-friendly.

    WHY sync wrapper: streamlit reruns the script top-to-bottom on each
    interaction; nested asyncio in that model is awkward. httpx sync client
    is the cleanest path.
    """
    base = get_api_base_url().rstrip("/")
    with httpx.Client(timeout=60.0) as client:
        resp = client.post(f"{base}/api/v1/chat", json=payload)
        resp.raise_for_status()
        return resp.json()


def clear_session_sync(session_id: str) -> bool:
    base = get_api_base_url().rstrip("/")
    try:
        with httpx.Client(timeout=10.0) as client:
            resp = client.delete(f"{base}/api/v1/session/{session_id}")
            return resp.status_code == 200
    except httpx.HTTPError:
        return False


def health_sync() -> Dict[str, Any]:
    base = get_api_base_url().rstrip("/")
    try:
        with httpx.Client(timeout=10.0) as client:
            resp = client.get(f"{base}/health")
            resp.raise_for_status()
            return resp.json()
    except Exception as e:
        return {"status": "unreachable", "error": str(e)}
