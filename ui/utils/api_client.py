"""Async HTTP client wrapping the FastAPI backend."""
from __future__ import annotations

import json
import os
from typing import Any, Dict, Generator, Optional, Tuple

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
        with httpx.Client(timeout=3.0) as client:
            resp = client.delete(f"{base}/api/v1/session/{session_id}")
            return resp.status_code == 200
    except Exception:
        return False


def stream_chat_sync(
    payload: Dict[str, Any],
) -> Generator[Tuple[str, Any], None, None]:
    """Consume the SSE /chat/stream endpoint and yield (event_type, data) tuples.

    Yields:
      ("meta",     dict)  — intent/tools/citations, emitted before tokens
      ("token",    str)   — incremental text chunk for st.write_stream()
      ("guardrail",dict)  — if output guardrail blocks the answer
      ("done",     dict)  — latency/confidence/disclaimer
    """
    base = get_api_base_url().rstrip("/")
    with httpx.Client(timeout=120.0) as client:
        with client.stream("POST", f"{base}/api/v1/chat/stream", json=payload) as resp:
            resp.raise_for_status()
            event_type = "message"
            for raw_line in resp.iter_lines():
                line = raw_line.strip()
                if line.startswith("event:"):
                    event_type = line[len("event:"):].strip()
                elif line.startswith("data:"):
                    data_str = line[len("data:"):].strip()
                    try:
                        data = json.loads(data_str)
                    except json.JSONDecodeError:
                        data = data_str
                    if event_type == "token":
                        yield ("token", data.get("token", "") if isinstance(data, dict) else data)
                    else:
                        yield (event_type, data)
                    event_type = "message"


def health_sync() -> Dict[str, Any]:
    base = get_api_base_url().rstrip("/")
    try:
        with httpx.Client(timeout=10.0) as client:
            resp = client.get(f"{base}/health")
            resp.raise_for_status()
            return resp.json()
    except Exception as e:
        return {"status": "unreachable", "error": str(e)}
