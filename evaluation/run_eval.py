"""DeepEval runner — runs all 30 questions through the live agent.

Usage:
    python -m evaluation.run_eval [--api-base http://localhost:8000]

WHY hits the live API: end-to-end testing through the actual graph (with all
guardrails active) catches regressions that unit tests miss.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
import uuid
from pathlib import Path
from typing import Any, Dict, List

import httpx


DATASETS = [
    "coverage_qa.json",
    "repair_cost.json",
    "total_loss.json",
    "fnol_guidance.json",
    "rental_ancillary.json",
]


def load_dataset(base_dir: Path) -> List[Dict[str, Any]]:
    items: List[Dict[str, Any]] = []
    for name in DATASETS:
        path = base_dir / name
        if path.exists():
            items.extend(json.loads(path.read_text(encoding="utf-8")))
    return items


async def _call_chat(client: httpx.AsyncClient, base_url: str, item: Dict[str, Any]) -> Dict[str, Any]:
    payload = {
        "session_id": str(uuid.uuid4()),
        "message": item["question"],
        "policy_tier": item.get("policy_tier"),
        "coverage_type": item.get("coverage_type"),
        "vehicle_category": item.get("vehicle_category"),
        "state_code": item.get("state_code"),
        "vehicle_year": item.get("vehicle_year"),
        "acv": item.get("acv"),
        "repair_cost": item.get("repair_cost"),
    }
    payload = {k: v for k, v in payload.items() if v is not None}
    resp = await client.post(f"{base_url}/api/v1/chat", json=payload, timeout=60.0)
    resp.raise_for_status()
    return resp.json()


def _basic_score(item: Dict[str, Any], response: Dict[str, Any]) -> Dict[str, Any]:
    """Lightweight scoring that runs WITHOUT deepeval — keyword + intent checks.

    WHY a fallback scorer: deepeval requires its own OPENAI calls and may not
    be installed in every environment. Basic scoring still verifies the agent
    is producing on-topic, citation-bearing answers.
    """
    answer = (response.get("answer") or "").lower()
    expected_kws = [k.lower() for k in (item.get("expected_keywords") or [])]
    matched = [k for k in expected_kws if k in answer]
    keyword_score = len(matched) / max(1, len(expected_kws))
    intent_match = (
        item.get("expected_intent") is None
        or response.get("intent_detected") == item.get("expected_intent")
    )
    return {
        "id": item["id"],
        "intent_match": intent_match,
        "intent_expected": item.get("expected_intent"),
        "intent_actual": response.get("intent_detected"),
        "keywords_expected": expected_kws,
        "keywords_matched": matched,
        "keyword_score": round(keyword_score, 2),
        "citations_count": len(response.get("citations") or []),
        "guardrail_triggered": response.get("guardrail_triggered", False),
        "latency_ms": response.get("latency_ms", 0),
        "answer_preview": (response.get("answer") or "")[:200],
    }


async def run(base_url: str, dataset_dir: Path, output: Path) -> None:
    items = load_dataset(dataset_dir)
    print(f"Loaded {len(items)} test cases from {dataset_dir}")

    results: List[Dict[str, Any]] = []
    async with httpx.AsyncClient() as client:
        for i, item in enumerate(items, start=1):
            print(f"[{i}/{len(items)}] {item['id']}: {item['question'][:80]}")
            try:
                resp = await _call_chat(client, base_url, item)
                results.append(_basic_score(item, resp))
            except Exception as e:
                results.append({"id": item["id"], "error": str(e)})
                print(f"  ERROR: {e}")

    # Aggregate.
    total = len(results)
    intent_correct = sum(1 for r in results if r.get("intent_match"))
    avg_keyword = sum(r.get("keyword_score", 0) for r in results) / max(1, total)
    with_citations = sum(1 for r in results if r.get("citations_count", 0) > 0)
    avg_latency = sum(r.get("latency_ms", 0) for r in results) / max(1, total)

    summary = {
        "total": total,
        "intent_accuracy": round(intent_correct / max(1, total), 3),
        "avg_keyword_score": round(avg_keyword, 3),
        "responses_with_citations": with_citations,
        "avg_latency_ms": round(avg_latency, 1),
    }
    print("\n=== SUMMARY ===")
    print(json.dumps(summary, indent=2))

    output.write_text(json.dumps({"summary": summary, "results": results}, indent=2))
    print(f"\nDetails written to {output}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--api-base", default="http://localhost:8000")
    parser.add_argument("--dataset-dir", default=str(Path(__file__).parent / "test_dataset"))
    parser.add_argument("--output", default="evaluation_results.json")
    args = parser.parse_args()
    asyncio.run(run(args.api_base, Path(args.dataset_dir), Path(args.output)))


if __name__ == "__main__":
    main()
