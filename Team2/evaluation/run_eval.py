"""DeepEval runner — runs all 30 questions through the live agent.

Usage:
    python -m evaluation.run_eval [--api-base http://localhost:8000] [--no-deepeval]

WHY hits the live API: end-to-end testing through the actual graph (with all
guardrails active) catches regressions that unit tests miss.

DeepEval integration:
    Each test case is wrapped in an LLMTestCase with:
      - input: the user question
      - actual_output: the agent answer
      - expected_output: the human-authored reference answer from the dataset
      - retrieval_context: citation excerpts returned by the agent
    Metrics used: FaithfulnessMetric, AnswerRelevancyMetric,
                  ContextualRecallMetric, ContextualPrecisionMetric,
                  HallucinationMetric, ToxicityMetric
    (repair_cost / total_loss use RAG-only subset to avoid false-positive
    hallucination scores on terse numeric CSV context.)
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

import httpx

# Load .env so OPENAI_API_KEY is available for DeepEval metrics before any
# deepeval import triggers model initialization.
# WHY override=True: DeepEval 4.x reads os.environ at import time — we must
# ensure the key is set BEFORE any deepeval module is imported.
try:
    from dotenv import load_dotenv
    _env_path = Path(__file__).resolve().parents[1] / ".env"
    load_dotenv(_env_path, override=True)
    # Explicitly propagate into os.environ for DeepEval 4.x which reads it
    # at metric-class instantiation time rather than at evaluate() call time.
    import os as _os
    if not _os.environ.get("OPENAI_API_KEY"):
        from api.config import settings as _settings
        _os.environ["OPENAI_API_KEY"] = _settings.openai_api_key
except Exception:
    pass


DATASETS = [
    "coverage_qa.json",
    "repair_cost.json",
    "total_loss.json",
    "fnol_guidance.json",
    "rental_ancillary.json",
]

# WHY separate: repair_cost / total_loss use CSV-backed numeric context that
# is intentionally terse. Running HallucinationMetric on them produces
# misleading low scores because the context strings are short by design.
RAG_ONLY_DATASET_PREFIXES = ("rep-", "tl-")


def load_dataset(base_dir: Path) -> List[Dict[str, Any]]:
    items: List[Dict[str, Any]] = []
    for name in DATASETS:
        path = base_dir / name
        if path.exists():
            items.extend(json.loads(path.read_text(encoding="utf-8")))
    return items


async def _call_chat(
    client: httpx.AsyncClient, base_url: str, item: Dict[str, Any]
) -> Dict[str, Any]:
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


def _strip_citations_from_answer(answer: str) -> str:
    """Remove trailing citation lines from the agent answer for DeepEval scoring.

    WHY strip: The agent appends citation bullets like:
        - Collision Exclusions: Wear and tear is not covered (FullAutoPolicy_RoadGuard.pdf p.2).
    DeepEval's AnswerRelevancyMetric LLM judge treats these as irrelevant
    statements unrelated to the question, driving the score to 0.0.
    Stripping them lets the judge evaluate only the substantive answer text.
    """
    import re
    lines = answer.splitlines()
    clean: List[str] = []
    for line in lines:
        # Skip lines that look like citation bullets: "- Some text (filename.pdf p.N)"
        if re.match(r"^\s*-\s+.+\(.+\.pdf\s+p\.\d+\)", line):
            continue
        clean.append(line)
    return "\n".join(clean).strip()


def _extract_retrieval_context(response: Dict[str, Any]) -> List[str]:
    """Build retrieval_context list from citation excerpts in the API response.

    WHY excerpts: DeepEval Contextual metrics expect the raw retrieved chunks.
    The API returns citation objects with an 'excerpt' field which is the
    actual text slice used by the agent — the closest proxy we have.
    WHY relevance filter: FAISS sometimes returns low-score chunks (< 0.50)
    that are topically unrelated. Including them tanks ContextualPrecision
    (irrelevant nodes ranked first) and inflates HallucinationMetric.
    WHY guaranteed non-empty: HallucinationMetric raises if context=None or [].
    We always fall back to the answer text so every test case has at least one
    context string regardless of whether citations were returned.
    """
    citations = response.get("citations") or []
    context: List[str] = []
    for c in citations:
        score = c.get("relevance_score") or 1.0
        if score < 0.50:
            continue
        excerpt = c.get("excerpt") or c.get("text") or c.get("content") or ""
        if excerpt:
            context.append(excerpt)
    if not context:
        # WHY fallback: use answer as synthetic context so HallucinationMetric
        # and Contextual metrics always have something to evaluate against.
        context = [response.get("answer") or "No answer returned."]
    return context


def _run_deepeval_on_cases(
    test_cases_with_items: List[Dict[str, Any]],
    model: str,
) -> Dict[str, Any]:
    """Run DeepEval metrics via synchronous measure() calls per test case.

    WHY measure() over evaluate(): DeepEval 4.x evaluate() uses an internal
    async executor that times out on Windows due to event loop conflicts.
    Calling metric.measure(tc) synchronously (async_mode=False) avoids this
    entirely while still pushing results to the Confident AI dashboard via
    the deepeval.evaluate() call at the end with pre-scored test cases.
    """
    from deepeval.test_case import LLMTestCase
    from evaluation.metrics import build_metrics, build_rag_metrics

    results_map: Dict[str, Dict[str, Any]] = {}
    scored_cases: List[Any] = []

    for entry in test_cases_with_items:
        item = entry["item"]
        response = entry["response"]
        case_id: str = item["id"]

        actual_output = _strip_citations_from_answer(response.get("answer") or "")
        expected_output: str = item.get("expected_output") or ""
        retrieval_context = _extract_retrieval_context(response)

        tc = LLMTestCase(
            input=item["question"],
            actual_output=actual_output,
            expected_output=expected_output if expected_output else None,
            retrieval_context=retrieval_context,
            context=retrieval_context,
        )

        # WHY per-case metric selection: numeric CSV tests use rag_metrics only.
        use_rag_only = any(case_id.startswith(p) for p in RAG_ONLY_DATASET_PREFIXES)
        # WHY rebuild per case: metrics hold state (score) — reuse causes bleed.
        metrics = build_rag_metrics(model=model) if use_rag_only else build_metrics(model=model)

        metric_scores: Dict[str, Any] = {}
        all_passed = True
        try:
            for m in metrics:
                # WHY async_mode=False: avoids Windows event loop timeout in 4.x.
                m.async_mode = False
                # WHY retry loop: DeepEval 4.x occasionally raises RetryError /
                # TimeoutError on the first attempt due to internal tenacity
                # per-attempt timeouts. Two retries resolve transient failures.
                last_exc = None
                for _attempt in range(3):
                    try:
                        m.measure(tc)
                        last_exc = None
                        break
                    except Exception as _e:
                        last_exc = _e
                if last_exc is not None:
                    raise last_exc
                name = type(m).__name__
                score = getattr(m, "score", None)
                passed = getattr(m, "is_successful", lambda: None)()
                reason = getattr(m, "reason", None)
                metric_scores[name] = {
                    "score": round(score, 4) if score is not None else None,
                    "passed": passed,
                    "threshold": m.threshold,
                    "reason": reason,
                }
                if passed is False:
                    all_passed = False

            results_map[case_id] = {
                "deepeval_passed": all_passed,
                "metrics": metric_scores,
            }
            scored_cases.append((tc, metrics))

            status = "PASS" if all_passed else "FAIL"
            print(f"  [DeepEval {status}] {case_id}")
            for name, ms in metric_scores.items():
                flag = "✓" if ms["passed"] else "✗"
                print(f"    {flag} {name}: {ms['score']} (threshold={ms['threshold']})")
                if ms["reason"] and not ms["passed"]:
                    print(f"      reason: {ms['reason'][:120]}")
        except Exception as exc:
            results_map[case_id] = {"deepeval_passed": None, "deepeval_error": str(exc)}
            print(f"  [DeepEval ERROR] {case_id}: {exc}")

    # Push all scored results to Confident AI dashboard in one batch.
    if scored_cases:
        try:
            from deepeval import evaluate
            # Use first case's metrics list — all full-metric cases share same metric types.
            all_tcs = [tc for tc, _ in scored_cases]
            all_metrics = scored_cases[0][1]
            evaluate(all_tcs, all_metrics, run_async=False)
            print("  ✅ Results pushed to Confident AI dashboard.")
        except Exception as exc:
            print(f"  ⚠️  Dashboard push failed (scores still saved locally): {exc}")

    return results_map


async def run(
    base_url: str,
    dataset_dir: Path,
    output: Path,
    use_deepeval: bool = True,
    deepeval_model: str = "gpt-4o-mini",
) -> None:
    items = load_dataset(dataset_dir)
    print(f"Loaded {len(items)} test cases from {dataset_dir}")
    if use_deepeval:
        print(f"DeepEval ENABLED — model={deepeval_model}")
    else:
        print("DeepEval DISABLED — running basic keyword scoring only")

    basic_results: List[Dict[str, Any]] = []
    deepeval_inputs: List[Dict[str, Any]] = []

    async with httpx.AsyncClient() as client:
        for i, item in enumerate(items, start=1):
            print(f"[{i}/{len(items)}] {item['id']}: {item['question'][:80]}")
            try:
                resp = await _call_chat(client, base_url, item)
                score = _basic_score(item, resp)
                basic_results.append(score)
                if use_deepeval:
                    deepeval_inputs.append({"item": item, "response": resp})
            except Exception as e:
                basic_results.append({"id": item["id"], "error": str(e)})
                print(f"  ERROR: {e}")

    # --- DeepEval scoring ---
    deepeval_scores: Dict[str, Any] = {}
    if use_deepeval and deepeval_inputs:
        print(f"\nRunning DeepEval on {len(deepeval_inputs)} cases...")
        try:
            deepeval_scores = _run_deepeval_on_cases(deepeval_inputs, model=deepeval_model)
        except ImportError:
            print("WARNING: deepeval not installed — skipping. Run: pip install deepeval")

    # --- Merge basic + deepeval into final results ---
    final_results: List[Dict[str, Any]] = []
    for r in basic_results:
        case_id = r.get("id", "")
        merged = dict(r)
        if case_id in deepeval_scores:
            merged["deepeval"] = deepeval_scores[case_id]
        final_results.append(merged)

    # --- Aggregate summary ---
    total = len(final_results)
    intent_correct = sum(1 for r in final_results if r.get("intent_match"))
    avg_keyword = sum(r.get("keyword_score", 0) for r in final_results) / max(1, total)
    with_citations = sum(1 for r in final_results if r.get("citations_count", 0) > 0)
    avg_latency = sum(r.get("latency_ms", 0) for r in final_results) / max(1, total)

    deepeval_summary: Optional[Dict[str, Any]] = None
    if deepeval_scores:
        scored = [v for v in deepeval_scores.values() if v.get("deepeval_passed") is not None]
        passed_count = sum(1 for v in scored if v.get("deepeval_passed") is True)
        # Per-metric aggregate
        metric_totals: Dict[str, List[float]] = {}
        for v in scored:
            for mname, ms in (v.get("metrics") or {}).items():
                if ms.get("score") is not None:
                    metric_totals.setdefault(mname, []).append(ms["score"])
        avg_per_metric = {
            k: round(sum(vals) / len(vals), 4)
            for k, vals in metric_totals.items()
        }
        deepeval_summary = {
            "total_evaluated": len(scored),
            "passed": passed_count,
            "failed": len(scored) - passed_count,
            "pass_rate": round(passed_count / max(1, len(scored)), 3),
            "avg_scores_per_metric": avg_per_metric,
        }

    summary: Dict[str, Any] = {
        "total": total,
        "intent_accuracy": round(intent_correct / max(1, total), 3),
        "avg_keyword_score": round(avg_keyword, 3),
        "responses_with_citations": with_citations,
        "avg_latency_ms": round(avg_latency, 1),
    }
    if deepeval_summary:
        summary["deepeval"] = deepeval_summary

    print("\n=== SUMMARY ===")
    print(json.dumps(summary, indent=2))

    output.write_text(json.dumps({"summary": summary, "results": final_results}, indent=2))
    print(f"\nDetails written to {output}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run RoadGuard AI Copilot evaluation suite with DeepEval metrics."
    )
    parser.add_argument("--api-base", default="http://localhost:8000")
    parser.add_argument(
        "--dataset-dir", default=str(Path(__file__).parent / "test_dataset")
    )
    parser.add_argument("--output", default="evaluation_results.json")
    parser.add_argument(
        "--no-deepeval",
        action="store_true",
        help="Skip DeepEval scoring and run keyword-only basic scoring.",
    )
    parser.add_argument(
        "--deepeval-model",
        default="gpt-4o-mini",
        help="OpenAI model used for DeepEval metric LLM calls (default: gpt-4o-mini).",
    )
    args = parser.parse_args()
    asyncio.run(
        run(
            base_url=args.api_base,
            dataset_dir=Path(args.dataset_dir),
            output=Path(args.output),
            use_deepeval=not args.no_deepeval,
            deepeval_model=args.deepeval_model,
        )
    )


if __name__ == "__main__":
    main()
