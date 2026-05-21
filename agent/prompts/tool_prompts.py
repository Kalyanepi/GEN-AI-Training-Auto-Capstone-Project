"""Synthesis prompt builder — assembles tool results + retrieved chunks into
the final LLM input.

WHY one builder rather than per-tool prompts: every synthesis call follows the
same shape (system + session context + tool data + retrieved chunks + user
question). Centralizing prevents inconsistent prompt structures.
"""
from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

from agent.state import ToolInvocation


def _format_tool_result(t: ToolInvocation) -> str:
    """Format one tool's result for the LLM context. Truncate large arrays."""
    name = t.get("tool_name", "tool")
    if not t.get("success"):
        return f"### Tool: {name}\nStatus: FAILED\nFallback: {t.get('fallback_message') or t.get('error')}\n"
    data = t.get("data") or {}
    # WHY: nested 'chunks' arrays already get rendered separately under
    # "Retrieved Policy Chunks" — strip them from the structured tool data
    # to avoid duplicate massive context.
    pruned = {k: v for k, v in data.items() if k != "chunks"}
    return f"### Tool: {name}\n```json\n{json.dumps(pruned, indent=2, default=str)}\n```\n"


def _format_chunks(tool_results: List[ToolInvocation]) -> str:
    """Collect all retrieved chunks across tools into one cited block."""
    blocks: List[str] = []
    seen_ids = set()
    for t in tool_results:
        chunks = (t.get("data") or {}).get("chunks") or []
        for c in chunks:
            cid = c.get("chunk_id") or f"{c.get('source_file')}:{c.get('page_number')}:{c.get('section_title')}"
            if cid in seen_ids:
                continue
            seen_ids.add(cid)
            section = c.get("section_title") or ""
            blocks.append(
                f"[{c.get('source_file')} p.{c.get('page_number')} | {section} | "
                f"score={c.get('similarity_score', 0):.2f}]\n{c.get('text','').strip()}"
            )
    if not blocks:
        return ""
    return "## Retrieved Policy Chunks\n" + "\n\n".join(blocks)


def build_synthesis_messages(
    system_prompt: str,
    user_message: str,
    session_context: Dict[str, Any],
    tool_results: List[ToolInvocation],
    history: Optional[List[Dict[str, str]]] = None,
) -> List[Dict[str, str]]:
    """Build the messages array for the synthesis LLM call."""
    sc_lines = []
    for k in ("policy_tier", "vehicle_category", "state_code", "coverage_type", "vehicle_year", "acv", "repair_cost"):
        v = session_context.get(k)
        if v not in (None, ""):
            sc_lines.append(f"- {k}: {v}")
    session_context_block = "## Session Context\n" + ("\n".join(sc_lines) if sc_lines else "- (none provided)")

    tools_block = "## Tool Results\n" + (
        "\n".join(_format_tool_result(t) for t in tool_results) if tool_results else "(no tools invoked)"
    )
    chunks_block = _format_chunks(tool_results)

    context = "\n\n".join(filter(None, [session_context_block, tools_block, chunks_block]))

    instruction = (
        "Using ONLY the data and chunks above, answer the user's question. "
        "Quote tier-correct figures only. Cite sources implicitly by referencing "
        "section titles and page numbers — explicit citation cards are rendered "
        "separately by the system. If the data does not address the question, "
        "say so and direct to the adjuster."
    )

    messages: List[Dict[str, str]] = [{"role": "system", "content": system_prompt}]
    if history:
        # WHY: include short history for follow-ups ("my car" -> Camry).
        messages.extend(history[-10:])
    messages.append({
        "role": "user",
        "content": f"{context}\n\n## User Question\n{user_message}\n\n## Instruction\n{instruction}",
    })
    return messages
