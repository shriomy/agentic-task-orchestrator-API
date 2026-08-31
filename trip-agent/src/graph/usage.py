"""Per-turn token/cost accounting.

Only 5 buckets exist, matching what the UI shows: context, memory,
system_prompt, tools, other. The first 4 describe what went into the *main
agent call's* prompt (see `categorize_agent_prompt`, which reuses the exact
same block-building helpers context.py uses, so the count matches what was
actually sent rather than a re-derived guess).

Everything else lands in "other" — deliberately, not by oversight: the
guardrail/utility model calls (classify_scope, split_request, preference
extraction, smalltalk, summarization) don't fit the 4 named categories, and
there is no dedicated "generation" bucket for the agent's own output tokens
either, so those land in "other" too.

Token counts are estimates via tiktoken, not the provider's own tokenizer —
close enough for a relative breakdown, not byte-exact for every provider.
Cost is only ever reported for a model in MODEL_PRICING; an unpriced model
shows tokens with cost_usd=None rather than a fabricated number.
"""

from __future__ import annotations

import json
import logging
from typing import Any

import tiktoken

from .context import _system_prompt, running_summary_text, selection_context
from .state import GraphState

logger = logging.getLogger(__name__)

# USD per 1M tokens: (input, output). Update when models or prices change —
# this is intentionally a small, maintained table, not exhaustive.
MODEL_PRICING: dict[str, tuple[float, float]] = {
    "gpt-4.1-mini": (0.40, 1.60),
    "gpt-4o-mini": (0.15, 0.60),
}

_DEFAULT_ENCODING = "cl100k_base"


def count_tokens(text: str, model: str | None = None) -> int:
    """Estimate token count for `text`. Never raises."""
    if not text:
        return 0
    try:
        encoding = tiktoken.encoding_for_model(model) if model else tiktoken.get_encoding(_DEFAULT_ENCODING)
    except KeyError:
        encoding = tiktoken.get_encoding(_DEFAULT_ENCODING)
    try:
        return len(encoding.encode(text))
    except Exception:
        return 0


def estimate_cost(model: str | None, input_tokens: int, output_tokens: int) -> float | None:
    """USD cost for this many tokens, or None if `model` isn't in MODEL_PRICING."""
    if not model or model not in MODEL_PRICING:
        return None
    input_rate, output_rate = MODEL_PRICING[model]
    return round((input_tokens * input_rate + output_tokens * output_rate) / 1_000_000, 6)


def _memory_text(state: GraphState) -> str:
    memory = state.user_memory_summary or {}
    preferences = memory.get("preferences") if isinstance(memory, dict) else None
    if not preferences:
        return ""
    lines = [f"- {key.replace('_', ' ')}: {value}" for key, value in sorted(preferences.items())]
    return "What you already know about this traveller (from earlier conversations):\n" + "\n".join(lines)


def _history_text(state: GraphState) -> str:
    from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

    parts = [
        str(getattr(message, "content", "") or "")
        for message in (state.messages or [])
        if isinstance(message, (HumanMessage, AIMessage, ToolMessage))
    ]
    return "\n".join(parts)


def _tools_text(tools: list[Any]) -> str:
    schemas = [
        {
            "name": getattr(tool, "name", ""),
            "description": getattr(tool, "description", "") or "",
            "parameters": getattr(tool, "args", {}) or {},
        }
        for tool in tools
    ]
    return json.dumps(schemas, default=str)


def categorize_agent_prompt(state: GraphState, tools: list[Any], model: str | None) -> dict[str, int]:
    """Token count per category for one AgentNode call.

    Rebuilds the same text blocks context.py's build_prompt() assembles
    (system prompt, running summary + selections digest as "context", the
    memory-preferences block, and the actual history messages) rather than
    inventing a separate representation, so the count tracks what was
    actually sent even as those helpers evolve.
    """
    system_text = _system_prompt(state.active_tools or [])
    memory_text = _memory_text(state)
    context_text = "\n\n".join(
        part for part in (running_summary_text(state), selection_context(state), _history_text(state)) if part
    )
    tools_text = _tools_text(tools)

    return {
        "system_prompt": count_tokens(system_text, model),
        "memory": count_tokens(memory_text, model),
        "context": count_tokens(context_text, model),
        "tools": count_tokens(tools_text, model),
    }


def _scale_to_total(breakdown: dict[str, int], target_total: int) -> dict[str, int]:
    """Proportionally scale an estimated breakdown to sum to an authoritative
    total (the provider-reported input_tokens), so the numbers shown stay
    internally consistent rather than drifting from the real total."""
    estimated_total = sum(breakdown.values())
    if estimated_total <= 0 or target_total <= 0:
        return breakdown
    scaled = {key: round(value * target_total / estimated_total) for key, value in breakdown.items()}
    # Rounding can leave the sum a token or two off; dump the remainder into
    # the largest bucket rather than leaving the displayed total wrong.
    drift = target_total - sum(scaled.values())
    if drift and scaled:
        biggest = max(scaled, key=lambda key: scaled[key])
        scaled[biggest] += drift
    return scaled


def agent_usage_entry(
    state: GraphState,
    tools: list[Any],
    model: str | None,
    response: Any,
) -> dict[str, Any]:
    """Build the usage_log entry for one AgentNode call."""
    breakdown = categorize_agent_prompt(state, tools, model)
    usage_metadata = getattr(response, "usage_metadata", None) or {}
    input_tokens = usage_metadata.get("input_tokens")
    output_tokens = usage_metadata.get("output_tokens")

    if input_tokens is None:
        # Fall back to estimating the whole prompt when the provider didn't
        # report usage — the category breakdown IS the estimate already, so
        # its own sum becomes the total.
        input_tokens = sum(breakdown.values())
    else:
        breakdown = _scale_to_total(breakdown, input_tokens)

    if output_tokens is None:
        output_tokens = count_tokens(str(getattr(response, "content", "") or ""), model)

    return {
        "node": "agent",
        "model": model,
        "system_prompt": breakdown["system_prompt"],
        "memory": breakdown["memory"],
        "context": breakdown["context"],
        "tools": breakdown["tools"],
        "other": output_tokens,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
    }


def other_usage_entry(node: str, model: str | None, input_tokens: int, output_tokens: int) -> dict[str, Any]:
    """Usage entry for a guardrail/utility call — everything lands in "other"."""
    return {
        "node": node,
        "model": model,
        "system_prompt": 0,
        "memory": 0,
        "context": 0,
        "tools": 0,
        "other": input_tokens + output_tokens,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
    }


def summarize_turn_usage(usage_log: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate a turn's usage_log entries into the shape persisted/streamed."""
    totals = {"context": 0, "memory": 0, "system_prompt": 0, "tools": 0, "other": 0}
    cost_total = 0.0
    cost_known = True
    model: str | None = None

    for entry in usage_log or []:
        for key in totals:
            totals[key] += int(entry.get(key, 0) or 0)
        entry_model = entry.get("model")
        if entry_model:
            model = model or entry_model
        cost = estimate_cost(entry_model, int(entry.get("input_tokens", 0) or 0), int(entry.get("output_tokens", 0) or 0))
        if cost is None:
            cost_known = False
        else:
            cost_total += cost

    total_tokens = sum(totals.values())
    return {
        "context_tokens": totals["context"],
        "memory_tokens": totals["memory"],
        "system_prompt_tokens": totals["system_prompt"],
        "tools_tokens": totals["tools"],
        "other_tokens": totals["other"],
        "total_tokens": total_tokens,
        "cost_usd": round(cost_total, 6) if (cost_known and usage_log) else None,
        "model": model,
    }
