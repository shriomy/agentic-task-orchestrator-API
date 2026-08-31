"""Per-turn token/cost usage accounting (graph/usage.py)."""

from __future__ import annotations

from src.graph import usage
from src.graph.state import GraphState


def test_count_tokens_handles_a_known_and_an_unknown_model():
    assert usage.count_tokens("hello there, traveller", "gpt-4o-mini") > 0
    assert usage.count_tokens("hello there, traveller", "some-model-nobody-heard-of") > 0


def test_count_tokens_is_zero_for_empty_text():
    assert usage.count_tokens("", "gpt-4o-mini") == 0


def test_estimate_cost_is_none_for_an_unpriced_model():
    assert usage.estimate_cost("some-model-nobody-heard-of", 1000, 500) is None


def test_estimate_cost_is_numeric_for_a_priced_model():
    cost = usage.estimate_cost("gpt-4o-mini", 1_000_000, 1_000_000)
    assert cost == usage.MODEL_PRICING["gpt-4o-mini"][0] + usage.MODEL_PRICING["gpt-4o-mini"][1]


def test_categorize_agent_prompt_returns_all_four_keys():
    state = GraphState(
        thread_id="t1",
        user_id="u1",
        active_tools=["web_search"],
        user_memory_summary={"preferences": {"home_city": "Kyoto"}},
    )
    breakdown = usage.categorize_agent_prompt(state, tools=[], model="gpt-4o-mini")
    assert set(breakdown) == {"system_prompt", "memory", "context", "tools"}
    assert breakdown["system_prompt"] > 0  # the static prompt is never empty
    assert breakdown["memory"] > 0  # a preference was set above


def test_categorize_agent_prompt_memory_is_zero_without_preferences():
    state = GraphState(thread_id="t1", user_id="u1", active_tools=[])
    breakdown = usage.categorize_agent_prompt(state, tools=[], model="gpt-4o-mini")
    assert breakdown["memory"] == 0


def test_other_usage_entry_puts_everything_in_other():
    entry = usage.other_usage_entry("classify_scope", "gpt-4o-mini", 100, 20)
    assert entry["other"] == 120
    assert entry["context"] == entry["memory"] == entry["system_prompt"] == entry["tools"] == 0
    assert entry["input_tokens"] == 100
    assert entry["output_tokens"] == 20


def test_summarize_turn_usage_aggregates_across_entries():
    log = [
        usage.other_usage_entry("classify_scope", "gpt-4o-mini", 50, 10),
        {
            "node": "agent",
            "model": "gpt-4.1-mini",
            "system_prompt": 200,
            "memory": 0,
            "context": 100,
            "tools": 50,
            "other": 30,
            "input_tokens": 350,
            "output_tokens": 30,
        },
    ]
    summary = usage.summarize_turn_usage(log)
    assert summary["system_prompt_tokens"] == 200
    assert summary["context_tokens"] == 100
    assert summary["tools_tokens"] == 50
    assert summary["other_tokens"] == 30 + 60  # agent's output + classify_scope's 60
    assert summary["total_tokens"] == sum(
        summary[k] for k in ("context_tokens", "memory_tokens", "system_prompt_tokens", "tools_tokens", "other_tokens")
    )
    assert summary["cost_usd"] is not None  # both models are priced
    assert summary["model"] == "gpt-4o-mini"  # first entry's model, in order


def test_summarize_turn_usage_cost_is_none_when_any_model_is_unpriced():
    log = [usage.other_usage_entry("classify_scope", "mystery-model", 50, 10)]
    summary = usage.summarize_turn_usage(log)
    assert summary["cost_usd"] is None


def test_summarize_turn_usage_of_empty_log_is_all_zero_no_cost():
    summary = usage.summarize_turn_usage([])
    assert summary["total_tokens"] == 0
    assert summary["cost_usd"] is None
