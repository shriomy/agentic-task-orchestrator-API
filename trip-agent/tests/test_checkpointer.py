"""Checkpoint serialization: every type reachable from GraphState must survive
a real round trip through the checkpointer's serializer, not just the default
in-memory saver's permissive fallback.

Regression: `RunningSummary` (langmem's summary dataclass, held in
`state.running_summary`) was missing from the serializer's msgpack allowlist.
It deserialized fine on the FIRST use (never round-tripped yet), but once a
summary existed and the checkpoint was written then reloaded, it came back as a
plain dict — and passing a dict into `summarize_messages()` blew up with
`'dict' object has no attribute 'summarized_message_ids'` on the next
summarization attempt for that thread, permanently breaking it (caught and
logged as a warning, so the thread degraded to "keep full history" forever
rather than crashing outright, but summarization never worked again).
"""

from __future__ import annotations

from langgraph.checkpoint.memory import InMemorySaver

from src.graph.graph import build_graph
from src.memory.checkpointer import build_serializer

USER = "checkpoint-test-user"


def test_build_serializer_includes_running_summary_in_the_allowlist():
    from langmem.short_term.summarization import RunningSummary

    from src.graph.state import Selection, SelectionOption

    serde = build_serializer()
    allowed = getattr(serde, "_allowed_msgpack_modules", None)
    assert allowed is True or allowed is None or {
        (t.__module__, t.__qualname__) if isinstance(t, type) else t for t in allowed
    } >= {
        (RunningSummary.__module__, RunningSummary.__qualname__),
        (Selection.__module__, Selection.__qualname__),
        (SelectionOption.__module__, SelectionOption.__qualname__),
    }


class _FakeSummaryModel:
    """Stands in for the real summarizer LLM langmem calls internally.

    `summarize_messages()` invokes `model.invoke(prompt_messages)` itself (it
    is not routed through AgentNode), so patching `fast_model()` is what's
    needed here — a `ScriptedModel` for the agent's own replies is not enough.
    """

    def invoke(self, prompt, config=None, **kwargs):
        from langchain_core.messages import AIMessage

        return AIMessage(content="Summary of the conversation so far.")


def test_a_running_summary_survives_a_real_checkpoint_round_trip(monkeypatch, no_memory):
    """Force summarization every turn and drive several turns through a graph
    built with the REAL serializer (not the bare InMemorySaver default), so the
    summary is actually written to and read back from the checkpoint store."""
    from langchain_core.messages import AIMessage

    monkeypatch.setattr(
        "src.graph.nodes.classify_scope",
        lambda message, context="": __import__("src.guardrails.scope", fromlist=["ScopeVerdict"]).ScopeVerdict(
            label="in_scope", reason="test"
        ),
    )
    monkeypatch.setattr(
        "src.graph.nodes.split_request",
        lambda message: __import__("src.guardrails.output", fromlist=["RequestSplit"]).RequestSplit(
            allowed_request=message, withheld=[]
        ),
    )
    monkeypatch.setattr("src.graph.nodes.settings.summary_token_threshold", 600)
    # summarize_messages() calls this itself to produce the summary text — must
    # not hit a real provider in a unit test.
    monkeypatch.setattr("src.graph.nodes.fast_model", lambda: _FakeSummaryModel())

    from helpers import ScriptedModel

    # Long enough that a couple of turns' worth of history crosses the 600-token
    # budget on real approximate token counting, without needing a real LLM.
    padding = " This place has wonderful gardens, temples, and seasonal festivals worth visiting." * 12
    model = ScriptedModel([AIMessage(content=f"Reply {i}.{padding}") for i in range(10)])
    graph = build_graph(checkpointer=InMemorySaver(serde=build_serializer()))
    for node in graph.nodes.values():
        for attribute in ("runnable", "bound", "func"):
            candidate = getattr(node, attribute, None)
            for inner in (candidate, getattr(candidate, "func", None)):
                if type(inner).__name__ == "AgentNode":
                    inner._model = model

    config = {"configurable": {"thread_id": "rt-1", "auth_user_id": USER}}
    seen_types: list[str] = []
    for i in range(5):
        result = graph.invoke(
            {"thread_id": "rt-1", "user_id": USER, "message": f"turn {i} about Kyoto's places and events"},
            config=config,
        )
        seen_types.append(type(result.get("running_summary")).__name__)
        assert result.get("bot_response"), f"turn {i} produced no answer"

    # Once a summary is created, every later turn must keep getting a real
    # RunningSummary back — never the bare dict that caused the crash.
    first_summary_index = next((i for i, t in enumerate(seen_types) if t == "RunningSummary"), None)
    assert first_summary_index is not None, f"summarization never triggered: {seen_types}"
    assert all(t == "RunningSummary" for t in seen_types[first_summary_index:]), seen_types

    final_summary = graph.get_state(config).values.get("running_summary")
    assert hasattr(final_summary, "summarized_message_ids")
