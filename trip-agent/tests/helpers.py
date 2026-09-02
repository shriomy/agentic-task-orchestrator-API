"""Shared test doubles."""

from __future__ import annotations

from typing import Any


class ScriptedModel:
    """Returns pre-baked AIMessages in order, recording what it was asked."""

    def __init__(self, responses: list[Any]) -> None:
        self.responses = list(responses)
        self.calls: list[list[Any]] = []

    def bind_tools(self, tools: Any, **kwargs: Any) -> "ScriptedModel":
        self.bound_tools = [getattr(t, "name", str(t)) for t in tools]
        return self

    def with_structured_output(self, schema: Any, **kwargs: Any) -> "ScriptedModel":
        self.structured_schema = schema
        return self

    def invoke(self, messages: Any, config: Any = None, **kwargs: Any) -> Any:
        self.calls.append(messages)
        if not self.responses:
            from langchain_core.messages import AIMessage

            return AIMessage(content="(script exhausted)")
        return self.responses.pop(0)


class AlwaysSearchModel:
    """Stub for `fast_model()` in ToolSearchNode: always calls `search_tools`
    with the request text unchanged, reproducing the old always-retrieve
    behavior for tests that aren't specifically exercising the search-vs-answer
    gate itself."""

    def bind_tools(self, tools: Any, **kwargs: Any) -> "AlwaysSearchModel":
        return self

    def invoke(self, messages: Any, config: Any = None, **kwargs: Any) -> Any:
        from langchain_core.messages import AIMessage

        content = str(messages[-1].content) if messages else ""
        query = content.split("current request:", 1)[-1].strip() or content
        return AIMessage(
            content="",
            tool_calls=[{"name": "search_tools", "args": {"query": query}, "id": "search-1", "type": "tool_call"}],
        )
