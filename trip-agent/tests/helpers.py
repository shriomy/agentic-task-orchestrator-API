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
