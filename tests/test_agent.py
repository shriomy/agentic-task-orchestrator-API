from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from src.agent import Agent, build_tool_definitions
from src.state import AgentState


@dataclass
class FakeLLMClient:
    responses: list[dict[str, Any]]

    def chat(self, *, messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None = None) -> dict[str, Any]:
        if not self.responses:
            raise AssertionError("No fake LLM responses left")
        return self.responses.pop(0)


def make_agent(responses: list[dict[str, Any]]) -> Agent:
    tools, tool_functions = build_tool_definitions()
    return Agent(llm_client=FakeLLMClient(responses), tools=tools, tool_functions=tool_functions, max_iterations=5)


def test_no_tool_final_answer() -> None:
    agent = make_agent([{"role": "assistant", "content": "An API is an interface for software communication."}])
    result = agent.run("Explain what an API is.", state=AgentState())
    assert "interface" in result.answer


def test_weather_tool_loop() -> None:
    agent = make_agent([
        {"role": "assistant", "tool_calls": [{"id": "call_1", "type": "function", "function": {"name": "get_weather", "arguments": '{"latitude": 6.9271, "longitude": 79.8612}'}}]},
        {"role": "assistant", "content": "It is sunny in Colombo."},
    ])
    agent.tool_functions["get_weather"] = lambda args: {"current_weather": {"temperature": 30}}
    result = agent.run("What is the current weather in Colombo?", state=AgentState())
    assert "sunny" in result.answer


def test_unknown_tool_becomes_tool_error() -> None:
    agent = make_agent([
        {"role": "assistant", "tool_calls": [{"id": "call_1", "type": "function", "function": {"name": "missing_tool", "arguments": "{}"}}]},
        {"role": "assistant", "content": "I cannot use that tool."},
    ])
    result = agent.run("Use a missing tool.", state=AgentState())
    assert "cannot" in result.answer.lower()


def test_multiple_tools_in_sequence() -> None:
    agent = make_agent(
        [
            {
                "role": "assistant",
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {"name": "get_weather", "arguments": '{"latitude": 6.9271, "longitude": 79.8612}'},
                    },
                    {
                        "id": "call_2",
                        "type": "function",
                        "function": {"name": "web_search", "arguments": '{"query": "Sri Lanka weather news"}'},
                    },
                ],
            },
            {"role": "assistant", "content": "It is warm in Colombo, and there are several related news stories."},
        ]
    )
    agent.tool_functions["get_weather"] = lambda args: {"current_weather": {"temperature": 30}}
    agent.tool_functions["web_search"] = lambda args: {"results": [{"title": "Weather update", "url": "https://example.com"}]}
    result = agent.run("What is the current weather in Colombo and what are the latest weather-related news stories in Sri Lanka?", state=AgentState())
    assert "news" in result.answer.lower()


def test_tool_failure_is_captured() -> None:
    agent = make_agent(
        [
            {
                "role": "assistant",
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {"name": "get_weather", "arguments": '{"latitude": 6.9271, "longitude": 79.8612}'},
                    }
                ],
            },
            {"role": "assistant", "content": "I could not fetch the weather right now."},
        ]
    )

    def failing_weather(args: dict[str, Any]) -> dict[str, Any]:
        raise RuntimeError("simulated failure")

    agent.tool_functions["get_weather"] = failing_weather
    result = agent.run("What is the current weather in Colombo?", state=AgentState())
    assert "could not" in result.answer.lower()


def test_conversation_state_is_preserved() -> None:
    state = AgentState()
    agent = make_agent([{"role": "assistant", "content": "Colombo is 30 degrees right now."}])
    first = agent.run("What is the weather in Colombo?", state=state)
    assert first.answer
    assert len(state.messages) >= 3

    follow_up_agent = make_agent([{"role": "assistant", "content": "Tomorrow should be slightly cooler."}])
    second = follow_up_agent.run("What about tomorrow?", state=state)
    assert "tomorrow" in second.answer.lower()
