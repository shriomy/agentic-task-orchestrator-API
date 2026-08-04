from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from src.llm import LLMClient, parse_tool_arguments
from src.state import AgentState
from src.tool_registry import ToolDefinition, ToolFunction, build_tool_schema


@dataclass
class AgentResult:
    answer: str
    state: AgentState


class AgentError(RuntimeError):
    pass


class Agent:
    def __init__(self, *, llm_client: LLMClient, tools: list[ToolDefinition], tool_functions: dict[str, ToolFunction], max_iterations: int = 10) -> None:
        self.llm_client = llm_client
        self.tools = tools
        self.tool_functions = tool_functions
        self.max_iterations = max_iterations

    def run(self, user_input: str, state: AgentState | None = None) -> AgentResult:
        active_state = state or AgentState()
        if not active_state.messages:
            active_state.add_message(
                "system",
                "You are a helpful assistant that can decide when to use tools. Use tools only when needed and always return a concise final answer when done.",
            )
        active_state.add_message("user", user_input)
        active_state.add_trace("user_message_received", content=user_input)

        while active_state.iteration < self.max_iterations:
            active_state.iteration += 1
            active_state.add_trace("llm_request", iteration=active_state.iteration, message_count=len(active_state.messages), tool_count=len(self.tools))
            llm_message = self.llm_client.chat(
                messages=active_state.messages,
                tools=[build_tool_schema(tool) for tool in self.tools],
            )
            active_state.messages.append(llm_message)
            active_state.add_trace(
                "llm_response",
                iteration=active_state.iteration,
                has_tool_calls=bool(llm_message.get("tool_calls")),
                content=llm_message.get("content"),
            )

            tool_calls = llm_message.get("tool_calls") or []
            if not tool_calls:
                content = llm_message.get("content")
                if not isinstance(content, str):
                    raise AgentError("LLM returned no final content")
                active_state.add_trace("final_answer", iteration=active_state.iteration)
                return AgentResult(answer=content, state=active_state)

            active_state.tool_calls.extend(tool_calls)
            for tool_call in tool_calls:
                tool_name = tool_call.get("function", {}).get("name")
                tool_call_id = tool_call.get("id")
                raw_arguments = tool_call.get("function", {}).get("arguments")
                if not isinstance(tool_name, str) or not isinstance(tool_call_id, str):
                    raise AgentError("Malformed tool call from LLM")
                active_state.add_trace("tool_call_requested", iteration=active_state.iteration, tool_name=tool_name, tool_call_id=tool_call_id, arguments=raw_arguments)
                tool_function = self.tool_functions.get(tool_name)
                if tool_function is None:
                    tool_result = {"error": f"Unknown tool requested: {tool_name}"}
                    active_state.last_tool_error = tool_result["error"]
                    active_state.add_trace("tool_call_failed", iteration=active_state.iteration, tool_name=tool_name, reason="unknown_tool")
                else:
                    try:
                        parsed_arguments = parse_tool_arguments(raw_arguments)
                        active_state.add_trace("tool_arguments_parsed", iteration=active_state.iteration, tool_name=tool_name, arguments=parsed_arguments)
                        tool_result = tool_function(parsed_arguments)
                        active_state.last_tool_name = tool_name
                        active_state.last_tool_error = None
                        active_state.add_trace("tool_call_completed", iteration=active_state.iteration, tool_name=tool_name, result=tool_result)
                    except Exception as exc:
                        tool_result = {"error": str(exc)}
                        active_state.last_tool_name = tool_name
                        active_state.last_tool_error = str(exc)
                        active_state.add_trace("tool_call_failed", iteration=active_state.iteration, tool_name=tool_name, reason=str(exc))
                active_state.add_message("tool", content=str(tool_result), tool_call_id=tool_call_id)
                active_state.add_trace("tool_result_appended", iteration=active_state.iteration, tool_name=tool_name, tool_call_id=tool_call_id)

        active_state.add_trace("max_iterations_reached", max_iterations=self.max_iterations)
        raise AgentError(f"Maximum iterations reached: {self.max_iterations}")


def build_tool_definitions() -> tuple[list[ToolDefinition], dict[str, ToolFunction]]:
    from src.tools.tavily_search import web_search, web_search_definition
    from src.tools.weather import get_weather, get_weather_definition

    tools = [ToolDefinition(**web_search_definition), ToolDefinition(**get_weather_definition)]
    tool_functions = {"web_search": web_search, "get_weather": get_weather}
    return tools, tool_functions
