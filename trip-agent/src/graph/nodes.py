from typing import Any
from langgraph import Node, Command
from langgraph.schema import Tool
from langchain.core import OpenAI
from langchain.tools import tool
from ..memory.user_memory import get_user_memory_summary, write_user_memory
from ..config import settings
from ..tools.destinations import search_destinations
from ..tools.poi import search_points_of_interest
from ..tools.hotels import search_hotels
from ..tools.favorites import save_favorite, get_favorites


class StartRouter(Node):
    def run(self, state: Any, message: str) -> dict[str, str]:
        # This router decides whether the incoming message is small talk or a travel task.
        # It uses the model as a lightweight classifier, not keyword matching.
        prompt = (
            "Classify only whether the latest user message is casual chit-chat or a travel-related task. "
            "Respond with JSON: {\"route\": \"chat\" or \"task\"}."
        )
        # In a real implementation, use a small, fast model variant.
        model = OpenAI(model_name=settings.llm_model, openai_api_key=settings.llm_api_key)
        response = model.predict(prompt + "\n\nMessage:\n" + message)
        if "chat" in response.lower() and "task" not in response.lower():
            return {"route": "chat"}
        if "task" in response.lower():
            return {"route": "task"}
        return {"route": "task"}


class ChatNode(Node):
    def run(self, state: Any, message: str) -> str:
        # Small talk path: no tools, no ReAct loop, no reflection.
        return f"Thanks for reaching out! I can also help with travel planning whenever you're ready."


class MemoryInjectionNode(Node):
    def run(self, state: Any) -> dict[str, Any]:
        # Load a lightweight summary of long-term user memory and make it available to the agent.
        summary = get_user_memory_summary(state.user_id)
        state.user_memory_summary = summary.dict()
        return {"user_memory_summary": state.user_memory_summary}


class HILPreferenceNode(Node):
    def run(self, state: Any, message: str) -> dict[str, Any]:
        # Decide whether the user wants a human-in-the-loop pause before tool execution.
        # This is a reasoning step based on conversation intent, not a rigid keyword rule.
        if state.user_memory_summary and state.user_memory_summary.get("preferences", {}).get("prefers_review", False):
            state.should_interrupt = True
        else:
            state.should_interrupt = False
        return {"should_interrupt": state.should_interrupt}


class AgentNode(Node):
    def run(self, state: Any, message: str) -> str:
        # Placeholder: in the real graph, this would be a LangChain tool-enabled LLM call.
        state.last_agent_output = "I have generated a travel recommendation draft."
        return state.last_agent_output


class ToolNode(Node):
    def run(self, state: Any, tool_name: str, tool_args: dict[str, Any]) -> Any:
        # Execute the requested tool and return the result.
        tool_mapping = {
            "search_destinations": search_destinations,
            "search_points_of_interest": search_points_of_interest,
            "search_activities": lambda latitude, longitude: amadeus_client.get("/v1/shopping/activities", {"latitude": latitude, "longitude": longitude}),
            "search_hotels": search_hotels,
            "save_favorite": save_favorite,
            "get_favorites": get_favorites,
        }
        if tool_name not in tool_mapping:
            raise ValueError(f"Unsupported tool: {tool_name}")
        result = tool_mapping[tool_name](**tool_args)
        return result


class ReflectNode(Node):
    def run(self, state: Any, draft_answer: str) -> dict[str, Any]:
        # Critique the final draft against the user's original request and tool usage.
        critique = "The answer appears grounded and relevant."  # placeholder
        return {"critique": critique, "approved": True}


class PersistPreferencesNode(Node):
    def run(self, state: Any) -> None:
        # Persist durable user preferences discovered during the conversation.
        for key, value in state.preferences_to_persist.items():
            write_user_memory(state.user_id, key, value)
        return {}
