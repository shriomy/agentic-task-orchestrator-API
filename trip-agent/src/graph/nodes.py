from typing import Any

from langchain.chat_models import init_chat_model

from ..memory.user_memory import get_user_memory_summary, write_user_memory
from ..config import settings
from ..tools.destinations import search_destinations
from ..tools.poi import search_points_of_interest
from ..tools.hotels import search_hotels
from ..tools.favorites import save_favorite, get_favorites
from ..tools.amadeus_client import amadeus_client


def _trim_message(state: Any) -> str:
    return getattr(state, "message", None) or (state.get("message") if isinstance(state, dict) else "")


class StartRouter:
    def __call__(self, state: Any) -> dict[str, str]:
        # This router decides whether the incoming message is small talk or a travel task.
        prompt = (
            "Classify only whether the latest user message is casual chit-chat or a travel-related task. "
            "Respond with JSON: {\"route\": \"chat\" or \"task\"}."
        )
        model = init_chat_model(
            model=settings.llm_model,
            model_provider=settings.llm_provider,
            openai_api_key=settings.llm_api_key,
        )
        message = _trim_message(state)
        response = model.predict(prompt + "\n\nMessage:\n" + message)
        if "chat" in response.lower() and "task" not in response.lower():
            return {"route": "chat"}
        if "task" in response.lower():
            return {"route": "task"}
        return {"route": "task"}


class ChatNode:
    def __call__(self, state: Any) -> dict[str, str]:
        return {"bot_response": "Thanks for reaching out! I can also help with travel planning whenever you're ready."}


class MemoryInjectionNode:
    def __call__(self, state: Any) -> dict[str, Any]:
        summary = get_user_memory_summary(state.user_id)
        state.user_memory_summary = summary.dict()
        return {"user_memory_summary": state.user_memory_summary}


class HILPreferenceNode:
    def __call__(self, state: Any) -> dict[str, Any]:
        if state.user_memory_summary and state.user_memory_summary.get("preferences", {}).get("prefers_review", False):
            state.should_interrupt = True
        else:
            state.should_interrupt = False
        return {"should_interrupt": state.should_interrupt}


class AgentNode:
    def __call__(self, state: Any) -> dict[str, Any]:
        state.last_agent_output = "I have generated a travel recommendation draft."
        return {"last_agent_output": state.last_agent_output}


class ToolNode:
    def __call__(self, state: Any) -> dict[str, Any]:
        tool_mapping = {
            "search_destinations": search_destinations,
            "search_points_of_interest": search_points_of_interest,
            "search_activities": lambda latitude, longitude: amadeus_client.get(
                "/v1/shopping/activities", {"latitude": latitude, "longitude": longitude}
            ),
            "search_hotels": search_hotels,
            "save_favorite": save_favorite,
            "get_favorites": get_favorites,
        }
        tool_name = getattr(state, "tool_name", None)
        tool_args = getattr(state, "tool_args", {}) or {}
        if not tool_name:
            return {}
        if tool_name not in tool_mapping:
            raise ValueError(f"Unsupported tool: {tool_name}")
        return {"tool_result": tool_mapping[tool_name](**tool_args)}


class ReflectNode:
    def __call__(self, state: Any) -> dict[str, Any]:
        critique = "The answer appears grounded and relevant."
        return {"critique": critique, "approved": True}


class PersistPreferencesNode:
    def __call__(self, state: Any) -> dict[str, Any]:
        for key, value in state.preferences_to_persist.items():
            write_user_memory(state.user_id, key, value)
        return {}
