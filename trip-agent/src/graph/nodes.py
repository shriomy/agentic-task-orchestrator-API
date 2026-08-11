import json
from typing import Any

from langchain.chat_models import init_chat_model
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.tools import tool
from langgraph.prebuilt.tool_node import ToolNode, tools_condition
from langgraph.types import Command, interrupt

from ..memory.user_memory import get_user_memory_summary, write_user_memory
from ..config import settings
from ..tools.destinations import search_destinations
from ..tools.poi import search_points_of_interest
from ..tools.hotels import search_hotels
from ..tools.favorites import save_favorite, get_favorites
from ..tools.activities import search_activities


def _trim_message(state: Any) -> str:
    return getattr(state, "message", None) or ""


@tool
def search_destinations_tool(query: str) -> dict:
    return search_destinations(query)


@tool
def search_points_of_interest_tool(city: str, latitude: float, longitude: float, category: str | None = None) -> dict:
    return search_points_of_interest(city, latitude, longitude, category)


@tool
def search_activities_tool(latitude: float, longitude: float) -> dict:
    return search_activities(latitude, longitude)


@tool
def search_hotels_tool(city_code: str, check_in: str, check_out: str, guests: int, max_price: float | None = None) -> dict:
    return search_hotels(city_code, check_in, check_out, guests, max_price)


@tool
def save_favorite_tool(user_id: str, city: str, events: list[dict], hotel: dict, notes: str | None = None) -> dict:
    return save_favorite(user_id, city, events, hotel, notes)


@tool
def get_favorites_tool(user_id: str) -> dict:
    return get_favorites(user_id)


def _tool_call_wrapper(request: Any, execute: Any) -> Any:
    result = execute(request)
    if isinstance(result, ToolMessage):
        prior_messages = []
        if isinstance(request.state, dict):
            prior_messages = request.state.get("messages", []) or []
        elif hasattr(request.state, "messages"):
            prior_messages = getattr(request.state, "messages", []) or []
        return Command(update={"messages": [*prior_messages, result]})
    return result


TOOL_NODE = ToolNode(
    [
        search_destinations_tool,
        search_points_of_interest_tool,
        search_activities_tool,
        search_hotels_tool,
        save_favorite_tool,
        get_favorites_tool,
    ],
    handle_tool_errors=True,
    messages_key="messages",
    wrap_tool_call=_tool_call_wrapper,
)


class StartRouter:
    def __call__(self, state: Any) -> dict[str, str]:
        prompt = (
            "Decide whether the latest user message is casual chit-chat or a travel planning task. "
            "Respond only with JSON: {\"route\": \"chat\" or \"task\"}."
        )
        model = init_chat_model(
            model=settings.llm_model,
            model_provider=settings.llm_provider,
            openai_api_key=settings.llm_api_key,
        )
        message = _trim_message(state)
        response = model.predict(prompt + "\n\nMessage:\n" + message)
        try:
            payload = json.loads(response)
            return {"route": payload.get("route", "task")}
        except json.JSONDecodeError:
            return {"route": "task" if "task" in response.lower() else "chat"}


class ChatNode:
    def __call__(self, state: Any) -> dict[str, str]:
        return {"bot_response": "Thanks for reaching out! I can also help with travel planning whenever you're ready."}


class MemoryInjectionNode:
    def __call__(self, state: Any) -> dict[str, Any]:
        if state.message:
            state.messages.append(HumanMessage(content=state.message))
        summary = get_user_memory_summary(state.user_id)
        state.user_memory_summary = summary.dict()
        return {"user_memory_summary": state.user_memory_summary, "messages": state.messages}


class HILPreferenceNode:
    def __call__(self, state: Any) -> dict[str, Any]:
        model = init_chat_model(
            model=settings.llm_model,
            model_provider=settings.llm_provider,
            openai_api_key=settings.llm_api_key,
        )
        prompt = (
            "Determine whether the user wants to review planned tool actions before they run. "
            "Respond only with JSON: {\"wants_confirmation\": true or false, \"at_step\": \"before_tools\" or null}."
        )
        response = model.predict(prompt + "\n\nMessage:\n" + _trim_message(state))
        hil_preference = None
        try:
            payload = json.loads(response)
            if payload.get("wants_confirmation"):
                hil_preference = "review_before_tools"
        except json.JSONDecodeError:
            if "review" in response.lower() or "confirm" in response.lower():
                hil_preference = "review_before_tools"
        state.hil_preference = hil_preference
        return {"hil_preference": state.hil_preference}


class AgentNode:
    def __init__(self) -> None:
        self.model = init_chat_model(
            model=settings.llm_model,
            model_provider=settings.llm_provider,
            openai_api_key=settings.llm_api_key,
        ).bind_tools(
            [
                search_destinations_tool,
                search_points_of_interest_tool,
                search_activities_tool,
                search_hotels_tool,
                save_favorite_tool,
                get_favorites_tool,
            ]
        )

    def __call__(self, state: Any) -> dict[str, Any]:
        if not state.messages:
            state.messages.append(HumanMessage(content=_trim_message(state)))

        response = self.model.invoke({"messages": state.messages})
        if isinstance(response, AIMessage):
            state.messages.append(response)
            state.last_agent_output = response.content
        else:
            state.last_agent_output = str(response)
            if hasattr(response, "content"):
                state.messages.append(response)
        return {"messages": state.messages, "last_agent_output": state.last_agent_output}


class AgentRoutingNode:
    def __call__(self, state: Any) -> str:
        if (isinstance(state, dict) and (messages := state.get("messages", []))) or (
            messages := getattr(state, "messages", [])
        ):
            ai_message = messages[-1]
        else:
            return "reflect"

        tool_calls = getattr(ai_message, "tool_calls", None)
        if tool_calls:
            if state.hil_preference == "review_before_tools" and state.hil_review_reply is None:
                return "hil_interrupt"
            return "tool"
        return "reflect"


class HILInterruptNode:
    def __call__(self, state: Any) -> dict[str, Any]:
        if state.hil_preference != "review_before_tools":
            return {}
        if state.hil_review_reply is not None:
            return {"hil_review_reply": state.hil_review_reply}

        last_message = state.messages[-1] if state.messages else None
        tool_calls = getattr(last_message, "tool_calls", None)
        if not tool_calls:
            return {}

        prompt = {
            "reason": "The user asked to review tool calls before execution.",
            "pending_tools": [tc.name for tc in tool_calls],
        }
        reply = interrupt(prompt)
        state.hil_review_reply = reply
        return {"hil_review_reply": reply}


class ReflectRoutingNode:
    def __call__(self, state: Any) -> str:
        if getattr(state, "approved", False) is False and getattr(state, "reflection_count", 0) < 3:
            return "agent"
        return "persist_preferences"


class ReflectNode:
    MAX_REFLECTIONS = 3

    def __call__(self, state: Any) -> dict[str, Any]:
        state.reflection_count += 1
        if state.reflection_count > self.MAX_REFLECTIONS:
            return {"critique": "Reflection cap reached.", "approved": True}

        model = init_chat_model(
            model=settings.llm_model,
            model_provider=settings.llm_provider,
            openai_api_key=settings.llm_api_key,
        )
        prompt = (
            "You are a second-pass reviewer. Evaluate the latest draft answer and any tool results. "
            "If it is complete and accurate, respond with JSON: {\"approved\": true}. "
            "If it needs more work, respond with JSON: {\"approved\": false, \"critique\": \"...\"}."
        )
        draft = state.last_agent_output or ""
        response = model.predict(prompt + "\n\nDraft:\n" + draft)
        try:
            payload = json.loads(response)
            approved = bool(payload.get("approved"))
            critique = payload.get("critique", "No critique provided.")
        except json.JSONDecodeError:
            approved = False
            critique = response
        if state.reflection_count >= self.MAX_REFLECTIONS:
            approved = True
            critique = critique or "Reflection cap forced approval."
        return {"critique": critique, "approved": approved}


class PersistPreferencesNode:
    def __call__(self, state: Any) -> dict[str, Any]:
        for key, value in state.preferences_to_persist.items():
            write_user_memory(state.user_id, key, value)
        return {}
