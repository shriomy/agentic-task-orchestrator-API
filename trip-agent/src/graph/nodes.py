import json
import re
from typing import Annotated, Any, Literal

from langchain.chat_models import init_chat_model
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool
from langgraph.prebuilt import InjectedState
from langgraph.prebuilt.tool_node import ToolNode
from langgraph.types import Command, interrupt
from pydantic import BaseModel

from ..config import settings
from ..memory.user_memory import get_user_memory_summary, write_user_memory
from ..tools.activities import search_events
from ..tools.destinations import search_destinations
from ..tools.favorites import delete_favorite, get_favorites, save_favorite, update_favorite
from ..tools.hotels import search_accommodations
from ..tools.poi import search_places
from .state import GraphState, Selection

try:
    from langmem.short_term import SummarizationNode as LangMemSummarizationNode
except ImportError:  # pragma: no cover - optional dependency at runtime
    class LangMemSummarizationNode:  # type: ignore[no-redef]
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            self.args = args
            self.kwargs = kwargs

        def __call__(self, state: Any) -> dict[str, Any]:
            return {}


class RouteDecision(BaseModel):
    route: Literal["chat", "task", "off_topic"] = "task"


def _trim_message(state: Any) -> str:
    if isinstance(state, dict):
        return str(state.get("message") or "")
    return str(getattr(state, "message", None) or "")


def _coerce_state(state: Any) -> dict[str, Any]:
    if isinstance(state, dict):
        return state
    if hasattr(state, "model_dump"):
        return state.model_dump()
    return state.__dict__


def _current_user_id(state: Any) -> str:
    if isinstance(state, dict):
        return str(state.get("user_id") or "")
    return str(getattr(state, "user_id", "") or "")


def _selection_context(state: Any) -> str:
    selections = getattr(state, "selections", None) or []
    if not selections:
        return "No selections yet."
    lines = []
    for selection in selections:
        if selection.picked:
            lines.append(f"turn {selection.turn_index}: picked {selection.picked}; skipped {', '.join(selection.skipped) if selection.skipped else 'none'}")
        else:
            lines.append(f"turn {selection.turn_index}: no pick yet; options {', '.join(selection.options)}")
    return "Selections so far: " + " | ".join(lines)


def _build_model(model_name: str | None = None, *, fast: bool = False):
    provider = settings.llm_provider or "openai"
    api_key = settings.llm_api_key
    base_url = settings.llm_api_base
    if provider == "openrouter":
        api_key = settings.openrouter_api_key or settings.llm_api_key
        base_url = settings.openrouter_api_base or "https://openrouter.ai/api/v1"
        provider = "openai"
    if fast:
        model_name = model_name or "gpt-4o-mini"
    else:
        model_name = model_name or settings.llm_model
    kwargs = {}
    if api_key:
        kwargs["api_key"] = api_key
    if base_url:
        kwargs["base_url"] = base_url
    return init_chat_model(model=model_name, model_provider=provider, **kwargs)


def _tool_call_wrapper(request: Any, execute: Any) -> Any:
    result = execute(request)
    if isinstance(result, Command):
        return result
    if isinstance(result, ToolMessage):
        prior_messages = []
        if isinstance(request.state, dict):
            prior_messages = request.state.get("messages", []) or []
        elif hasattr(request.state, "messages"):
            prior_messages = getattr(request.state, "messages", []) or []
        tool_round_count = 0
        if isinstance(request.state, dict):
            tool_round_count = int(request.state.get("tool_round_count", 0))
        else:
            tool_round_count = int(getattr(request.state, "tool_round_count", 0))
        return Command(update={"messages": [*prior_messages, result], "tool_round_count": tool_round_count + 1})
    return result


def scrub_internals(text: str) -> str:
    if not text:
        return text
    scrubbed = text
    scrubbed = re.sub(r"(?i)sk-[A-Za-z0-9]{10,}", "[REDACTED_API_KEY]", scrubbed)
    scrubbed = re.sub(r"(?i)(api[_-]?key|token|secret|base_url|endpoint)\s*[:=]\s*[^\s\"'`]+", r"\1=[REDACTED]", scrubbed)
    scrubbed = re.sub(r"(?i)Authorization\s*:\s*Bearer\s+[A-Za-z0-9._-]+", "Authorization: [REDACTED]", scrubbed)
    scrubbed = re.sub(r"https?://[^\s\"'`]+", "[REDACTED_ENDPOINT]", scrubbed)
    scrubbed = re.sub(r"(?i)\b(search_destinations_tool|search_places_tool|search_events_tool|search_accommodations_tool|save_favorite_tool|get_favorites_tool|update_favorite_tool|delete_favorite_tool)\b", "[tool]", scrubbed)
    return scrubbed


def _draft_has_internal_leaks(draft: str) -> bool:
    leaks = [
        r"(?i)\b(search_destinations_tool|search_places_tool|search_events_tool|search_accommodations_tool|save_favorite_tool|get_favorites_tool|update_favorite_tool|delete_favorite_tool)\b",
        r"(?i)sk-[A-Za-z0-9]{10,}",
        r"(?i)(api[_-]?key|token|secret|base_url|endpoint)\s*[:=]\s*[^\s\"'`]+",
        r"https?://[^\s\"'`]+",
    ]
    return any(re.search(pattern, draft) for pattern in leaks)


@tool
def search_destinations_tool(query: str) -> dict:
    return search_destinations(query)


@tool
def search_places_tool(city: str, latitude: float | None = None, longitude: float | None = None, category: str | None = None) -> dict:
    return search_places(city=city, latitude=latitude, longitude=longitude, category=category)


@tool
def search_events_tool(city: str, start_date: str | None = None, end_date: str | None = None, keyword: str | None = None) -> dict:
    return search_events(city=city, keyword=keyword, start_date=start_date, end_date=end_date)


@tool
def search_accommodations_tool(location: str, check_in: str, check_out: str, adults: int = 2, max_price: float | None = None) -> dict:
    return search_accommodations(location=location, check_in=check_in, check_out=check_out, adults=adults, max_price=max_price)


@tool
def save_favorite_tool(city: str, events: list[dict], hotel: dict | None = None, notes: str | None = None, state: Annotated[Any, InjectedState] = None) -> dict:
    user_id = _current_user_id(state)
    return save_favorite(user_id, city, events, hotel or {}, notes)


@tool
def get_favorites_tool(state: Annotated[Any, InjectedState] = None) -> dict:
    user_id = _current_user_id(state)
    return get_favorites(user_id)


@tool
def update_favorite_tool(favorite_id: str, updates: dict[str, Any], state: Annotated[Any, InjectedState] = None) -> Command:
    return ask_user_to_select_tool(
        state=state,
        options=[f"Confirm update for favorite {favorite_id}", "Skip update"],
        prompt="A destructive favorite update is pending. Confirm or skip it.",
        action="update_favorite",
        target_id=favorite_id,
        payload={"favorite_id": favorite_id, "updates": updates},
    )


@tool
def delete_favorite_tool(favorite_id: str, state: Annotated[Any, InjectedState] = None) -> Command:
    return ask_user_to_select_tool(
        state=state,
        options=[f"Confirm delete for favorite {favorite_id}", "Skip delete"],
        prompt="A destructive favorite delete is pending. Confirm or skip it.",
        action="delete_favorite",
        target_id=favorite_id,
        payload={"favorite_id": favorite_id},
    )


@tool
def ask_user_to_select_tool(
    options: list[str],
    state: Annotated[Any, InjectedState] = None,
    prompt: str = "Please choose one option.",
    action: str = "selection",
    target_id: str | None = None,
    payload: dict[str, Any] | None = None,
) -> Command:
    state_data = _coerce_state(state)
    prior = state_data.get("selections", []) if isinstance(state_data, dict) else getattr(state, "selections", []) or []
    turn_index = int(state_data.get("turn_index", 0) if isinstance(state_data, dict) else getattr(state, "turn_index", 0) or 0)
    next_turn_index = turn_index + 1
    selection = Selection(options=options, picked=None, skipped=[], turn_index=next_turn_index)
    reply = interrupt({"prompt": prompt, "options": options, "action": action, "target_id": target_id, "payload": payload or {}, "turn_index": next_turn_index})
    if isinstance(reply, dict):
        chosen = reply.get("picked") or reply.get("selection")
    else:
        chosen = reply
    if chosen in options:
        selection.picked = chosen
        selection.skipped = [item for item in options if item != chosen]
    else:
        selection.picked = None
        selection.skipped = list(options)
    updated = [*prior, selection]
    return Command(update={"selections": updated, "turn_index": next_turn_index})


TOOL_NODE = ToolNode(
    [
        search_destinations_tool,
        search_places_tool,
        search_events_tool,
        search_accommodations_tool,
        save_favorite_tool,
        get_favorites_tool,
        update_favorite_tool,
        delete_favorite_tool,
        ask_user_to_select_tool,
    ],
    handle_tool_errors=True,
    messages_key="messages",
    wrap_tool_call=_tool_call_wrapper,
)


class StartRouter:
    def __call__(self, state: Any) -> dict[str, str]:
        model = _build_model("gpt-4o-mini", fast=True).with_structured_output(RouteDecision)
        prompt = (
            "Classify the latest message as travel-related, travel-favorites related, or unrelated. "
            "Use 'task' for travel planning or saved favorites; 'chat' for brief social conversation; 'off_topic' for anything else."
        )
        result = model.invoke([HumanMessage(content=f"{prompt}\n\nMessage:\n{_trim_message(state)}")])
        return {"route": result.route}


class ChatNode:
    def __call__(self, state: Any) -> dict[str, str]:
        return {"bot_response": "Thanks for reaching out! I can also help with travel planning or your saved favorites whenever you are ready."}


class OffTopicNode:
    def __call__(self, state: Any) -> dict[str, str]:
        return {"bot_response": "I can help with travel planning, destinations, events, accommodations, and your saved favorites. Please ask about those topics."}


class MemoryInjectionNode:
    def __call__(self, state: Any) -> dict[str, Any]:
        if getattr(state, "message", None):
            current_messages = list(getattr(state, "messages", []) or [])
            current_messages.append(HumanMessage(content=state.message))
            state.messages = current_messages
        summary = get_user_memory_summary(_current_user_id(state))
        state.user_memory_summary = summary.model_dump()
        return {"user_memory_summary": state.user_memory_summary, "messages": state.messages}


class SummarizationNode(LangMemSummarizationNode):
    def __init__(self, token_threshold: int = 12000, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.token_threshold = token_threshold

    def __call__(self, state: Any) -> dict[str, Any]:
        messages = list(getattr(state, "messages", []) or [])
        if not messages:
            return {}
        selections = [msg for msg in messages if isinstance(msg, SystemMessage) and "Selections so far" in str(msg.content)]
        non_selection_messages = [msg for msg in messages if not (isinstance(msg, SystemMessage) and "Selections so far" in str(msg.content))]
        if len(non_selection_messages) < 6:
            return {}
        approximate_tokens = sum(len(str(getattr(msg, "content", ""))) // 4 for msg in messages)
        if approximate_tokens < self.token_threshold:
            return {}
        summary_text = "Conversation summary: " + " ".join(str(getattr(msg, "content", "")) for msg in non_selection_messages[-6:])
        new_messages = [*selections, SystemMessage(content=summary_text)]
        return {"messages": new_messages}


class AgentNode:
    def __init__(self) -> None:
        self.model = _build_model().bind_tools(
            [
                search_destinations_tool,
                search_places_tool,
                search_events_tool,
                search_accommodations_tool,
                save_favorite_tool,
                get_favorites_tool,
                update_favorite_tool,
                delete_favorite_tool,
                ask_user_to_select_tool,
            ]
        )

    def __call__(self, state: Any) -> dict[str, Any]:
        messages = list(getattr(state, "messages", []) or [])
        if not messages:
            messages.append(HumanMessage(content=_trim_message(state)))
        selection_summary = _selection_context(state)
        messages = [SystemMessage(content=selection_summary), *messages]
        response = self.model.invoke({"messages": messages})
        if isinstance(response, AIMessage):
            state.messages = messages + [response]
            state.last_agent_output = response.content
        else:
            state.last_agent_output = str(response)
            if hasattr(response, "content"):
                state.messages = messages + [response]
        return {"messages": state.messages, "last_agent_output": state.last_agent_output}


class AgentRoutingNode:
    def __call__(self, state: Any) -> str:
        messages = getattr(state, "messages", []) or []
        if not messages:
            return "reflect"
        ai_message = messages[-1]
        tool_calls = getattr(ai_message, "tool_calls", None)
        if tool_calls:
            return "tool"
        return "reflect"


class ReflectRoutingNode:
    def __call__(self, state: Any) -> str:
        if getattr(state, "approved", False) is False and getattr(state, "reflection_count", 0) < 3:
            return "agent"
        return "persist_preferences"


class ReflectNode:
    MAX_REFLECTIONS = 3

    def __call__(self, state: Any) -> dict[str, Any]:
        state.reflection_count = int(getattr(state, "reflection_count", 0) or 0) + 1
        draft = str(getattr(state, "last_agent_output", "") or "")
        if _draft_has_internal_leaks(draft):
            state.approved = False
            state.critique = "Draft response leaked tool names, raw queries, endpoints, or credentials. Rewrite it without internal details."
            state.last_agent_output = scrub_internals(draft)
            return {"critique": state.critique, "approved": False, "last_agent_output": state.last_agent_output}

        model = _build_model().with_structured_output(RouteDecision)
        prompt = (
            "Review the draft output for factual completeness and for any leaked tool names, raw queries, API keys, or endpoints. "
            "If it is safe and complete, return {\"route\": \"task\"}. If it leaks internals or is incomplete, return {\"route\": \"off_topic\"}."
        )
        response = model.invoke([HumanMessage(content=f"{prompt}\n\nDraft:\n{draft}")])
        approved = response.route == "task"
        critique = "Draft passed the rubric." if approved else "Draft leaked internal tool or API details; rewrite required."

        if state.reflection_count >= self.MAX_REFLECTIONS:
            approved = True
            critique = "Reflection cap reached; final response scrubbed."
        cleaned = scrub_internals(draft)
        return {"critique": critique, "approved": approved, "last_agent_output": cleaned, "bot_response": cleaned}


class PersistPreferencesNode:
    def __call__(self, state: Any) -> dict[str, Any]:
        for key, value in (getattr(state, "preferences_to_persist", {}) or {}).items():
            write_user_memory(_current_user_id(state), key, value)
        return {"bot_response": scrub_internals(getattr(state, "last_agent_output", "") or getattr(state, "bot_response", "") or "")}
