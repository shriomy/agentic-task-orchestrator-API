from typing import Any

from langgraph.graph import StateGraph
from .state import GraphState
from .nodes import (
    StartRouter,
    ChatNode,
    MemoryInjectionNode,
    HILPreferenceNode,
    AgentNode,
    AgentRoutingNode,
    HILInterruptNode,
    TOOL_NODE,
    ReflectNode,
    ReflectRoutingNode,
    PersistPreferencesNode,
)
from .memory.checkpointer import conversation_checkpointer


def _route_to_node(state: Any) -> str:
    route = getattr(state, "route", None)
    if route == "chat":
        return "chat"
    return "memory_injection"


builder = StateGraph(state_schema=GraphState)

builder.add_node("start_router", StartRouter())
builder.add_node("chat", ChatNode())
builder.add_node("memory_injection", MemoryInjectionNode())
builder.add_node("hil_preference", HILPreferenceNode())
builder.add_node("agent", AgentNode())
builder.add_node("agent_router", AgentRoutingNode())
builder.add_node("hil_interrupt", HILInterruptNode())
builder.add_node("tool", TOOL_NODE)
builder.add_node("reflect", ReflectNode())
builder.add_node("reflect_router", ReflectRoutingNode())
builder.add_node("persist_preferences", PersistPreferencesNode())

builder.set_entry_point("start_router")
builder.add_conditional_edges(
    "start_router",
    _route_to_node,
    path_map={"chat": "chat", "task": "memory_injection"},
)

builder.add_edge("memory_injection", "hil_preference")
builder.add_edge("hil_preference", "agent")
builder.add_edge("agent", "agent_router")

def _agent_routing(state: Any) -> str:
    messages = getattr(state, "messages", []) or []
    if not messages:
        return "reflect"
    last_message = messages[-1]
    tool_calls = getattr(last_message, "tool_calls", []) or []
    if tool_calls:
        if getattr(state, "hil_preference", None) == "review_before_tools" and getattr(
            state, "hil_review_reply", None
        ) is None:
            return "hil_interrupt"
        return "tool"
    return "reflect"

builder.add_conditional_edges(
    "agent_router",
    _agent_routing,
    path_map={"hil_interrupt": "hil_interrupt", "tool": "tool", "reflect": "reflect"},
)

builder.add_edge("hil_interrupt", "tool")
builder.add_edge("tool", "agent")
builder.add_edge("reflect", "reflect_router")

builder.add_conditional_edges(
    "reflect_router",
    lambda state: "agent" if getattr(state, "approved", False) is False and getattr(state, "reflection_count", 0) < 3 else "persist_preferences",
    path_map={"agent": "agent", "persist_preferences": "persist_preferences"},
)

builder.set_finish_point("persist_preferences")

graph = builder.compile(
    checkpointer=conversation_checkpointer,
    name="travel_discovery_agent",
)
