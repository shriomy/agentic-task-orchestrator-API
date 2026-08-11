from typing import Any

from langgraph.graph import StateGraph
from .state import GraphState
from .nodes import (
    StartRouter,
    ChatNode,
    MemoryInjectionNode,
    HILPreferenceNode,
    AgentNode,
    ToolNode,
    ReflectNode,
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
builder.add_node("tool", ToolNode())
builder.add_node("reflect", ReflectNode())
builder.add_node("persist_preferences", PersistPreferencesNode())

builder.set_entry_point("start_router")
builder.add_conditional_edges(
    "start_router",
    _route_to_node,
    path_map={"chat": "chat", "task": "memory_injection"},
)

builder.add_edge("memory_injection", "hil_preference")
builder.add_edge("hil_preference", "agent")
builder.add_edge("agent", "reflect")
builder.add_edge("reflect", "persist_preferences")
builder.set_finish_point("persist_preferences")

graph = builder.compile(
    checkpointer=conversation_checkpointer,
    name="travel_discovery_agent",
)
