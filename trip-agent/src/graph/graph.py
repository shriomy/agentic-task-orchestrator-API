from typing import Any

from langgraph.graph import END, StateGraph

from .memory.checkpointer import conversation_checkpointer
from .nodes import (
    AgentNode,
    AgentRoutingNode,
    ChatNode,
    MemoryInjectionNode,
    OffTopicNode,
    PersistPreferencesNode,
    ReflectNode,
    ReflectRoutingNode,
    StartRouter,
    SummarizationNode,
    TOOL_NODE,
)
from .state import GraphState


def _route_to_node(state: Any) -> str:
    return getattr(state, "route", "task")


builder = StateGraph(state_schema=GraphState)

builder.add_node("start_router", StartRouter())
builder.add_node("chat", ChatNode())
builder.add_node("off_topic", OffTopicNode())
builder.add_node("memory_injection", MemoryInjectionNode())
builder.add_node("summarize_messages", SummarizationNode())
builder.add_node("agent", AgentNode())
builder.add_node("agent_router", AgentRoutingNode())
builder.add_node("tool", TOOL_NODE)
builder.add_node("reflect", ReflectNode())
builder.add_node("reflect_router", ReflectRoutingNode())
builder.add_node("persist_preferences", PersistPreferencesNode())

builder.set_entry_point("start_router")
builder.add_conditional_edges(
    "start_router",
    _route_to_node,
    path_map={"chat": "chat", "task": "memory_injection", "off_topic": "off_topic"},
)
builder.add_edge("chat", END)
builder.add_edge("off_topic", END)
builder.add_edge("memory_injection", "summarize_messages")
builder.add_edge("summarize_messages", "agent")
builder.add_edge("agent", "agent_router")


def _agent_routing(state: Any) -> str:
    messages = getattr(state, "messages", []) or []
    if not messages:
        return "reflect"
    last_message = messages[-1]
    tool_calls = getattr(last_message, "tool_calls", []) or []
    if tool_calls:
        return "tool"
    return "reflect"


builder.add_conditional_edges(
    "agent_router",
    _agent_routing,
    path_map={"tool": "tool", "reflect": "reflect"},
)
builder.add_edge("tool", "summarize_messages")
builder.add_edge("summarize_messages", "agent")
builder.add_edge("reflect", "reflect_router")

builder.add_conditional_edges(
    "reflect_router",
    lambda state: "agent" if getattr(state, "approved", False) is False and getattr(state, "reflection_count", 0) < 3 else "persist_preferences",
    path_map={"agent": "agent", "persist_preferences": "persist_preferences"},
)
builder.add_edge("persist_preferences", END)

graph = builder.compile(
    checkpointer=conversation_checkpointer,
    name="travel_discovery_agent",
)
