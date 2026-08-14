"""Graph assembly.

    START
      |
      v
  preprocess ---- out_of_scope ----> END       (guardrail 1 refused it)
      |      \
      |       `-- smalltalk --------> END
      v
   summarize                                    (langmem, only when over budget)
      |
      v
    agent <-------------.
      |                 |
      |  tool_calls?    |
      +---- tools ------'                       (may interrupt for a pick)
      |
      v
   finalize                                     (guardrail 2, output half)
      |
      v
 persist_preferences ---------------> END       (cross-thread memory)

The routers are plain functions passed to `add_conditional_edges`. The previous
version registered them with `add_node`, which cannot work: a node must return a
state update, and these return a destination name.
"""

from __future__ import annotations

from typing import Any, Literal

from langchain_core.messages import AIMessage
from langgraph.graph import END, START, StateGraph

from ..config import settings
from ..memory.checkpointer import get_checkpointer
from .nodes import (
    AgentNode,
    FinalizeNode,
    OutOfScopeNode,
    PersistPreferencesNode,
    PreprocessNode,
    SmalltalkNode,
    SummarizationNode,
    ToolExecutorNode,
)
from .state import GraphState


def route_after_preprocess(state: GraphState) -> Literal["out_of_scope", "smalltalk", "summarize"]:
    """Guardrail 1's decision, applied."""
    if state.scope == "out_of_scope":
        return "out_of_scope"
    if state.scope == "smalltalk":
        return "smalltalk"
    return "summarize"


def route_after_agent(state: GraphState) -> Literal["tools", "finalize"]:
    """Continue the tool loop while the model is still calling tools."""
    messages = state.messages or []
    if not messages:
        return "finalize"

    last = messages[-1]
    if not isinstance(last, AIMessage) or not getattr(last, "tool_calls", None):
        return "finalize"

    if int(state.tool_round_count or 0) >= settings.max_tool_rounds:
        # Stop looping. FinalizeNode closes off the dangling tool calls so the
        # next turn's message history is still valid.
        return "finalize"

    return "tools"


def build_graph(checkpointer: Any | None = None) -> Any:
    builder = StateGraph(state_schema=GraphState)

    builder.add_node("preprocess", PreprocessNode())
    builder.add_node("out_of_scope", OutOfScopeNode())
    builder.add_node("smalltalk", SmalltalkNode())
    builder.add_node("summarize", SummarizationNode())
    builder.add_node("agent", AgentNode())
    builder.add_node("tools", ToolExecutorNode())
    builder.add_node("finalize", FinalizeNode())
    builder.add_node("persist_preferences", PersistPreferencesNode())

    builder.add_edge(START, "preprocess")
    builder.add_conditional_edges(
        "preprocess",
        route_after_preprocess,
        {"out_of_scope": "out_of_scope", "smalltalk": "smalltalk", "summarize": "summarize"},
    )
    builder.add_edge("out_of_scope", END)
    builder.add_edge("smalltalk", END)
    builder.add_edge("summarize", "agent")
    builder.add_conditional_edges(
        "agent",
        route_after_agent,
        {"tools": "tools", "finalize": "finalize"},
    )
    builder.add_edge("tools", "agent")
    builder.add_edge("finalize", "persist_preferences")
    builder.add_edge("persist_preferences", END)

    return builder.compile(
        checkpointer=checkpointer if checkpointer is not None else get_checkpointer(),
        name="trip_organiser_agent",
    )


_graph: Any = None


def get_graph() -> Any:
    """Compiled graph, built once per process."""
    global _graph
    if _graph is None:
        _graph = build_graph()
    return _graph
