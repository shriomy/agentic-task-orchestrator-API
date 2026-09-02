"""Graph assembly.

    START
      |
      v
  preprocess ---- out_of_scope ----> END       (guardrail 1 refused it)
      |      \
      |       `-- smalltalk --------> END
      v
  tool_search ---- (no tool needed) -> END      (direct answer, skips agent)
      |
      v
   summarize                                    (langmem, only when over budget)
      |
      v
    agent <-----------------.
      |                     |
      |  tool_calls? -------+  tools            (may interrupt for a pick)
      |                     |
      |  skipped a pause? --+  require_selection
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
    RequireSelectionNode,
    SmalltalkNode,
    SummarizationNode,
    ToolExecutorNode,
    ToolSearchNode,
)
from .state import GraphState


def route_after_preprocess(state: GraphState) -> Literal["out_of_scope", "smalltalk", "tool_search"]:
    """Guardrail 1's decision, applied."""
    if state.scope == "out_of_scope":
        return "out_of_scope"
    if state.scope == "smalltalk":
        return "smalltalk"
    return "tool_search"


def route_after_tool_search(state: GraphState) -> Literal["summarize", "end"]:
    """Whether ToolSearchNode answered directly (no tool needed) or found a
    query worth searching with. Decided after the node runs, since — unlike
    out_of_scope/smalltalk, which are known before their node runs — this
    depends on whether the model's single tool call fired."""
    return "end" if state.bot_response is not None else "summarize"


def route_after_agent(state: GraphState) -> Literal["tools", "require_selection", "finalize"]:
    """Continue the tool loop, enforce a skipped pause, or wrap up."""
    messages = state.messages or []
    if not messages:
        return "finalize"

    last = messages[-1]
    if isinstance(last, AIMessage) and getattr(last, "tool_calls", None):
        if int(state.tool_round_count or 0) >= settings.max_tool_rounds:
            # Stop looping. FinalizeNode closes off the dangling tool calls so
            # the next turn's message history is still valid.
            return "finalize"
        return "tools"

    # The agent wants to finish. If the user asked to choose and no pause has
    # happened this turn, send it back — but only after at least one lookup ran,
    # since there is nothing to offer before that.
    if (
        state.wants_selection
        and not state.has_selection_this_turn()
        and int(state.tool_round_count or 0) > 0
        and int(state.selection_nudges or 0) < RequireSelectionNode.MAX_NUDGES
    ):
        return "require_selection"

    return "finalize"


def build_graph(checkpointer: Any | None = None) -> Any:
    builder = StateGraph(state_schema=GraphState)

    builder.add_node("preprocess", PreprocessNode())
    builder.add_node("out_of_scope", OutOfScopeNode())
    builder.add_node("smalltalk", SmalltalkNode())
    builder.add_node("tool_search", ToolSearchNode())
    builder.add_node("summarize", SummarizationNode())
    builder.add_node("agent", AgentNode())
    builder.add_node("tools", ToolExecutorNode())
    builder.add_node("require_selection", RequireSelectionNode())
    builder.add_node("finalize", FinalizeNode())
    builder.add_node("persist_preferences", PersistPreferencesNode())

    builder.add_edge(START, "preprocess")
    builder.add_conditional_edges(
        "preprocess",
        route_after_preprocess,
        {"out_of_scope": "out_of_scope", "smalltalk": "smalltalk", "tool_search": "tool_search"},
    )
    builder.add_edge("out_of_scope", END)
    builder.add_edge("smalltalk", END)
    builder.add_conditional_edges(
        "tool_search",
        route_after_tool_search,
        {"summarize": "summarize", "end": END},
    )
    builder.add_edge("summarize", "agent")
    builder.add_conditional_edges(
        "agent",
        route_after_agent,
        {"tools": "tools", "require_selection": "require_selection", "finalize": "finalize"},
    )
    builder.add_edge("tools", "agent")
    builder.add_edge("require_selection", "agent")
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
