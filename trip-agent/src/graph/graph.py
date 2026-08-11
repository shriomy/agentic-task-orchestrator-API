from langgraph import StateGraph, Node, Command
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


graph = StateGraph(
    state_type=GraphState,
    nodes={
        "start_router": StartRouter(),
        "chat": ChatNode(),
        "memory_injection": MemoryInjectionNode(),
        "hil_preference": HILPreferenceNode(),
        "agent": AgentNode(),
        "tool": ToolNode(),
        "reflect": ReflectNode(),
        "persist_preferences": PersistPreferencesNode(),
    },
    initial_node="start_router",
)

# Graph topology:
# START -> start_router
# start_router.route == "chat" -> chat
# start_router.route == "task" -> memory_injection -> hil_preference -> agent
# agent -> tool (if tool requested) -> agent
# agent -> reflect (when no tool requested)
# reflect.approved == True -> persist_preferences -> END
# reflect.approved == False -> agent
