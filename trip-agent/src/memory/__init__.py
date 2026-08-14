"""Memory scopes for the trip agent.

State   - thread-scoped graph state, checkpointed on thread_id  (checkpointer.py)
Context - derived per call from state; not stored separately     (graph/context.py)
Memory  - cross-thread durable facts, keyed on user_id           (user_memory.py)
Records - human-readable transcript in Supabase agent_* tables   (conversations.py)
"""

from .checkpointer import close_checkpointer, get_checkpointer
from .user_memory import (
    UserMemorySummary,
    delete_user_memory,
    get_user_memory_summary,
    write_user_memory,
)

__all__ = [
    "close_checkpointer",
    "get_checkpointer",
    "UserMemorySummary",
    "delete_user_memory",
    "get_user_memory_summary",
    "write_user_memory",
]
