from pydantic import BaseModel
from typing import Any


class GraphState(BaseModel):
    thread_id: str
    user_id: str
    user_memory_summary: dict[str, Any] | None = None
    last_agent_output: str | None = None
    should_interrupt: bool | None = None
    dialect_step: str | None = None
    preferences_to_persist: dict[str, Any] = {}
