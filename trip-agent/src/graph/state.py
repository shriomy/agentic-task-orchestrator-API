from pydantic import BaseModel
from typing import Any
from pydantic import BaseModel, Field


class GraphState(BaseModel):
    thread_id: str
    user_id: str
    route: str | None = None
    message: str | None = None
    messages: list[Any] = Field(default_factory=list)
    user_memory_summary: dict[str, Any] | None = None
    hil_preference: str | None = None
    hil_review_reply: Any | None = None
    bot_response: str | None = None
    last_agent_output: str | None = None
    critique: str | None = None
    approved: bool | None = None
    reflection_count: int = 0
    preferences_to_persist: dict[str, Any] = Field(default_factory=dict)
