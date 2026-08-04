from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


Message = dict[str, Any]


@dataclass
class AgentState:
    messages: list[Message] = field(default_factory=list)
    iteration: int = 0
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    last_tool_name: str | None = None
    last_tool_error: str | None = None

    def add_message(self, role: str, content: str | None = None, **extra: Any) -> None:
        message: Message = {"role": role}
        if content is not None:
            message["content"] = content
        message.update(extra)
        self.messages.append(message)
