from __future__ import annotations

from dataclasses import dataclass
import json
from typing import Any, Protocol

import requests


class LLMClient(Protocol):
    def chat(self, *, messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None = None) -> dict[str, Any]:
        ...


@dataclass
class ChatCompletionsClient:
    api_key: str
    api_base: str
    model: str
    timeout_seconds: float = 20.0

    def chat(self, *, messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None = None) -> dict[str, Any]:
        url = f"{self.api_base.rstrip('/')}/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload: dict[str, Any] = {"model": self.model, "messages": messages}
        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = "auto"

        response = requests.post(url, headers=headers, json=payload, timeout=self.timeout_seconds)
        response.raise_for_status()
        data = response.json()
        return data["choices"][0]["message"]


def parse_tool_arguments(arguments: Any) -> dict[str, Any]:
    if isinstance(arguments, dict):
        return arguments
    if isinstance(arguments, str):
        try:
            parsed = json.loads(arguments)
        except json.JSONDecodeError as exc:
            raise ValueError("Tool arguments were not valid JSON") from exc
        if not isinstance(parsed, dict):
            raise ValueError("Tool arguments must decode to an object")
        return parsed
    raise ValueError("Tool arguments must be a JSON string or object")
