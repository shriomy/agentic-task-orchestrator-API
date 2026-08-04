from __future__ import annotations

from typing import Any

import requests

from src.config import settings


def web_search(args: dict[str, Any]) -> dict[str, Any]:
    query = args.get("query")
    if not isinstance(query, str) or not query.strip():
        raise ValueError("query must be a non-empty string")
    if not settings.tavily_api_key:
        raise RuntimeError("TAVILY_API_KEY is not configured")

    response = requests.post(
        "https://api.tavily.com/search",
        json={"api_key": settings.tavily_api_key, "query": query, "max_results": 5},
        timeout=settings.request_timeout_seconds,
    )
    response.raise_for_status()
    data = response.json()
    results = data.get("results")
    if not isinstance(results, list):
        raise RuntimeError("Unexpected Tavily response shape")
    return {"query": query, "results": results}


web_search_definition = {
    "name": "web_search",
    "description": "Search the web for recent or external information using Tavily.",
    "parameters": {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "Search query"},
        },
        "required": ["query"],
        "additionalProperties": False,
    },
}
