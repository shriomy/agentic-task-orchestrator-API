"""Tavily web search — destination discovery.

Tavily's search endpoint is `POST /search` with a JSON body; the previous
`GET /v1/search` in this file was not a real route. The key travels as a
bearer token, and `api_key` is also included in the body for compatibility
with older Tavily deployments.
"""

from typing import Any

from ..config import settings
from .http import ToolError, post_json


def search_destinations(
    query: str,
    *,
    max_results: int = 6,
    search_depth: str = "basic",
    include_answer: bool = True,
) -> dict:
    """Search the web for destinations, seasonal advice, packing tips, safety info.

    Returns a compact shape rather than Tavily's full payload — the raw response
    carries long page excerpts that would crowd out the rest of the context.
    """
    if not settings.tavily_api_key:
        raise ToolError("Web search is not configured.")

    payload: dict[str, Any] = {
        "api_key": settings.tavily_api_key,
        "query": query,
        "search_depth": search_depth,
        "max_results": max(1, min(max_results, 10)),
        "include_answer": include_answer,
        "topic": "general",
    }

    data = post_json(
        f"{settings.tavily_root_url}/search",
        json_body=payload,
        headers={
            "Authorization": f"Bearer {settings.tavily_api_key}",
            "Content-Type": "application/json",
        },
        provider="the web search provider",
    )

    results = []
    for item in (data.get("results") or [])[:max_results]:
        snippet = str(item.get("content") or "")
        results.append(
            {
                "title": item.get("title"),
                "url": item.get("url"),
                "snippet": snippet[:600],
                "score": item.get("score"),
            }
        )

    return {
        "source": "tavily",
        "query": query,
        "answer": data.get("answer"),
        "results": results,
    }
