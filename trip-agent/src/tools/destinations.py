import requests

from ..config import settings


def search_destinations(query: str) -> dict:
    """Discover travel destinations via Tavily web search."""
    response = requests.get(
        "https://api.tavily.com/v1/search",
        params={"query": query},
        headers={"Authorization": f"Bearer {settings.tavily_api_key}"},
        timeout=settings.request_timeout_seconds,
    )
    response.raise_for_status()
    return response.json()
