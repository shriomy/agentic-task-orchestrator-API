import requests
from ..config import settings


def search_destinations(query: str) -> dict:
    """Discover travel destinations via Tavily web search.

    This tool is intentionally separate from Amadeus; it provides broad inspiration
    and destination discovery from the web.
    """
    response = requests.get(
        "https://api.tavily.com/v1/search",
        params={"q": query},
        headers={"Authorization": f"Bearer {settings.tavily_api_key}"},
        timeout=settings.request_timeout_seconds,
    )
    response.raise_for_status()
    return response.json()
