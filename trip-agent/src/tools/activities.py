import requests

from ..config import settings


def search_events(city: str, keyword: str | None = None, start_date: str | None = None, end_date: str | None = None) -> dict:
    if not settings.ticketmaster_api_key:
        raise RuntimeError("TICKETMASTER_API_KEY is required for event search.")

    params = {
        "apikey": settings.ticketmaster_api_key,
        "city": city,
    }
    if keyword:
        params["keyword"] = keyword
    if start_date:
        params["startDateTime"] = start_date
    if end_date:
        params["endDateTime"] = end_date

    response = requests.get(
        "https://app.ticketmaster.com/discovery/v2/events.json",
        params=params,
        timeout=settings.request_timeout_seconds,
    )
    response.raise_for_status()
    return response.json()
