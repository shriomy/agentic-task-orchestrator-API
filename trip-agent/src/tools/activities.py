"""Ticketmaster Discovery API — events and activities at a destination.

The base URL now comes from `settings.ticketmaster_root_url` instead of being
hardcoded, so TICKETMASTER_ROOT_URL in .env is actually honoured.
"""

from datetime import datetime, timedelta, timezone
from typing import Any

from ..config import settings
from .http import ToolError, get_json


def _iso_utc(value: str | None) -> str | None:
    """Coerce a date-ish string into the ISO-8601 form Ticketmaster requires.

    The API rejects offsets with a colon and requires seconds, so `2026-09-01`
    and `2026-09-01T10:00` both need normalising to `2026-09-01T00:00:00Z`.
    """
    if not value:
        return None
    text = value.strip()
    if text.endswith("Z") and "T" in text and len(text) == 20:
        return text
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _normalise_event(event: dict[str, Any]) -> dict[str, Any]:
    dates = (event.get("dates") or {}).get("start") or {}
    venues = ((event.get("_embedded") or {}).get("venues") or [{}])
    venue = venues[0] if venues else {}
    price_ranges = (event.get("priceRanges") or [{}])[0]
    classifications = (event.get("classifications") or [{}])[0]

    return {
        "event_id": event.get("id"),
        "name": event.get("name"),
        "date": dates.get("dateTime") or dates.get("localDate"),
        "local_time": dates.get("localTime"),
        "venue": venue.get("name"),
        "city": ((venue.get("city") or {}).get("name")),
        "country": ((venue.get("country") or {}).get("countryCode")),
        "lat": ((venue.get("location") or {}).get("latitude")),
        "lon": ((venue.get("location") or {}).get("longitude")),
        "segment": ((classifications.get("segment") or {}).get("name")),
        "genre": ((classifications.get("genre") or {}).get("name")),
        "price_min": price_ranges.get("min"),
        "price_max": price_ranges.get("max"),
        "currency": price_ranges.get("currency"),
        "url": event.get("url"),
        "image": next((img.get("url") for img in (event.get("images") or []) if img.get("width", 0) > 600), None),
        "source": "ticketmaster",
    }


def search_events(
    city: str,
    keyword: str | None = None,
    start_date: str | None = None,
    end_date: str | None = None,
    country_code: str | None = None,
    classification: str | None = None,
    size: int = 12,
) -> dict:
    """Search events in a city, optionally within a date window.

    When no window is given, defaults to the next 30 days — "events in Paris"
    with no dates should not return last year's listings.
    """
    if not settings.ticketmaster_api_key:
        raise ToolError("Event search is not configured.")

    window_start = _iso_utc(start_date)
    window_end = _iso_utc(end_date)
    if not window_start:
        window_start = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    if not window_end:
        window_end = (datetime.now(timezone.utc) + timedelta(days=30)).strftime("%Y-%m-%dT%H:%M:%SZ")

    data = get_json(
        f"{settings.ticketmaster_root_url.rstrip('/')}/events.json",
        params={
            "apikey": settings.ticketmaster_api_key,
            "city": city,
            "keyword": keyword,
            "countryCode": country_code,
            "classificationName": classification,
            "startDateTime": window_start,
            "endDateTime": window_end,
            "size": max(1, min(size, 100)),
            "sort": "date,asc",
        },
        provider="the events provider",
    )

    events = (data.get("_embedded") or {}).get("events") or []
    page = data.get("page") or {}
    return {
        "source": "ticketmaster",
        "destination": city,
        "window": {"start": window_start, "end": window_end},
        "total_available": page.get("totalElements", len(events)),
        "events": [_normalise_event(event) for event in events],
    }


def search_attractions(keyword: str, size: int = 10) -> dict:
    """Search attractions (artists, teams, venues-as-entities) by keyword."""
    if not settings.ticketmaster_api_key:
        raise ToolError("Event search is not configured.")

    data = get_json(
        f"{settings.ticketmaster_root_url.rstrip('/')}/attractions.json",
        params={
            "apikey": settings.ticketmaster_api_key,
            "keyword": keyword,
            "size": max(1, min(size, 100)),
        },
        provider="the events provider",
    )
    attractions = (data.get("_embedded") or {}).get("attractions") or []
    return {
        "source": "ticketmaster",
        "keyword": keyword,
        "attractions": [
            {
                "attraction_id": item.get("id"),
                "name": item.get("name"),
                "type": ((item.get("classifications") or [{}])[0].get("segment") or {}).get("name"),
                "url": item.get("url"),
            }
            for item in attractions
        ],
    }
