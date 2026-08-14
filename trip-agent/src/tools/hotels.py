"""Booking.com via RapidAPI — accommodation search.

This API cannot search by plain city name (tool_query.md §2): `searchHotels`
requires a `dest_id` + `search_type` that only `searchDestination` can supply,
so the wrapper chains the two calls. Auth is header-based, not a query param.
"""

from datetime import date, timedelta
from typing import Any

from ..config import settings
from .http import ToolError, get_json

# Booking's own labels for the property types a traveller actually distinguishes.
STAY_TYPE_HINTS = {
    "hotel": "hotel",
    "hostel": "hostel",
    "cabin": "chalet",
    "apartment": "apartment",
    "guesthouse": "guest house",
    "villa": "villa",
    "resort": "resort",
}


def _headers() -> dict[str, str]:
    key = settings.booking_api_key
    if not key:
        raise ToolError("Accommodation search is not configured.")
    return {"x-rapidapi-key": key, "x-rapidapi-host": settings.rapidapi_booking_host}


def _default_window() -> tuple[str, str]:
    """A 2-night stay two weeks out, used when the user names no dates."""
    arrival = date.today() + timedelta(days=14)
    return arrival.isoformat(), (arrival + timedelta(days=2)).isoformat()


def resolve_destination(location: str) -> dict[str, Any]:
    """Turn a place name into the dest_id / search_type pair searchHotels needs."""
    data = get_json(
        f"{settings.booking_root_url}/hotels/searchDestination",
        params={"query": location},
        headers=_headers(),
        provider="the accommodation provider",
    )
    matches = data.get("data") or []
    if not matches:
        raise ToolError(f"No bookable destination matched '{location}'.")

    # Prefer a city-level match; fall back to the provider's own top result.
    city = next((m for m in matches if str(m.get("search_type", "")).lower() == "city"), None)
    chosen = city or matches[0]
    return {
        "dest_id": str(chosen.get("dest_id")),
        "search_type": chosen.get("search_type") or "CITY",
        "label": chosen.get("label") or chosen.get("name") or location,
        "country": chosen.get("country"),
        "lat": chosen.get("latitude"),
        "lon": chosen.get("longitude"),
    }


def _normalise_hotel(entry: dict[str, Any], check_in: str, check_out: str) -> dict[str, Any]:
    """Read the fields out of `.property`, where this API actually nests them."""
    prop = entry.get("property") or entry
    price = ((prop.get("priceBreakdown") or {}).get("grossPrice") or {})
    photos = prop.get("photoUrls") or []
    hotel_id = prop.get("id") or entry.get("hotel_id")

    return {
        "hotel_id": str(hotel_id) if hotel_id is not None else None,
        "name": prop.get("name"),
        "stay_type": prop.get("accommodationTypeName") or prop.get("propertyClass"),
        "price": price.get("value"),
        "currency": price.get("currency"),
        "review_score": prop.get("reviewScore"),
        "review_count": prop.get("reviewCount"),
        "star_rating": prop.get("propertyClass"),
        "lat": prop.get("latitude"),
        "lon": prop.get("longitude"),
        "photo": photos[0] if photos else None,
        "check_in": check_in,
        "check_out": check_out,
        "source": "booking.com",
    }


def search_accommodations(
    location: str,
    check_in: str | None = None,
    check_out: str | None = None,
    adults: int = 2,
    children_ages: str | None = None,
    rooms: int = 1,
    max_price: float | None = None,
    min_price: float | None = None,
    stay_type: str | None = None,
    currency: str = "USD",
    limit: int = 10,
) -> dict:
    """Search night stays (hotels/hostels/cabins) at a destination.

    `stay_type` filters the normalised results client-side, because the API's
    `categories_filter` codes must come from its own /getFilter lookup table
    and cannot be guessed from a free-text word like "hostel".
    """
    if not check_in or not check_out:
        check_in, check_out = _default_window()

    destination = resolve_destination(location)

    data = get_json(
        f"{settings.booking_root_url}/hotels/searchHotels",
        params={
            "dest_id": destination["dest_id"],
            "search_type": destination["search_type"],
            "arrival_date": check_in,
            "departure_date": check_out,
            "adults": adults,
            "children_age": children_ages,
            "room_qty": rooms,
            "page_number": 1,
            "price_min": min_price,
            "price_max": max_price,
            "currency_code": currency,
            "units": "metric",
            "languagecode": "en-us",
        },
        headers=_headers(),
        provider="the accommodation provider",
    )

    # `data.hotels` holds the results; `appear[]` is Booking's own UI/tracking
    # noise and is deliberately discarded.
    payload = data.get("data") or {}
    hotels = [_normalise_hotel(entry, check_in, check_out) for entry in (payload.get("hotels") or [])]

    if stay_type:
        needle = STAY_TYPE_HINTS.get(stay_type.lower().rstrip("s"), stay_type).lower()
        filtered = [h for h in hotels if needle in str(h.get("stay_type") or "").lower()]
        # Only apply the filter when it leaves something to show.
        hotels = filtered or hotels

    return {
        "source": "booking.com",
        "destination": destination["label"],
        "check_in": check_in,
        "check_out": check_out,
        "guests": {"adults": adults, "rooms": rooms, "children_ages": children_ages},
        "total_available": (payload.get("meta") or [{}])[0].get("title") if isinstance(payload.get("meta"), list) else None,
        "accommodations": hotels[:limit],
    }


def get_accommodation_details(
    hotel_id: str,
    check_in: str | None = None,
    check_out: str | None = None,
    adults: int = 2,
    rooms: int = 1,
    currency: str = "USD",
) -> dict:
    """Full detail for one property from a previous search."""
    if not check_in or not check_out:
        check_in, check_out = _default_window()

    data = get_json(
        f"{settings.booking_root_url}/hotels/getHotelDetails",
        params={
            "hotel_id": hotel_id,
            "arrival_date": check_in,
            "departure_date": check_out,
            "adults": adults,
            "room_qty": rooms,
            "currency_code": currency,
        },
        headers=_headers(),
        provider="the accommodation provider",
    )
    payload = data.get("data") or {}
    return {
        "hotel_id": str(payload.get("hotel_id") or hotel_id),
        "name": payload.get("hotel_name"),
        "address": payload.get("address"),
        "city": payload.get("city"),
        "review_score": payload.get("review_nr") and payload.get("reviewScore"),
        "url": payload.get("url"),
        "facilities": [f.get("name") for f in (payload.get("facilities_block") or {}).get("facilities", [])][:12],
        "check_in": check_in,
        "check_out": check_out,
        "source": "booking.com",
    }
