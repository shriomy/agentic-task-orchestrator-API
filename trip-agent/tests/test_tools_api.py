"""The four travel API wrappers: correct endpoints, correct parameter names,
correct multi-step chains. Every one of these was wrong before.
"""

from __future__ import annotations

from typing import Any

import pytest

from src.config import settings
from src.tools.activities import _iso_utc, search_events
from src.tools.destinations import search_destinations
from src.tools.hotels import search_accommodations
from src.tools.http import ToolError
from src.tools.poi import search_places


class Recorder:
    """Captures outbound calls and replays canned JSON per URL fragment."""

    def __init__(self, routes: dict[str, Any]) -> None:
        self.routes = routes
        self.calls: list[dict[str, Any]] = []

    def __call__(self, url: str, **kwargs: Any) -> Any:
        self.calls.append({"url": url, **kwargs})
        for fragment, payload in self.routes.items():
            if fragment in url:
                return _Response(payload)
        raise AssertionError(f"unexpected request to {url}")

    def call_to(self, fragment: str) -> dict[str, Any]:
        matches = [call for call in self.calls if fragment in call["url"]]
        assert matches, f"expected a call to {fragment}, got {[c['url'] for c in self.calls]}"
        return matches[0]


class _Response:
    def __init__(self, payload: Any, status: int = 200) -> None:
        self._payload = payload
        self.status_code = status

    def raise_for_status(self) -> None:
        return None

    def json(self) -> Any:
        return self._payload


# --------------------------------------------------------------------------- #
# Tavily
# --------------------------------------------------------------------------- #


def test_web_search_posts_to_the_real_tavily_endpoint(monkeypatch):
    """It was a GET to /v1/search, which is not a route Tavily serves."""
    monkeypatch.setattr(settings, "tavily_api_key", "tv-key")
    recorder = Recorder({"api.tavily.com": {"answer": "Go north.", "results": [{"title": "Tromso", "url": "https://x.example/1", "content": "cold"}]}})
    monkeypatch.setattr("src.tools.http.requests.post", recorder)

    result = search_destinations("top destinations in Norway", max_results=3)

    call = recorder.call_to("api.tavily.com")
    assert call["url"] == "https://api.tavily.com/search"
    assert call["json"]["query"] == "top destinations in Norway"
    assert call["json"]["max_results"] == 3
    assert call["headers"]["Authorization"] == "Bearer tv-key"
    assert call["timeout"] == settings.request_timeout_seconds
    assert result["answer"] == "Go north."
    assert result["results"][0]["title"] == "Tromso"


def test_web_search_without_a_key_fails_cleanly(monkeypatch):
    monkeypatch.setattr(settings, "tavily_api_key", None)
    with pytest.raises(ToolError, match="not configured"):
        search_destinations("anywhere")


# --------------------------------------------------------------------------- #
# OpenTripMap — geoname then radius
# --------------------------------------------------------------------------- #


def test_place_search_geocodes_then_calls_radius(monkeypatch):
    """The old code sent radius-only params to geoname and got no POI list."""
    monkeypatch.setattr(settings, "opentripmap_api_key", "otm-key")
    recorder = Recorder(
        {
            "places/geoname": {"name": "Moscow", "country": "RU", "lat": 55.75, "lon": 37.61},
            "places/radius": [
                {"xid": "W1", "name": "Red Square", "kinds": "historic", "rate": 3, "point": {"lat": 55.75, "lon": 37.62}},
                {"xid": "W2", "name": "", "kinds": "other", "rate": 1, "point": {"lat": 0, "lon": 0}},
            ],
        }
    )
    monkeypatch.setattr("src.tools.http.requests.get", recorder)

    result = search_places(city="Moscow", category="historic", limit=10)

    geoname = recorder.call_to("places/geoname")
    assert geoname["params"] == {"name": "Moscow", "apikey": "otm-key"}
    assert "kinds" not in geoname["params"], "geoname does not accept kinds"
    assert "lat" not in geoname["params"], "geoname does not accept lat"

    radius = recorder.call_to("places/radius")
    assert radius["params"]["lat"] == 55.75
    assert radius["params"]["lon"] == 37.61
    assert radius["params"]["kinds"] == "historic"
    assert radius["params"]["format"] == "json"

    assert result["destination"] == "Moscow"
    assert [place["name"] for place in result["places"]] == ["Red Square"]
    assert result["places"][0]["place_id"] == "W1"


def test_unnamed_osm_nodes_are_dropped(monkeypatch):
    monkeypatch.setattr(settings, "opentripmap_api_key", "otm-key")
    monkeypatch.setattr(
        "src.tools.http.requests.get",
        Recorder(
            {
                "places/geoname": {"name": "Kyoto", "lat": 35.0, "lon": 135.7},
                "places/radius": [{"xid": "W9", "kinds": "other", "point": {"lat": 1, "lon": 1}}],
            }
        ),
    )
    assert search_places(city="Kyoto")["places"] == []


def test_an_unknown_city_is_reported_not_silently_empty(monkeypatch):
    monkeypatch.setattr(settings, "opentripmap_api_key", "otm-key")
    monkeypatch.setattr("src.tools.http.requests.get", Recorder({"places/geoname": {}}))
    with pytest.raises(ToolError, match="Could not find a location"):
        search_places(city="Narnia")


# --------------------------------------------------------------------------- #
# Ticketmaster
# --------------------------------------------------------------------------- #


def test_event_search_uses_the_configured_root_url_and_normalises_dates(monkeypatch):
    monkeypatch.setattr(settings, "ticketmaster_api_key", "tm-key")
    monkeypatch.setattr(settings, "ticketmaster_root_url", "https://app.ticketmaster.com/discovery/v2/")
    recorder = Recorder(
        {
            "events.json": {
                "_embedded": {
                    "events": [
                        {
                            "id": "G5v",
                            "name": "Jazz Night",
                            "dates": {"start": {"dateTime": "2026-09-02T19:00:00Z"}},
                            "_embedded": {"venues": [{"name": "Le Duc", "city": {"name": "Paris"}}]},
                            "priceRanges": [{"min": 30, "max": 80, "currency": "EUR"}],
                            "url": "https://ticketmaster.com/event/G5v",
                            "images": [{"url": "https://img.example/a.jpg", "width": 1024}],
                        }
                    ]
                },
                "page": {"totalElements": 1},
            }
        }
    )
    monkeypatch.setattr("src.tools.http.requests.get", recorder)

    result = search_events("Paris", start_date="2026-09-01", end_date="2026-09-05")

    call = recorder.call_to("events.json")
    assert call["url"] == "https://app.ticketmaster.com/discovery/v2/events.json"
    assert call["params"]["apikey"] == "tm-key"
    assert call["params"]["city"] == "Paris"
    # ISO-8601 with seconds and a Z suffix, which is what the API requires.
    assert call["params"]["startDateTime"] == "2026-09-01T00:00:00Z"
    assert call["params"]["endDateTime"] == "2026-09-05T00:00:00Z"

    event = result["events"][0]
    assert event["name"] == "Jazz Night"
    assert event["venue"] == "Le Duc"
    assert event["price_min"] == 30
    assert event["currency"] == "EUR"
    assert event["url"] == "https://ticketmaster.com/event/G5v"


def test_event_search_defaults_to_a_forward_looking_window(monkeypatch):
    """'events in Paris' with no dates must not return last year's listings."""
    monkeypatch.setattr(settings, "ticketmaster_api_key", "tm-key")
    recorder = Recorder({"events.json": {"_embedded": {"events": []}, "page": {}}})
    monkeypatch.setattr("src.tools.http.requests.get", recorder)

    result = search_events("Paris")

    params = recorder.call_to("events.json")["params"]
    assert params["startDateTime"] < params["endDateTime"]
    assert result["window"]["start"].endswith("Z")


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("2026-09-01", "2026-09-01T00:00:00Z"),
        ("2026-09-01T10:30", "2026-09-01T10:30:00Z"),
        ("2026-09-01T10:30:00Z", "2026-09-01T10:30:00Z"),
        ("not a date", None),
        (None, None),
    ],
)
def test_date_normalisation(raw, expected):
    assert _iso_utc(raw) == expected


# --------------------------------------------------------------------------- #
# Booking.com — searchDestination then searchHotels
# --------------------------------------------------------------------------- #


def test_accommodation_search_resolves_dest_id_before_searching(monkeypatch):
    """searchHotels needs dest_id + search_type; a city name alone cannot work."""
    monkeypatch.setattr(settings, "rapidapi_key", "rapid-key")
    recorder = Recorder(
        {
            "searchDestination": {
                "data": [
                    {"dest_id": "900001", "search_type": "REGION", "label": "Rome region"},
                    {"dest_id": "-126693", "search_type": "CITY", "label": "Rome, Italy"},
                ]
            },
            "searchHotels": {
                "data": {
                    "hotels": [
                        {
                            "property": {
                                "id": 191605,
                                "name": "Hotel Artemide",
                                "reviewScore": 8.9,
                                "reviewCount": 4210,
                                "accommodationTypeName": "Hotel",
                                "priceBreakdown": {"grossPrice": {"value": 245.5, "currency": "USD"}},
                                "latitude": 41.9,
                                "longitude": 12.5,
                                "photoUrls": ["https://img.example/h.jpg"],
                            }
                        }
                    ],
                    "meta": [{"title": "212 properties"}],
                },
                # Booking's UI/tracking noise, which must be ignored.
                "appear": [{"banner": "sign in for discounts"}],
            },
        }
    )
    monkeypatch.setattr("src.tools.http.requests.get", recorder)

    result = search_accommodations("Rome", check_in="2026-09-01", check_out="2026-09-05", adults=2)

    lookup = recorder.call_to("searchDestination")
    assert lookup["params"] == {"query": "Rome"}
    assert lookup["headers"]["x-rapidapi-key"] == "rapid-key"
    assert lookup["headers"]["x-rapidapi-host"] == "booking-com15.p.rapidapi.com"

    search = recorder.call_to("searchHotels")
    params = search["params"]
    # The city-level match is preferred over the region.
    assert params["dest_id"] == "-126693"
    assert params["search_type"] == "CITY"
    # Real parameter names, not the invented checkin_date/location/max_price.
    assert params["arrival_date"] == "2026-09-01"
    assert params["departure_date"] == "2026-09-05"
    assert "location" not in params
    assert "checkin_date" not in params

    hotel = result["accommodations"][0]
    assert hotel["hotel_id"] == "191605"
    assert hotel["name"] == "Hotel Artemide"
    assert hotel["price"] == 245.5
    assert hotel["currency"] == "USD"
    assert hotel["review_score"] == 8.9
    assert "appear" not in result


def test_max_price_maps_to_price_max(monkeypatch):
    monkeypatch.setattr(settings, "rapidapi_key", "rapid-key")
    recorder = Recorder(
        {
            "searchDestination": {"data": [{"dest_id": "1", "search_type": "CITY", "label": "Rome"}]},
            "searchHotels": {"data": {"hotels": []}},
        }
    )
    monkeypatch.setattr("src.tools.http.requests.get", recorder)

    search_accommodations("Rome", check_in="2026-09-01", check_out="2026-09-03", max_price=150)

    assert recorder.call_to("searchHotels")["params"]["price_max"] == 150


def test_a_stay_type_filters_the_results(monkeypatch):
    monkeypatch.setattr(settings, "rapidapi_key", "rapid-key")
    recorder = Recorder(
        {
            "searchDestination": {"data": [{"dest_id": "1", "search_type": "CITY", "label": "Krabi"}]},
            "searchHotels": {
                "data": {
                    "hotels": [
                        {"property": {"id": 1, "name": "Grand Hotel", "accommodationTypeName": "Hotel"}},
                        {"property": {"id": 2, "name": "Backpackers", "accommodationTypeName": "Hostel"}},
                    ]
                }
            },
        }
    )
    monkeypatch.setattr("src.tools.http.requests.get", recorder)

    result = search_accommodations("Krabi", check_in="2026-09-01", check_out="2026-09-03", stay_type="hostels")

    assert [stay["name"] for stay in result["accommodations"]] == ["Backpackers"]


def test_no_bookable_destination_is_reported(monkeypatch):
    monkeypatch.setattr(settings, "rapidapi_key", "rapid-key")
    monkeypatch.setattr("src.tools.http.requests.get", Recorder({"searchDestination": {"data": []}}))
    with pytest.raises(ToolError, match="No bookable destination"):
        search_accommodations("Narnia", check_in="2026-09-01", check_out="2026-09-03")


def test_missing_dates_fall_back_to_a_sensible_window(monkeypatch):
    monkeypatch.setattr(settings, "rapidapi_key", "rapid-key")
    recorder = Recorder(
        {
            "searchDestination": {"data": [{"dest_id": "1", "search_type": "CITY", "label": "Rome"}]},
            "searchHotels": {"data": {"hotels": []}},
        }
    )
    monkeypatch.setattr("src.tools.http.requests.get", recorder)

    result = search_accommodations("Rome")

    assert result["check_in"] < result["check_out"]
    assert recorder.call_to("searchHotels")["params"]["arrival_date"] == result["check_in"]
