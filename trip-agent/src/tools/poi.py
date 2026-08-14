"""OpenTripMap — points of interest at a destination.

Implements the geocode-then-search chain described in tool_query.md §5: the
`geoname` endpoint only turns a placename into coordinates, so a second call to
`radius` is required to actually list places. The old single malformed call to
`geoname` with `kinds`/`lat`/`lon` could never return a POI list.
"""

from typing import Any

from ..config import settings
from .http import ToolError, get_json

# A sensible default spread of sightseeing categories. Callers can override.
DEFAULT_KINDS = "interesting_places,museums,historic,architecture,cultural,natural"


def _base(endpoint: str) -> str:
    return f"{settings.opentripmap_root_url}/{settings.opentripmap_lang}/places/{endpoint}"


def geocode_place(name: str, country: str | None = None) -> dict:
    """Resolve a placename to coordinates. `geoname` accepts only name + country."""
    if not settings.opentripmap_api_key:
        raise ToolError("Place search is not configured.")

    data = get_json(
        _base("geoname"),
        params={"name": name, "country": country, "apikey": settings.opentripmap_api_key},
        provider="the places provider",
    )
    if not isinstance(data, dict) or data.get("lat") is None or data.get("lon") is None:
        raise ToolError(f"Could not find a location called '{name}'.")
    return {
        "name": data.get("name") or name,
        "country": data.get("country"),
        "lat": data["lat"],
        "lon": data["lon"],
        "population": data.get("population"),
        "timezone": data.get("timezone"),
    }


def _normalise(feature: dict[str, Any]) -> dict[str, Any]:
    """Flatten a `radius` result. format=json gives flat dicts, geojson nests them."""
    if "properties" in feature:
        properties = feature.get("properties") or {}
        geometry = (feature.get("geometry") or {}).get("coordinates") or [None, None]
        lon, lat = geometry[0], geometry[1]
    else:
        properties = feature
        lon, lat = feature.get("point", {}).get("lon"), feature.get("point", {}).get("lat")
        if lon is None:
            lon, lat = feature.get("lon"), feature.get("lat")

    xid = properties.get("xid")
    return {
        "place_id": xid,
        "name": properties.get("name") or "Unnamed place",
        "kinds": properties.get("kinds"),
        "rate": properties.get("rate"),
        "distance_m": properties.get("dist"),
        "lat": lat,
        "lon": lon,
        "url": f"https://opentripmap.com/en/card/{xid}" if xid else None,
        "source": "opentripmap",
    }


def search_places(
    city: str | None = None,
    latitude: float | None = None,
    longitude: float | None = None,
    category: str | None = None,
    radius_m: int = 12000,
    limit: int = 15,
    min_rate: str = "2",
) -> dict:
    """List notable places near a city (or explicit coordinates).

    `min_rate` filters out the long tail of unnamed/unrated OSM nodes that make
    raw OpenTripMap output unusable for a travel recommendation.
    """
    if not settings.opentripmap_api_key:
        raise ToolError("Place search is not configured.")

    resolved_name = city
    if latitude is None or longitude is None:
        if not city:
            raise ToolError("A city name or a latitude/longitude pair is required.")
        located = geocode_place(city)
        latitude, longitude = located["lat"], located["lon"]
        resolved_name = located["name"]

    raw = get_json(
        _base("radius"),
        params={
            "apikey": settings.opentripmap_api_key,
            "lat": latitude,
            "lon": longitude,
            "radius": radius_m,
            "kinds": category or DEFAULT_KINDS,
            "rate": min_rate,
            "format": "json",
            "limit": max(1, min(limit, 50)),
        },
        provider="the places provider",
    )

    features = raw.get("features", []) if isinstance(raw, dict) else (raw or [])
    places = [_normalise(item) for item in features if isinstance(item, dict)]
    # Named places first — OSM nodes with no name are useless to a traveller.
    places = [place for place in places if place["name"] != "Unnamed place"]

    return {
        "source": "opentripmap",
        "destination": resolved_name,
        "center": {"lat": latitude, "lon": longitude},
        "radius_m": radius_m,
        "places": places[:limit],
    }


def get_place_details(place_id: str) -> dict:
    """Full detail for one place, using its `xid` from a previous search."""
    if not settings.opentripmap_api_key:
        raise ToolError("Place search is not configured.")

    data = get_json(
        _base(f"xid/{place_id}"),
        params={"apikey": settings.opentripmap_api_key},
        provider="the places provider",
    )
    info = data.get("info") or {}
    return {
        "place_id": data.get("xid"),
        "name": data.get("name"),
        "kinds": data.get("kinds"),
        "description": (info.get("descr") or data.get("wikipedia_extracts", {}).get("text") or "")[:1200],
        "image": data.get("preview", {}).get("source") or data.get("image"),
        "wikipedia": data.get("wikipedia"),
        "address": data.get("address"),
        "lat": (data.get("point") or {}).get("lat"),
        "lon": (data.get("point") or {}).get("lon"),
        "source": "opentripmap",
    }
