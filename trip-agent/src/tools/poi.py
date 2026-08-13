import requests

from ..config import settings


def search_places(city: str | None = None, latitude: float | None = None, longitude: float | None = None, category: str | None = None) -> dict:
    if not settings.opentripmap_api_key:
        raise RuntimeError("OPENTRIPMAP_API_KEY is required for place search.")

    params = {"apikey": settings.opentripmap_api_key}
    if city:
        params["city"] = city
    if latitude is not None:
        params["lat"] = latitude
    if longitude is not None:
        params["lon"] = longitude
    if category:
        params["kinds"] = category

    response = requests.get(
        "https://api.opentripmap.com/0.1/en/places/geoname",
        params=params,
        timeout=settings.request_timeout_seconds,
    )
    response.raise_for_status()
    return response.json()
