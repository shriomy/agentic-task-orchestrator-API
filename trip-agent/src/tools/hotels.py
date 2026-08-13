import requests

from ..config import settings


def search_accommodations(location: str, check_in: str, check_out: str, adults: int = 2, max_price: float | None = None) -> dict:
    if not settings.hotel_search_api_key:
        raise RuntimeError("HOTEL_SEARCH_API_KEY is required for accommodation search.")

    headers = {
        "x-rapidapi-key": settings.hotel_search_api_key,
        "x-rapidapi-host": "booking-com15.p.rapidapi.com",
    }
    params = {
        "location": location,
        "checkin_date": check_in,
        "checkout_date": check_out,
        "adults": adults,
    }
    if max_price is not None:
        params["max_price"] = max_price

    response = requests.get(
        "https://booking-com15.p.rapidapi.com/api/v1/hotels/searchHotels",
        params=params,
        headers=headers,
        timeout=settings.request_timeout_seconds,
    )
    response.raise_for_status()
    return response.json()
