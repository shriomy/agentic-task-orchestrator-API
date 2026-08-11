from ..tools.amadeus_client import amadeus_client


def search_hotels(city_code: str, check_in: str, check_out: str, guests: int, max_price: float | None = None) -> dict:
    params = {
        "cityCode": city_code,
        "checkInDate": check_in,
        "checkOutDate": check_out,
        "adults": guests,
    }
    if max_price is not None:
        params["maxRate"] = max_price
    return amadeus_client.get("/v2/shopping/hotel-offers", params=params)
