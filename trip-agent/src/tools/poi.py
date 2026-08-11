from ..tools.amadeus_client import amadeus_client


def search_points_of_interest(city: str, latitude: float, longitude: float, category: str | None = None) -> dict:
    params = {
        "cityCode": city,
        "latitude": latitude,
        "longitude": longitude,
    }
    if category:
        params["keyword"] = category
    return amadeus_client.get("/v1/reference-data/locations/pois", params=params)
