from ..tools.amadeus_client import amadeus_client


def search_activities(latitude: float, longitude: float) -> dict:
    return amadeus_client.get(
        "/v1/shopping/activities",
        {"latitude": latitude, "longitude": longitude},
    )
