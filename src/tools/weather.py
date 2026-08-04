from __future__ import annotations

from typing import Any

import requests

from src.config import settings


def get_weather(args: dict[str, Any]) -> dict[str, Any]:
    latitude = args.get("latitude")
    longitude = args.get("longitude")
    if not isinstance(latitude, (int, float)) or not isinstance(longitude, (int, float)):
        raise ValueError("latitude and longitude must be numbers")

    response = requests.get(
        "https://api.open-meteo.com/v1/forecast",
        params={
            "latitude": latitude,
            "longitude": longitude,
            "current_weather": "true",
            "forecast_days": 1,
        },
        timeout=settings.request_timeout_seconds,
    )
    response.raise_for_status()
    data = response.json()
    current_weather = data.get("current_weather")
    if not isinstance(current_weather, dict):
        raise RuntimeError("Unexpected Open-Meteo response shape")
    return {
        "latitude": latitude,
        "longitude": longitude,
        "current_weather": current_weather,
        "timezone": data.get("timezone"),
    }


get_weather_definition = {
    "name": "get_weather",
    "description": "Get the current weather from Open-Meteo for a latitude and longitude.",
    "parameters": {
        "type": "object",
        "properties": {
            "latitude": {"type": "number"},
            "longitude": {"type": "number"},
        },
        "required": ["latitude", "longitude"],
        "additionalProperties": False,
    },
}
