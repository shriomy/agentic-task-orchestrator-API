from src.tools.weather import get_weather


def test_get_weather_argument_validation() -> None:
    try:
        get_weather({"latitude": "bad", "longitude": 79})
    except ValueError as exc:
        assert "latitude and longitude" in str(exc)
    else:
        raise AssertionError("expected ValueError")
