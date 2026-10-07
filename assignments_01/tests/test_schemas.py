import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from weatherkit import WeatherResponse


def test_valid_weather_response():
    # I use __file__ so this path still works even if pytest runs
    # from a different folder.
    file_path = Path(__file__).parent.parent / "weather_raw.json"

    with open(file_path) as file:
        data = json.load(file)

    response = WeatherResponse.model_validate(data)

    assert response.latitude == 35.216976
    assert response.timezone == "GMT"
    assert len(response.hourly.time) == 168


def test_invalid_latitude():
    data = {
        "latitude": 200.0,
        "longitude": -80.83189,
        "timezone": "GMT",
        "elevation": 254.0,
        "hourly": {
            "time": ["2026-04-08T00:00"],
            "temperature_2m": [16.8],
            "precipitation": [0.0],
        },
    }

    with pytest.raises(ValidationError):
        WeatherResponse.model_validate(data)


def test_mismatched_hourly_lengths():
    data = {
        "latitude": 35.216976,
        "longitude": -80.83189,
        "timezone": "GMT",
        "elevation": 254.0,
        "hourly": {
            "time": [
                "2026-04-08T00:00",
                "2026-04-08T01:00",
                "2026-04-08T02:00",
            ],
            "temperature_2m": [
                16.8,
                15.1,
            ],
            "precipitation": [
                0.0,
                0.0,
                0.0,
            ],
        },
    }

    with pytest.raises(ValidationError):
        WeatherResponse.model_validate(data)


def test_null_temperature():
    data = {
        "latitude": 35.216976,
        "longitude": -80.83189,
        "timezone": "GMT",
        "elevation": 254.0,
        "hourly": {
            "time": [
                "2026-04-08T00:00",
                "2026-04-08T01:00",
            ],
            "temperature_2m": [
                16.8,
                None,
            ],
            "precipitation": [
                0.0,
                0.0,
            ],
        },
    }

    with pytest.raises(ValidationError):
        WeatherResponse.model_validate(data)