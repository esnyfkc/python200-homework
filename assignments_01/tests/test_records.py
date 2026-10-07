import json
from pathlib import Path

from weatherkit import WeatherResponse, HourlyReading, to_readings


def test_to_readings_keeps_order():
    file_path = Path(__file__).parent.parent / "weather_raw.json"

    with open(file_path) as file:
        data = json.load(file)

    response = WeatherResponse.model_validate(data)
    readings = to_readings(response)

    assert len(readings) == 168
    assert readings[0].timestamp == "2026-04-08T00:00"
    assert readings[-1].timestamp == "2026-04-14T23:00"


def test_to_readings_matches_input_values():
    file_path = Path(__file__).parent.parent / "weather_raw.json"

    with open(file_path) as file:
        data = json.load(file)

    response = WeatherResponse.model_validate(data)
    readings = to_readings(response)

    i = 10

    assert readings[i].timestamp == response.hourly.time[i]
    assert readings[i].temperature_c == response.hourly.temperature_2m[i]
    assert readings[i].precipitation_mm == response.hourly.precipitation[i]


def test_hourly_readings_with_same_values_are_equal():
    reading_a = HourlyReading(
        timestamp="2026-04-08T00:00",
        temperature_c=16.8,
        precipitation_mm=0.0,
    )

    reading_b = HourlyReading(
        timestamp="2026-04-08T00:00",
        temperature_c=16.8,
        precipitation_mm=0.0,
    )

    assert reading_a == reading_b