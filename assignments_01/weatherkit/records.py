from dataclasses import dataclass

from .schemas import WeatherResponse


@dataclass
class HourlyReading:
    """Represents one hourly weather reading.

    temperature_c is in Celsius.
    precipitation_mm is in millimeters.
    """

    timestamp: str
    temperature_c: float
    precipitation_mm: float


def to_readings(response: WeatherResponse) -> list[HourlyReading]:
    """Convert the API response into hourly reading objects.

    Args:
        response: Validated weather API response.

    Returns:
        A list of HourlyReading objects in the same order as the API data.
    """
    readings = []

    for i in range(len(response.hourly.time)):
        reading = HourlyReading(
            timestamp=response.hourly.time[i],
            temperature_c=response.hourly.temperature_2m[i],
            precipitation_mm=response.hourly.precipitation[i],
        )

        readings.append(reading)

    return readings


# HourlyReading is a dataclass because this data is already inside the program
# after WeatherResponse validated it.
# WeatherResponse is Pydantic because it is where outside API data first enters
# the program and needs to be checked.

# I temporarily changed temperature_c to always use index 0.
# test_to_readings_matches_input_values caught it because the values
# no longer matched the same index from the API data.