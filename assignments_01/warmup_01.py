from dataclasses import dataclass, field, FrozenInstanceError

import pytest
from pydantic import BaseModel, Field, ValidationError, model_validator


# ============================================================
# --- Classes ---
# ============================================================

# Q1


class Thermometer:
    def __init__(self, location: str, readings: list[float] | None = None):
        self.location = location

        if readings is None:
            self.readings = []
        else:
            self.readings = readings

    def add(self, reading: float) -> None:
        self.readings.append(reading)

    def average(self) -> float | None:
        if not self.readings:
            return None

        return sum(self.readings) / len(self.readings)

    def hottest(self) -> float | None:
        if not self.readings:
            return None

        return max(self.readings)

    # Q2
    def __repr__(self) -> str:
        average = self.average()

        if average is None:
            average_text = "None"
        else:
            average_text = f"{average:.1f}"

        return (
            f"Thermometer(location={self.location!r}, "
            f"n_readings={len(self.readings)}, "
            f"average={average_text})"
        )


my_thermometer = Thermometer("San Jose")

my_thermometer.add(20.5)
my_thermometer.add(24.0)
my_thermometer.add(22.5)
my_thermometer.add(27.0)

print("Average:", my_thermometer.average())
print("Hottest:", my_thermometer.hottest())

# average() needs to check if the list is empty because otherwise it would
# divide by 0 and give a ZeroDivisionError.


# Q2

print("\n--- Thermometer __repr__ ---")

print(my_thermometer)

second_thermometer = Thermometer(
    "San Francisco",
    [16.0, 18.5, 19.0, 17.5],
)

print([my_thermometer, second_thermometer])

# Without __repr__, Python shows something like:
# <__main__.Thermometer object at 0x...>
# That does not really tell me what is inside the object, so it is not very
# helpful when debugging.


# Q3


class TemperatureAlert:
    def __init__(self, threshold: float = 30.0):
        self.threshold = threshold

    def breaches(self, thermometer: Thermometer) -> list[float]:
        return [
            reading
            for reading in thermometer.readings
            if reading > self.threshold
        ]


alert_22 = TemperatureAlert(22.0)
alert_25 = TemperatureAlert(25.0)

print("\n--- Temperature Alerts ---")

print("Above 22 C:", alert_22.breaches(my_thermometer))
print("Above 25 C:", alert_25.breaches(my_thermometer))

# I keep threshold inside TemperatureAlert because it belongs to that alert.
# If I have 20 thermometers, I can create the alert once and use the same
# threshold for all of them instead of passing it every time.


# ============================================================
# --- Dataclasses, Type Hints, and Docstrings ---
# ============================================================

# Dataclass Q1


@dataclass(frozen=True)
class Station:
    """Represents a weather station and its location information."""

    station_id: str
    name: str
    latitude: float
    longitude: float
    elevation: float


station_a = Station(
    station_id="SJC",
    name="San Jose",
    latitude=37.3382,
    longitude=-121.8863,
    elevation=25.0,
)

station_b = Station(
    station_id="SJC",
    name="San Jose",
    latitude=37.3382,
    longitude=-121.8863,
    elevation=25.0,
)

print("\n--- Dataclass Equality ---")

print("station_a == station_b:", station_a == station_b)

# This prints True because dataclass compares the field values.
# With the original normal class it would be False because station_a and
# station_b are two different objects.


# Dataclass Q2

# For Q2 I added frozen=True to Station.

print("\n--- Frozen Dataclass ---")

try:
    station_a.name = "Different Name"
except FrozenInstanceError as e:
    print("FrozenInstanceError:", e)


station_c = Station(
    station_id="SFO",
    name="San Francisco",
    latitude=37.6213,
    longitude=-122.3790,
    elevation=4.0,
)

station_set = {
    station_a,
    station_b,
    station_c,
}

print("Number of unique stations:", len(station_set))

# frozen=True makes the object immutable and also makes it hashable when its
# fields are hashable. Because of that I can use Station objects in a set.
# station_a and station_b are the same, so the set only keeps one of them.


# Dataclass Q3

# At first I tried:
#
# stations: list[Station] = []
#
# and got:
#
# ValueError: mutable default <class 'list'> for field stations is not allowed:
# use default_factory
#
# Python does not allow this because the same list could be shared between
# different StationBatch objects. default_factory=list gives each object its
# own new list.


@dataclass
class StationBatch:
    """Represents a group of stations in the same region."""

    region: str
    stations: list[Station] = field(default_factory=list)

    def add(self, station: Station) -> None:
        """Add a station to the batch."""
        self.stations.append(station)

    def highest(self) -> Station | None:
        """Return the station with the highest elevation."""
        if not self.stations:
            return None

        return max(
            self.stations,
            key=lambda station: station.elevation,
        )


bay_area = StationBatch("Bay Area")

bay_area.add(station_a)
bay_area.add(station_c)

print("\n--- Station Batch ---")

print("Highest station:", bay_area.highest())


# ============================================================
# --- Pydantic ---
# ============================================================

# Pydantic Q1


class Reading(BaseModel):
    station_id: str = Field(min_length=3)
    timestamp: str
    temperature_c: float = Field(ge=-90, le=60)
    humidity: float = Field(ge=0, le=100)

    # Q4
    @model_validator(mode="after")
    def check_failed_sensor(self) -> "Reading":
        if self.humidity == 0.0 and self.temperature_c < -40:
            raise ValueError(
                "humidity 0.0 with temperature below -40 "
                "looks like a failed sensor"
            )

        return self


valid_reading = Reading(
    station_id="SJC",
    timestamp="2026-10-06T12:00:00",
    temperature_c=22.5,
    humidity=45.0,
)

print("\n--- Valid Pydantic Reading ---")

print(valid_reading)


# Pydantic Q2

print("\n--- Pydantic Validation Errors ---")


# Missing required field

try:
    Reading(
        station_id="SJC",
        temperature_c=20.0,
        humidity=50.0,
    )

except ValidationError as e:
    print("\nMissing required field:")
    print(e)


# temperature_c = 150

try:
    Reading(
        station_id="SJC",
        timestamp="2026-10-06T13:00:00",
        temperature_c=150.0,
        humidity=50.0,
    )

except ValidationError as e:
    print("\nTemperature too high:")
    print(e)


# humidity = "very humid"

try:
    Reading(
        station_id="SJC",
        timestamp="2026-10-06T14:00:00",
        temperature_c=21.0,
        humidity="very humid",
    )

except ValidationError as e:
    print("\nInvalid humidity:")
    print(e)


converted_reading = Reading(
    station_id="SJC",
    timestamp="2026-10-06T15:00:00",
    temperature_c="21.5",
    humidity=40,
)

print("\n--- Pydantic Type Conversion ---")

print(converted_reading)

print(
    "temperature_c type:",
    type(converted_reading.temperature_c),
)

print(
    "humidity type:",
    type(converted_reading.humidity),
)

# Pydantic accepts "21.5" because it can convert that string to a float.
# It rejects "very humid" because that cannot be converted to a number.


# Pydantic Q3

print("\n--- Multiple Pydantic Errors ---")

try:
    Reading(
        station_id="A",
        temperature_c="not a number",
        humidity=50.0,
    )

except ValidationError as e:
    for error in e.errors():
        print("Location:", error["loc"])
        print("Message:", error["msg"])
        print()


# It reports 3 errors: station_id is too short, timestamp is missing,
# and temperature_c is not a number.
# Getting all of them together is better because I can fix everything at once
# instead of running the code again for every error.


# Pydantic Q4

print("\n--- Model Validator ---")


normal_reading = Reading(
    station_id="SJC",
    timestamp="2026-10-06T16:00:00",
    temperature_c=-20.0,
    humidity=0.0,
)

print("Valid combination:", normal_reading)


try:
    Reading(
        station_id="SJC",
        timestamp="2026-10-06T17:00:00",
        temperature_c=-50.0,
        humidity=0.0,
    )

except ValidationError as e:
    print("\nFailed sensor combination:")
    print(e)


# Field can validate one field by itself, but this rule depends on humidity
# and temperature together. That is why model_validator makes more sense here.


# ============================================================
# --- pytest ---
# ============================================================

# pytest Q1


def celsius_to_fahrenheit(celsius: float) -> float:
    """Convert Celsius to Fahrenheit."""

    return (celsius * 9 / 5) + 32


def test_celsius_to_fahrenheit():
    assert celsius_to_fahrenheit(0) == 32
    assert celsius_to_fahrenheit(100) == 212
    assert celsius_to_fahrenheit(37) == pytest.approx(98.6)

    # I used pytest.approx because floating point numbers are not always stored
    # exactly, so using == can fail even when the answer is basically the same.


# pytest Q2


def mean(values: list[float]) -> float:
    """Return the mean of a list of numbers."""

    if not values:
        raise ValueError("values cannot be empty")

    return sum(values) / len(values)


def test_mean_of_empty_raises():
    with pytest.raises(
        ValueError,
        match="empty",
    ):
        mean([])

    # pytest.raises(ValueError) only checks the error type.
    # match="empty" also checks that I got the error message I expected.


# pytest Q3


@pytest.mark.parametrize(
    "values, expected",
    [
        ([5.0], 5.0),
        ([1.0, 2.0, 3.0], 2.0),
        ([-3.0, -2.0, -1.0], -2.0),
        ([10.0, 20.0, 30.0, 40.0], 25.0),
    ],
)
def test_mean_values(
    values: list[float],
    expected: float,
):
    assert mean(values) == pytest.approx(expected)


# After running:
# pytest warmup_01.py -v
#
# I got:
#
# ========================== 6 passed in 0.13s ==========================


# Parametrize is better here because I can test different inputs with the same
# test instead of writing almost the same function four times.


# pytest Q4

# I temporarily changed:
#
# return (celsius * 9 / 5) + 32
#
# to:
#
# return (celsius * 9 / 4) + 32
#
# Then I ran:
#
# pytest warmup_01.py -v
#
# This was the failure:
#
# __________________________________ test_celsius_to_fahrenheit __________________________________
#
#     def test_celsius_to_fahrenheit():
#         assert celsius_to_fahrenheit(0) == 32
# >       assert celsius_to_fahrenheit(100) == 212
# E       assert 257.0 == 212
# E        +  where 257.0 = celsius_to_fahrenheit(100)
#
# FAILED warmup_01.py::test_celsius_to_fahrenheit - assert 257.0 == 212
# ========================== 1 failed, 5 passed in 0.12s ==========================
#
# Pytest showed me that my function returned 257.0 but it should be 212.
# That helps more than only saying "assertion failed" because I can see
# exactly what went wrong.
#
# After that I changed the formula back to 9 / 5.