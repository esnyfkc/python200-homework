import json
import os
from dataclasses import dataclass

import pytest


# Q1

class ModelInfo:
    def __init__(self, path):
        with open(path, "r") as file:
            self.data = json.load(file)

    def feature_count(self) -> int:
        return len(self.data["features"])


sample_data = {
    "features": [
        "temperature",
        "rain",
        "wind",
    ]
}

sample_path = "sample_model_info.json"

with open(sample_path, "w") as file:
    json.dump(sample_data, file)


model_info = ModelInfo(sample_path)

print("Feature count:", model_info.feature_count())

os.remove(sample_path)

# Loading the file in __init__ is better because it only reads the file once.
# If feature_count() is called many times, it can use the data already loaded.


# Q2

@dataclass
class DayForecast:
    date: str
    high_c: float
    will_rain: bool

    def summary(self) -> str:
        if self.will_rain:
            rain_text = "rain expected"
        else:
            rain_text = "no rain expected"

        return f"{self.date}: high {self.high_c}C, {rain_text}"


day1 = DayForecast(
    date="2023-06-15",
    high_c=24.1,
    will_rain=True,
)

day2 = DayForecast(
    date="2023-06-16",
    high_c=27.5,
    will_rain=False,
)

print(day1.summary())
print(day2.summary())

# @dataclass created the __init__ method for me,
# so I did not have to write it by hand.


# Q3

def parse_temperature(text: str) -> float:
    try:
        return float(text)
    except ValueError:
        raise ValueError("temperature must be a number")


def test_parse_temperature_valid():
    assert parse_temperature("24.1") == pytest.approx(24.1)


def test_parse_temperature_bad_input_raises():
    with pytest.raises(ValueError, match="temperature"):
        parse_temperature("hot")


# Using pytest.raises(ValueError) without match would only check
# the error type. It would not catch a wrong or unhelpful error message.