import json
from pathlib import Path

from weatherkit import WeatherResponse, to_readings, DailyAggregator


def main() -> None:
    """Load the weather data and print daily summaries."""

    file_path = Path(__file__).parent / "weather_raw.json"

    with open(file_path) as file:
        data = json.load(file)

    response = WeatherResponse.model_validate(data)

    # Task 2 check
    print("Latitude:", response.latitude)
    print("Timezone:", response.timezone)
    print("Hourly observations:", len(response.hourly.time))
    print()

    readings = to_readings(response)

    aggregator = DailyAggregator()

    summaries = aggregator.summarize(readings)
    incomplete = aggregator.incomplete_days(readings)

    print(
        f"{'Date':<12}"
        f"{'High':>8}"
        f"{'Low':>8}"
        f"{'Precip':>10}"
        f"{'Range':>10}"
    )

    print("-" * 48)

    for summary in summaries:
        print(
            f"{summary.date:<12}"
            f"{summary.temp_max:>8.1f}"
            f"{summary.temp_min:>8.1f}"
            f"{summary.precipitation_sum:>10.1f}"
            f"{summary.temp_range():>10.1f}"
        )

    if incomplete:
        print("\nIncomplete days:", ", ".join(incomplete))


if __name__ == "__main__":
    main()


# The if __name__ == "__main__" check makes sure main() only runs when I run
# this file directly. If someone imports report.py from another file, the
# report will not automatically print.


# Reflection

# 1.
# Right now WeatherResponse rejects the whole file if one temperature is null.
# This can be useful when I need complete data and do not want missing values
# to affect the results.
#
# But if I have a lot of good data and only one hour is missing, I might want
# to keep the rest instead of rejecting everything.
#
# To allow that, I could change:
#
# temperature_2m: list[float]
#
# to:
#
# temperature_2m: list[float | None]
#
# Then I would also need to handle the missing values later in the program.


# 2.
# If the pipeline runs at noon, the current day only has around 12 hours of
# data. With min_hours=24, that day would be dropped even though nothing is
# actually wrong yet. The day is just not finished.
#
# incomplete_days() helps because it shows which dates were dropped instead
# of making them disappear without telling us.


# 3.
# Having weatherkit as a package makes it easier to reuse the code.
# In Week 10, another pipeline can import WeatherResponse, to_readings,
# or DailyAggregator instead of copying the same code into another file.