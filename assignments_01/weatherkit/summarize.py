from dataclasses import dataclass

from .records import HourlyReading


@dataclass
class DailySummary:
    """Represents the weather summary for one day."""

    date: str
    temp_max: float
    temp_min: float
    precipitation_sum: float
    hours_observed: int

    def temp_range(self) -> float:
        """Return the difference between the highest and lowest temperature."""
        return self.temp_max - self.temp_min


class DailyAggregator:
    """Groups hourly weather readings into daily summaries."""

    def __init__(self, min_hours: int = 24) -> None:
        """Set the minimum number of hours needed for a daily summary."""
        self.min_hours = min_hours

    def summarize(
        self,
        readings: list[HourlyReading],
    ) -> list[DailySummary]:
        """Create daily summaries from hourly readings."""
        grouped = {}

        for reading in readings:
            date = reading.timestamp[:10]

            if date not in grouped:
                grouped[date] = []

            grouped[date].append(reading)

        summaries = []

        for date, daily_readings in grouped.items():
            if len(daily_readings) < self.min_hours:
                continue

            temperatures = [
                reading.temperature_c
                for reading in daily_readings
            ]

            precipitation = [
                reading.precipitation_mm
                for reading in daily_readings
            ]

            summary = DailySummary(
                date=date,
                temp_max=max(temperatures),
                temp_min=min(temperatures),
                precipitation_sum=sum(precipitation),
                hours_observed=len(daily_readings),
            )

            summaries.append(summary)

        return sorted(
            summaries,
            key=lambda summary: summary.date,
        )

    def incomplete_days(
        self,
        readings: list[HourlyReading],
    ) -> list[str]:
        """Return dates that do not have enough hourly readings."""
        grouped = {}

        for reading in readings:
            date = reading.timestamp[:10]

            if date not in grouped:
                grouped[date] = 0

            grouped[date] += 1

        incomplete = []

        for date, count in grouped.items():
            if count < self.min_hours:
                incomplete.append(date)

        return sorted(incomplete)