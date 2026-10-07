import pytest

from weatherkit import HourlyReading, DailyAggregator


@pytest.fixture
def sample_readings():
    return [
        HourlyReading("2026-04-08T00:00", 10.0, 0.0),
        HourlyReading("2026-04-08T01:00", 12.0, 0.5),
        HourlyReading("2026-04-08T02:00", 8.0, 1.0),
        HourlyReading("2026-04-09T00:00", 15.0, 0.0),
        HourlyReading("2026-04-09T01:00", 18.0, 0.2),
    ]


def test_grouping_creates_two_summaries(sample_readings):
    aggregator = DailyAggregator(min_hours=1)

    summaries = aggregator.summarize(sample_readings)

    assert len(summaries) == 2
    assert summaries[0].date == "2026-04-08"
    assert summaries[1].date == "2026-04-09"


def test_temp_max_and_min(sample_readings):
    aggregator = DailyAggregator(min_hours=1)

    summaries = aggregator.summarize(sample_readings)

    first_day = summaries[0]

    assert first_day.temp_max == 12.0
    assert first_day.temp_min == 8.0


def test_precipitation_sum(sample_readings):
    aggregator = DailyAggregator(min_hours=1)

    summaries = aggregator.summarize(sample_readings)

    first_day = summaries[0]

    assert first_day.precipitation_sum == pytest.approx(1.5)


def test_incomplete_day_is_dropped(sample_readings):
    aggregator = DailyAggregator(min_hours=3)

    summaries = aggregator.summarize(sample_readings)
    incomplete = aggregator.incomplete_days(sample_readings)

    assert len(summaries) == 1
    assert summaries[0].date == "2026-04-08"
    assert "2026-04-09" in incomplete


def test_lower_min_hours_keeps_day(sample_readings):
    aggregator = DailyAggregator(min_hours=2)

    summaries = aggregator.summarize(sample_readings)

    assert len(summaries) == 2


@pytest.mark.parametrize(
    "temp_max, temp_min, expected",
    [
        (20.0, 10.0, 10.0),
        (15.0, 15.0, 0.0),
        (5.0, -5.0, 10.0),
    ],
)
def test_temp_range(temp_max, temp_min, expected):
    from weatherkit import DailySummary

    summary = DailySummary(
        date="2026-04-08",
        temp_max=temp_max,
        temp_min=temp_min,
        precipitation_sum=0.0,
        hours_observed=24,
    )

    assert summary.temp_range() == expected