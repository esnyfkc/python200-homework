import joblib
import pandas as pd
import pytest

from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from weather_model import Prediction, WeatherClassifier


@pytest.fixture
def classifier(tmp_path):
    X_train = pd.DataFrame(
        [
            {
                "temperature_2m_max": 20.0,
                "temperature_2m_min": 10.0,
                "precipitation_sum": 0.0,
                "wind_speed_10m_max": 10.0,
            },
            {
                "temperature_2m_max": 22.0,
                "temperature_2m_min": 12.0,
                "precipitation_sum": 0.5,
                "wind_speed_10m_max": 12.0,
            },
            {
                "temperature_2m_max": 5.0,
                "temperature_2m_min": -2.0,
                "precipitation_sum": 10.0,
                "wind_speed_10m_max": 40.0,
            },
            {
                "temperature_2m_max": 3.0,
                "temperature_2m_min": -4.0,
                "precipitation_sum": 15.0,
                "wind_speed_10m_max": 45.0,
            },
        ]
    )

    y_train = [1, 1, 0, 0]

    pipeline = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("knn", KNeighborsClassifier(n_neighbors=1)),
        ]
    )

    pipeline.fit(
        X_train,
        y_train,
    )

    model_path = tmp_path / "tiny_model.pkl"

    joblib.dump(
        pipeline,
        model_path,
    )

    return WeatherClassifier(
        str(model_path)
    )


@pytest.mark.parametrize(
    "day, expected_label",
    [
        (
            {
                "temperature_2m_max": 20.0,
                "temperature_2m_min": 10.0,
                "precipitation_sum": 0.0,
                "wind_speed_10m_max": 10.0,
            },
            "good",
        ),
        (
            {
                "temperature_2m_max": 5.0,
                "temperature_2m_min": -2.0,
                "precipitation_sum": 10.0,
                "wind_speed_10m_max": 40.0,
            },
            "skip",
        ),
    ],
)
def test_expected_label(classifier, day, expected_label):
    result = classifier.predict([day])

    assert result[0].label == expected_label


def test_probability_is_between_zero_and_one(classifier):
    day = {
        "temperature_2m_max": 20.0,
        "temperature_2m_min": 10.0,
        "precipitation_sum": 0.0,
        "wind_speed_10m_max": 10.0,
    }

    result = classifier.predict([day])

    assert 0.0 <= result[0].probability <= 1.0


def test_three_days_return_three_predictions_in_order(classifier):
    days = [
        {
            "temperature_2m_max": 20.0,
            "temperature_2m_min": 10.0,
            "precipitation_sum": 0.0,
            "wind_speed_10m_max": 10.0,
        },
        {
            "temperature_2m_max": 5.0,
            "temperature_2m_min": -2.0,
            "precipitation_sum": 10.0,
            "wind_speed_10m_max": 40.0,
        },
        {
            "temperature_2m_max": 22.0,
            "temperature_2m_min": 12.0,
            "precipitation_sum": 0.5,
            "wind_speed_10m_max": 12.0,
        },
    ]

    results = classifier.predict(days)

    assert len(results) == 3

    assert all(
        isinstance(result, Prediction)
        for result in results
    )

    assert results[0].label == "good"
    assert results[1].label == "skip"
    assert results[2].label == "good"


def test_empty_list_returns_empty_list(classifier):
    assert classifier.predict([]) == []


def test_missing_feature_raises_value_error(classifier):
    day = {
        "temperature_2m_max": 20.0,
        "temperature_2m_min": 10.0,
        "precipitation_sum": 0.0,
    }

    with pytest.raises(
        ValueError,
        match="wind_speed_10m_max",
    ):
        classifier.predict([day])


def test_missing_model_file_raises_file_not_found(tmp_path):
    missing_path = tmp_path / "missing_model.pkl"

    with pytest.raises(
        FileNotFoundError,
        match="Model file not found",
    ):
        WeatherClassifier(str(missing_path))


# I temporarily changed classifier.py to always return "skip".
# test_expected_label and test_three_days_return_three_predictions_in_order
# caught the change, then I changed it back.