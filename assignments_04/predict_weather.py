from weather_model import WeatherClassifier


def main() -> None:
    classifier = WeatherClassifier(
        "models/weather_classifier.pkl"
    )

    days = [
        # Clearly good
        {
            "temperature_2m_max": 18.0,
            "temperature_2m_min": 8.0,
            "precipitation_sum": 0.0,
            "wind_speed_10m_max": 10.0,
        },

        # Clearly bad
        {
            "temperature_2m_max": 6.0,
            "temperature_2m_min": -1.0,
            "precipitation_sum": 12.0,
            "wind_speed_10m_max": 35.0,
        },

        # Windy day
        {
            "temperature_2m_max": 17.0,
            "temperature_2m_min": 7.0,
            "precipitation_sum": 0.5,
            "wind_speed_10m_max": 40.0,
        },

        # Borderline conditions
        {
            "temperature_2m_max": 24.0,
            "temperature_2m_min": 6.0,
            "precipitation_sum": 2.5,
            "wind_speed_10m_max": 27.0,
        },
    ]

    predictions = classifier.predict(days)

    for day, prediction in zip(
        days,
        predictions,
    ):
        print(
            f"high={day['temperature_2m_max']}C, "
            f"low={day['temperature_2m_min']}C, "
            f"rain={day['precipitation_sum']}mm, "
            f"wind={day['wind_speed_10m_max']}km/h "
            f"-> {prediction.label}, "
            f"probability={prediction.probability:.3f}"
        )


# Without this guard, main() would run automatically if another
# script imported predict_weather.py just to reuse something from it.

if __name__ == "__main__":
    main()


# --- Task 6: Reflection ---

# I select the features in a fixed order because the model was trained
# with a specific column order. If the columns were passed in the wrong
# order, the model could treat wind as temperature or rain as another
# feature and make a wrong prediction.

# The tests train their own small model instead of using the real model
# file. This makes the tests independent and repeatable. They do not
# break just because the real model file is missing or changed.

# This component is ready for a pipeline because WeatherClassifier can
# load the model once and predict a batch of weather records.
# Before production, I would add better logging and more input validation.