import json

import joblib
import pandas as pd


# --- Task 7: Load the Model and Make Predictions ---

model = joblib.load(
    "models/weather_classifier.pkl"
)

with open(
    "models/weather_classifier_metadata.json",
    "r",
) as file:
    metadata = json.load(file)


print("Model:", metadata["model"])
print("City:", metadata["city"])
print("Features:", metadata["features"])
print("Test Accuracy:", round(metadata["test_accuracy"], 3))
print("Test F1:", round(metadata["test_f1"], 3))


# I made a few different weather days to test the saved model.

example_days = pd.DataFrame(
    [
        # Good day
        {
            "temperature_2m_max": 18.0,
            "temperature_2m_min": 8.0,
            "precipitation_sum": 0.0,
            "wind_speed_10m_max": 10.0,
        },

        # Rainy day
        {
            "temperature_2m_max": 16.0,
            "temperature_2m_min": 8.0,
            "precipitation_sum": 10.0,
            "wind_speed_10m_max": 15.0,
        },

        # Windy day
        {
            "temperature_2m_max": 17.0,
            "temperature_2m_min": 7.0,
            "precipitation_sum": 0.5,
            "wind_speed_10m_max": 40.0,
        },

        # Borderline day - close to the rain and wind limits
        {
            "temperature_2m_max": 24.0,
            "temperature_2m_min": 6.0,
            "precipitation_sum": 2.5,
            "wind_speed_10m_max": 27.0,
        },
    ]
)


# Use the same feature order used during training.

example_days = example_days[
    metadata["features"]
]


predictions = model.predict(
    example_days
)

good_probabilities = model.predict_proba(
    example_days
)[:, 1]


print("\nPredictions:")

for i in range(len(example_days)):
    if predictions[i] == 1:
        label = "good"
    else:
        label = "skip"

    print(f"\nDay {i + 1}")

    for feature in metadata["features"]:
        print(
            f"{feature}: "
            f"{example_days.iloc[i][feature]}"
        )

    print("Prediction:", label)

    print(
        "Probability of good:",
        round(good_probabilities[i], 3),
    )


# --- Task 8: Reflection ---

# I used Day 4 as my borderline day because the rain and wind
# are close to the limits. The model gave it a probability of 0.089,
# so it was actually pretty confident about "skip".
# If the probability was 0.52, I would treat it as uncertain
# instead of making a strong "good" recommendation.

# If predict.py ran before train_classifier.py, the saved model files
# would not exist and joblib.load would fail.
# I could catch FileNotFoundError and print a message telling the user
# to run train_classifier.py first.

# For the cloud pipeline, I would replace these hard-coded example days
# with the real weather data passed into the pipeline each day.