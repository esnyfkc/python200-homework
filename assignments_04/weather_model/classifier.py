from dataclasses import dataclass
from pathlib import Path

import joblib
import pandas as pd


@dataclass
class Prediction:
    """Stores the result of one weather prediction."""

    label: str
    probability: float


class WeatherClassifier:
    """Loads a saved weather model and makes predictions."""

    required_features = [
        "temperature_2m_max",
        "temperature_2m_min",
        "precipitation_sum",
        "wind_speed_10m_max",
    ]

    def __init__(self, model_path: str) -> None:
        """Load a saved model from disk.

        Raises:
            FileNotFoundError: If the model file does not exist.
        """
        path = Path(model_path)

        if not path.exists():
            raise FileNotFoundError(
                f"Model file not found: {model_path}"
            )

        self.model = joblib.load(path)

    def predict(
        self,
        days: list[dict[str, float]],
    ) -> list[Prediction]:
        """Predict whether each day is good for running.

        Args:
            days: A list of weather dictionaries.

        Returns:
            A list of Prediction objects in the same order as the input.

        Raises:
            ValueError: If a day is missing a required feature.
        """
        if not days:
            return []

        for day in days:
            missing = [
                feature
                for feature in self.required_features
                if feature not in day
            ]

            if missing:
                raise ValueError(
                    f"Missing required features: {missing}"
                )

        data = pd.DataFrame(days)

        data = data[
            self.required_features
        ]

        labels = self.model.predict(data)

        probabilities = self.model.predict_proba(
            data
        )[:, 1]

        results = []

        for label, probability in zip(
            labels,
            probabilities,
        ):
            prediction_label = (
                "good"
                if label == 1
                else "skip"
            )

            results.append(
                Prediction(
                    label=prediction_label,
                    probability=float(probability),
                )
            )

        return results