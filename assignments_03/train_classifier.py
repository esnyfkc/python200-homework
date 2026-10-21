import json
import sys

import joblib
import pandas as pd
import requests
import sklearn

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
)


# --- Task 1: Fetch the Data ---

# I chose San Jose, California.

url = "https://archive-api.open-meteo.com/v1/archive"

params = {
    "latitude": 37.3382,
    "longitude": -121.8863,
    "start_date": "2023-01-01",
    "end_date": "2023-12-31",
    "daily": [
        "temperature_2m_max",
        "temperature_2m_min",
        "precipitation_sum",
        "wind_speed_10m_max",
    ],
    "timezone": "America/Los_Angeles",
}

response = requests.get(
    url,
    params=params,
)

response.raise_for_status()

df = pd.DataFrame(
    response.json()["daily"]
)

df["date"] = pd.to_datetime(
    df["time"]
)

df = df.drop(
    "time",
    axis=1,
)

print("Dataset shape:", df.shape)

print("\nFirst five rows:")
print(df.head())


# --- Task 2: Engineer the Label ---

df["good_for_running"] = (
    (df["temperature_2m_max"] >= 7)
    & (df["temperature_2m_max"] <= 26)
    & (df["temperature_2m_min"] >= 0)
    & (df["precipitation_sum"] < 3)
    & (df["wind_speed_10m_max"] < 30)
).astype(int)


print("\nClass Balance:")
print(
    df["good_for_running"].value_counts()
)

good_days = df["good_for_running"].sum()
total_days = len(df)
good_fraction = good_days / total_days

print("\nGood Days:", good_days)
print("Total Days:", total_days)
print("Fraction Good:", round(good_fraction, 3))

# About 55% of the days are good for running and 45% are skip days.
# The classes are fairly balanced, which seems reasonable for San Jose.
# Accuracy should be useful here, but I still want to check precision,
# recall, and F1 because the two types of mistakes are different.


# --- Task 3: KNN Classifier ---

feature_cols = [
    "temperature_2m_max",
    "temperature_2m_min",
    "precipitation_sum",
    "wind_speed_10m_max",
]

X = df[feature_cols]
y = df["good_for_running"]


X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y,
)


k_values = [
    1,
    3,
    5,
    7,
    9,
    11,
    15,
    21,
]

best_k = None
best_cv_score = 0

print("\nKNN Cross-Validation Scores:")

for k in k_values:
    knn_cv_pipeline = Pipeline(
        steps=[
            (
                "scaler",
                StandardScaler(),
            ),
            (
                "knn",
                KNeighborsClassifier(
                    n_neighbors=k
                ),
            ),
        ]
    )

    scores = cross_val_score(
        knn_cv_pipeline,
        X_train,
        y_train,
        cv=5,
        scoring="accuracy",
    )

    mean_score = scores.mean()

    print(
        f"k={k:2d}  "
        f"Mean CV Accuracy: {mean_score:.4f}"
    )

    if mean_score > best_cv_score:
        best_cv_score = mean_score
        best_k = k


print("\nBest k:", best_k)
print("Best CV Accuracy:", best_cv_score)


knn_model = Pipeline(
    steps=[
        (
            "scaler",
            StandardScaler(),
        ),
        (
            "knn",
            KNeighborsClassifier(
                n_neighbors=best_k
            ),
        ),
    ]
)

knn_model.fit(
    X_train,
    y_train,
)

knn_pred = knn_model.predict(
    X_test
)


knn_accuracy = accuracy_score(
    y_test,
    knn_pred,
)

knn_precision = precision_score(
    y_test,
    knn_pred,
)

knn_recall = recall_score(
    y_test,
    knn_pred,
)

knn_f1 = f1_score(
    y_test,
    knn_pred,
)


print("\nKNN Test Accuracy:")
print(knn_accuracy)

print("\nKNN Classification Report:")
print(
    classification_report(
        y_test,
        knn_pred,
    )
)

# I chose k=5 because it had the highest mean cross-validation
# accuracy, about 0.935.
# On the test set, KNN got about 0.918 accuracy.


# --- Task 4: Logistic Regression Classifier ---

logreg_model = Pipeline(
    steps=[
        (
            "scaler",
            StandardScaler(),
        ),
        (
            "logreg",
            LogisticRegression(
                max_iter=1000
            ),
        ),
    ]
)

logreg_model.fit(
    X_train,
    y_train,
)

logreg_pred = logreg_model.predict(
    X_test
)


logreg_accuracy = accuracy_score(
    y_test,
    logreg_pred,
)

logreg_precision = precision_score(
    y_test,
    logreg_pred,
)

logreg_recall = recall_score(
    y_test,
    logreg_pred,
)

logreg_f1 = f1_score(
    y_test,
    logreg_pred,
)


print("\nLogistic Regression Test Accuracy:")
print(logreg_accuracy)

print("Precision:", logreg_precision)
print("Recall:", logreg_recall)
print("F1:", logreg_f1)

print("\nLogistic Regression Classification Report:")
print(
    classification_report(
        y_test,
        logreg_pred,
    )
)


logreg = logreg_model.named_steps[
    "logreg"
]

print("\nLogistic Regression Coefficients:")

for feature, coef in zip(
    feature_cols,
    logreg.coef_[0],
):
    print(
        f"{feature:22s}: {coef:+.3f}"
    )

# Precipitation pushes the model most strongly toward "skip".
# Higher maximum temperature and stronger wind also push toward "skip".
# Minimum temperature has only a small positive effect toward "good".


# --- Task 5: Compare the Models and Choose One ---

knn_cm = confusion_matrix(
    y_test,
    knn_pred,
)

logreg_cm = confusion_matrix(
    y_test,
    logreg_pred,
)

knn_fp = knn_cm[0, 1]
knn_fn = knn_cm[1, 0]

logreg_fp = logreg_cm[0, 1]
logreg_fn = logreg_cm[1, 0]


print("\nModel Comparison:")

print("\nKNN:")
print("Accuracy:", knn_accuracy)
print("Precision:", knn_precision)
print("Recall:", knn_recall)
print("F1:", knn_f1)
print("False Positives:", knn_fp)
print("False Negatives:", knn_fn)

print("\nLogistic Regression:")
print("Accuracy:", logreg_accuracy)
print("Precision:", logreg_precision)
print("Recall:", logreg_recall)
print("F1:", logreg_f1)
print("False Positives:", logreg_fp)
print("False Negatives:", logreg_fn)


# KNN did better overall with about 0.918 accuracy and 0.927 F1.
# Logistic Regression had lower accuracy, but its precision was higher,
# about 0.943.
#
# KNN made 4 false positives and 2 false negatives.
# Logistic Regression made 2 false positives and 7 false negatives.
#
# I care more about false positives for this app because I do not want
# it to recommend running when the weather is actually bad.
#
# I would choose Logistic Regression even though KNN was more accurate.
# It is small, fast to predict, and easier to understand because I can
# see how each weather feature affects the result.
# I can also use predict_proba to see how confident the model is.


# --- Task 6: Save the Model and Metadata ---

chosen_model = logreg_model

joblib.dump(
    chosen_model,
    "models/weather_classifier.pkl",
)


metadata = {
    "model": "Logistic Regression",
    "city": "San Jose, California",
    "features": feature_cols,
    "label_rule": {
        "temperature_2m_max": "between 7 and 26 C",
        "temperature_2m_min": "at least 0 C",
        "precipitation_sum": "less than 3 mm",
        "wind_speed_10m_max": "less than 30 km/h",
    },
    "test_accuracy": logreg_accuracy,
    "test_f1": logreg_f1,
    "python_version": sys.version.split()[0],
    "sklearn_version": sklearn.__version__,
}


with open(
    "models/weather_classifier_metadata.json",
    "w",
) as file:
    json.dump(
        metadata,
        file,
        indent=4,
    )


print("\nSaved model:")
print("models/weather_classifier.pkl")

print("Saved metadata:")
print(
    "models/weather_classifier_metadata.json"
)

# I saved the full pipeline so the scaler and model stay together.
# I also saved the model information so predict.py can use
# the same features.