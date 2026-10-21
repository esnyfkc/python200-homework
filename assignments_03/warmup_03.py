import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, ConfusionMatrixDisplay, classification_report,
)
import joblib

df = pd.read_csv("weather_classification.csv")
numeric = ["temperature_2m_max", "temperature_2m_min", "precipitation_sum", "wind_speed_10m_max"]
X = df[numeric]
y = df["good_for_running"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)


# --- Preprocessing ---

# Q1

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("Scaled training means:")
print(X_train_scaled.mean(axis=0))

# I fit the scaler only on X_train because using X_test while fitting
# would leak information from the test data into the training process.


# Q2

encoder = OneHotEncoder(sparse_output=False)

season_encoded = encoder.fit_transform(
    df[["season"]]
)

print("\nOne-Hot Encoded Season Shape:")
print(season_encoded.shape)

print("Season Feature Names:")
print(encoder.get_feature_names_out())

# The season column became 4 columns because there are 4 categories:
# fall, spring, summer, and winter.


# Q3

features_with_season = numeric + ["season"]
X_with_season = df[features_with_season]

X_train_season, X_test_season, y_train_season, y_test_season = train_test_split(
    X_with_season,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y,
)

preprocessor = ColumnTransformer(
    transformers=[
        (
            "numeric",
            StandardScaler(),
            numeric,
        ),
        (
            "season",
            OneHotEncoder(),
            ["season"],
        ),
    ]
)

preprocessing_pipeline = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("model", LogisticRegression(max_iter=1000)),
    ]
)

preprocessing_pipeline.fit(
    X_train_season,
    y_train_season,
)

preprocessing_accuracy = preprocessing_pipeline.score(
    X_test_season,
    y_test_season,
)

print("\nPipeline Test Accuracy:")
print(preprocessing_accuracy)

# The pipeline handles the scaling and one-hot encoding automatically,
# so I do not have to preprocess them separately by hand.


# --- KNN ---

# Q1

knn_pipeline = Pipeline(
    steps=[
        ("scaler", StandardScaler()),
        ("knn", KNeighborsClassifier(n_neighbors=5)),
    ]
)

knn_pipeline.fit(
    X_train,
    y_train,
)

knn_pred = knn_pipeline.predict(X_test)

knn_accuracy = accuracy_score(
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


# Q2

k_values = [1, 3, 5, 7, 9, 11, 15, 21]

best_k = None
best_score = 0

print("\nKNN Cross-Validation Scores:")

for k in k_values:
    knn_cv_pipeline = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("knn", KNeighborsClassifier(n_neighbors=k)),
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

    print(f"k={k:2d}  Mean CV Accuracy: {mean_score:.4f}")

    if mean_score > best_score:
        best_score = mean_score
        best_k = k

print("\nBest k:", best_k)
print("Best CV Accuracy:", best_score)

# I would choose k=11 because it had the highest mean
# cross-validation accuracy, about 0.942.


# Q3

knn_raw = KNeighborsClassifier(
    n_neighbors=5
)

knn_raw.fit(
    X_train,
    y_train,
)

raw_pred = knn_raw.predict(X_test)

raw_accuracy = accuracy_score(
    y_test,
    raw_pred,
)


knn_scaled = Pipeline(
    steps=[
        ("scaler", StandardScaler()),
        ("knn", KNeighborsClassifier(n_neighbors=5)),
    ]
)

knn_scaled.fit(
    X_train,
    y_train,
)

scaled_pred = knn_scaled.predict(X_test)

scaled_accuracy = accuracy_score(
    y_test,
    scaled_pred,
)

print("\nKNN Without Scaling Accuracy:")
print(raw_accuracy)

print("KNN With Scaling Accuracy:")
print(scaled_accuracy)

# Scaling helped because the accuracy increased from 0.925
# to about 0.958.
# KNN uses distance, so features with bigger scales can affect
# the distance more than the other features.


# --- Logistic Regression ---

# Q1

pipeline = Pipeline(
    steps=[
        ("scaler", StandardScaler()),
        ("logreg", LogisticRegression(max_iter=1000)),
    ]
)

pipeline.fit(
    X_train,
    y_train,
)

logreg_pred = pipeline.predict(X_test)

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

print("\nLogistic Regression Metrics:")
print("Accuracy:", logreg_accuracy)
print("Precision:", logreg_precision)
print("Recall:", logreg_recall)
print("F1:", logreg_f1)


# Q2

logreg_model = pipeline.named_steps["logreg"]
coefficients = logreg_model.coef_[0]

print("\nLogistic Regression Coefficients:")

for feature, coef in zip(
    numeric,
    coefficients,
):
    print(f"{feature:22s}: {coef:+.3f}")

# Wind speed and precipitation push the model most strongly
# toward "skip" because they have the biggest negative coefficients.
# That makes sense because strong wind and heavy rain are not
# good conditions for running.


# Q3

good_probabilities = pipeline.predict_proba(
    X_test.iloc[:5]
)[:, 1]

print("\nProbability of Good for First 5 Test Days:")

for probability in good_probabilities:
    print(round(probability, 2))

# A probability close to 0.5 means the model is not very confident.
# One of my days was 0.47, so that prediction was close to
# the decision boundary.


# --- Evaluation ---

# Q1

cm = confusion_matrix(
    y_test,
    logreg_pred,
)

print("\nConfusion Matrix:")
print(cm)

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=["skip", "good"],
)

disp.plot()

plt.title("Logistic Regression Confusion Matrix")
plt.savefig("outputs/logreg_confusion_matrix.png")
plt.close()

# The model made 9 false positives and 7 false negatives.
# For a running app, false positives matter more to me because
# the app could recommend running on a day that is actually bad.
# A false negative only means missing a good running day.


# --- joblib ---

# Q1

joblib.dump(
    pipeline,
    "models/warmup_model.pkl",
)

loaded = joblib.load(
    "models/warmup_model.pkl"
)

assert (
    loaded.predict(X_test)
    == pipeline.predict(X_test)
).all()

print("\nPredictions match.")

# If I saved only LogisticRegression without the scaler,
# the model would expect already-scaled values.
# Raw weather data would not be on the same scale used during training.