import numpy as np
import requests
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score


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

response = requests.get(url, params=params)
response.raise_for_status()

df = pd.DataFrame(response.json()["daily"])

df["date"] = pd.to_datetime(df["time"])
df = df.drop("time", axis=1)

print("Dataset shape:", df.shape)

print("\nFirst five rows:")
print(df.head())


# --- Task 2: Look at the Features ---

print("\nData Summary:")
print(df.describe())

print("\nMissing Values:")
print(df.isnull().sum())

# There are no missing values, so I did not need to remove
# or fill any rows.


correlations = (
    df.select_dtypes(include="number")
    .corr()["temperature_2m_max"]
    .sort_values(ascending=False)
)

print("\nCorrelation with Daily High:")
print(correlations)

# Other than the target itself, temperature_2m_min has the strongest
# relationship with the daily high.
# Its correlation is about 0.84, so higher daily lows usually come
# with higher daily highs.


plt.figure()

plt.scatter(
    df["temperature_2m_min"],
    df["temperature_2m_max"],
)

plt.title("Daily High vs Low - San Jose")
plt.xlabel("Daily Low Temperature")
plt.ylabel("Daily High Temperature")

plt.savefig("outputs/project_high_vs_low.png")
plt.close()

# Daily low and daily high have a clear positive relationship.
# Higher low temperatures usually come with higher high temperatures.


# --- Task 3: Engineer a Feature ---

df["is_summer"] = (
    df["date"].dt.month.isin([6, 7, 8])
).astype(int)

summer_days = df["is_summer"].sum()
non_summer_days = len(df) - summer_days

print("\nSummer Days:", summer_days)
print("Non-Summer Days:", non_summer_days)

# I think is_summer can help because summer days are usually warmer
# than the rest of the year.


# --- Task 4: Baseline Model ---

X_baseline = df[["temperature_2m_min"]].values
y = df["temperature_2m_max"].values

X_train_base, X_test_base, y_train_base, y_test_base = train_test_split(
    X_baseline,
    y,
    test_size=0.2,
    random_state=42,
)

baseline_model = LinearRegression()
baseline_model.fit(X_train_base, y_train_base)

y_pred_base = baseline_model.predict(X_test_base)

baseline_rmse = np.sqrt(
    mean_squared_error(
        y_test_base,
        y_pred_base,
    )
)

baseline_r2 = r2_score(
    y_test_base,
    y_pred_base,
)

print("\nBaseline Model")
print("Slope:", baseline_model.coef_[0])
print("RMSE:", baseline_rmse)
print("R²:", baseline_r2)

# The slope is about 1.25, so when the daily low increases by 1 degree,
# the predicted daily high increases by about 1.25 degrees.

# The RMSE is about 3.75 C.
# This means a typical prediction is off by about 3.75 degrees Celsius.


# --- Task 5: Full Model ---

feature_cols = [
    "temperature_2m_min",
    "precipitation_sum",
    "wind_speed_10m_max",
    "is_summer",
]

X = df[feature_cols].values
y = df["temperature_2m_max"].values

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
)

model = LinearRegression()
model.fit(X_train, y_train)

train_pred = model.predict(X_train)
y_pred = model.predict(X_test)

train_r2 = r2_score(
    y_train,
    train_pred,
)

test_r2 = r2_score(
    y_test,
    y_pred,
)

full_rmse = np.sqrt(
    mean_squared_error(
        y_test,
        y_pred,
    )
)

print("\nFull Model")
print("Train R²:", train_r2)
print("Test R²:", test_r2)
print("RMSE:", full_rmse)
print("Baseline Test R²:", baseline_r2)
print("R² Improvement:", test_r2 - baseline_r2)

print("\nFeature Coefficients:")

for name, coef in zip(feature_cols, model.coef_):
    print(f"{name:22s}: {coef:+.3f}")


# The full model performs better than the baseline model.
# Test R² increased from about 0.68 to 0.78.
# RMSE also decreased from about 3.75 C to 3.12 C.

# temperature_2m_min has a positive coefficient.
# A higher daily low raises the predicted daily high.

# precipitation_sum has a negative coefficient.
# More precipitation lowers the predicted high when the other
# features stay the same.

# wind_speed_10m_max also has a small negative coefficient.
# Stronger wind slightly lowers the predicted high.

# is_summer has a positive coefficient of about 2.65.
# A summer day raises the predicted high by about 2.65 degrees
# when the other features stay the same.

# The wind result surprised me a little.
# Wind had a positive correlation with daily high by itself,
# but its coefficient became slightly negative in the full model.

# Train R² and test R² are very close.
# This tells me the model performs similarly on training and test data
# and does not show a big generalization gap.


# --- Task 6: Evaluate and Summarize ---

plt.figure()

plt.scatter(
    y_pred,
    y_test,
)

minimum = min(
    y_pred.min(),
    y_test.min(),
)

maximum = max(
    y_pred.max(),
    y_test.max(),
)

plt.plot(
    [minimum, maximum],
    [minimum, maximum],
)

plt.title("Predicted vs Actual (Full Model)")
plt.xlabel("Predicted High")
plt.ylabel("Actual High")

plt.savefig(
    "outputs/predicted_vs_actual_high.png"
)

plt.close()

# The errors are spread across the range.
# There are a few bigger misses in the low and middle temperatures,
# but the model does not clearly struggle more at the high end.


# Summary
#
# The dataset has 365 days of weather data.
# The test set has 73 days.
#
# The full model has an RMSE of about 3.12 C and an R² of about 0.78.
# A typical prediction is off by about 3.12 degrees Celsius.
#
# is_summer has the largest coefficient, about +2.65.
# A summer day raises the predicted high when the other features
# stay the same.
#
# One thing that surprised me was the wind result.
# Wind had a small positive correlation with daily high by itself,
# but its coefficient became slightly negative in the full model.