import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score


# --- scikit-learn API ---

# Q1

temp_min = np.array([1, 4, 8, 12, 16, 20]).reshape(-1, 1)
temp_max = np.array([8, 10, 15, 18, 23, 27])

model = LinearRegression()
model.fit(temp_min, temp_max)

predictions = model.predict(
    np.array([
        [6],
        [18],
    ])
)

print("Slope:", model.coef_[0])
print("Intercept:", model.intercept_)
print("Prediction for low 6:", predictions[0])
print("Prediction for low 18:", predictions[1])


# Q2

x = np.array([10, 20, 30, 40, 50])

print("\nOriginal shape:", x.shape)

x_2d = x.reshape(-1, 1)

print("New shape:", x_2d.shape)

# scikit-learn needs X to be 2D because each row is one data point
# and each column is one feature.


# --- Linear Regression ---

rng = np.random.default_rng(42)
num_days = 120
temp_min = rng.uniform(-5, 22, num_days)
is_rainy = rng.integers(0, 2, num_days).astype(float)
temp_max = 1.0 * temp_min + 6 - 4 * is_rainy + rng.normal(0, 2, num_days)


# Q1

plt.figure()

plt.scatter(
    temp_min,
    temp_max,
    c=is_rainy,
    cmap="coolwarm",
)

plt.title("Daily High vs Low")
plt.xlabel("Daily Low Temperature")
plt.ylabel("Daily High Temperature")

plt.savefig("outputs/high_vs_low.png")
plt.close()

# I can see two groups that follow a similar pattern.
# Rainy days are generally lower than dry days.
# This suggests is_rainy could be useful for predicting the daily high.


# Q2

X = temp_min.reshape(-1, 1)
y = temp_max

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
)

print("\nTrain/Test Shapes")
print("X_train:", X_train.shape)
print("X_test:", X_test.shape)
print("y_train:", y_train.shape)
print("y_test:", y_test.shape)


# Q3

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

rmse = np.sqrt(
    mean_squared_error(y_test, y_pred)
)

r2 = model.score(X_test, y_test)

print("\nOne Feature Model")
print("Slope:", model.coef_[0])
print("Intercept:", model.intercept_)
print("RMSE:", rmse)
print("R²:", r2)

# The slope is about 0.94.
# This means when the daily low increases by 1 degree,
# the predicted daily high increases by about 0.94 degrees.


# Q4

X_full = np.column_stack([temp_min, is_rainy])

X_train_full, X_test_full, y_train_full, y_test_full = train_test_split(
    X_full,
    temp_max,
    test_size=0.2,
    random_state=42,
)

model_full = LinearRegression()
model_full.fit(X_train_full, y_train_full)

y_pred_full = model_full.predict(X_test_full)

r2_full = model_full.score(
    X_test_full,
    y_test_full,
)

print("\nTwo Feature Model")
print("One feature R²:", r2)
print("Two feature R²:", r2_full)

print("temp_min coefficient:", model_full.coef_[0])
print("is_rainy coefficient:", model_full.coef_[1])

# Adding is_rainy helped because the test R² increased.
# The is_rainy coefficient is about -2.92.
# This means a rainy day lowers the predicted high by about 2.92 degrees
# compared with a dry day when temp_min stays the same.


# Q5

plt.figure()

plt.scatter(
    y_pred_full,
    y_test_full,
)

minimum = min(
    y_pred_full.min(),
    y_test_full.min(),
)

maximum = max(
    y_pred_full.max(),
    y_test_full.max(),
)

plt.plot(
    [minimum, maximum],
    [minimum, maximum],
)

plt.title("Predicted vs Actual")
plt.xlabel("Predicted High")
plt.ylabel("Actual High")

plt.savefig(
    "outputs/predicted_vs_actual_warmup.png"
)

plt.close()

# If a point is above the diagonal, the actual value is higher than
# the prediction, so the model predicted too low.
# If a point is below the diagonal, the prediction is higher than
# the actual value, so the model predicted too high.