# Import libraries
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Load dataset
housing = fetch_california_housing()
X = pd.DataFrame(housing.data, columns=housing.feature_names)[
    ["MedInc"]
]  # one feature for clarity
y = housing.target

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Transform features for polynomial regression
poly = PolynomialFeatures(degree=2)  # degree can be tuned (2, 3, etc.)
X_poly_train = poly.fit_transform(X_train)
X_poly_test = poly.transform(X_test)

# Train model
model = LinearRegression()
model.fit(X_poly_train, y_train)

# Predict
y_pred = model.predict(X_poly_test)

# Evaluate
print("Mean Squared Error:", mean_squared_error(y_test, y_pred))
print("R² Score:", r2_score(y_test, y_pred))

# ---- Minimal Graph with Regression Curve ----
plt.scatter(X_test, y_test, alpha=0.4, label="Actual Values")

# Sort for smooth curve plotting
sorted_axis = np.argsort(X_test.values.flatten())
X_sorted = X_test.values.flatten()[sorted_axis]
y_sorted_pred = y_pred[sorted_axis]

plt.plot(
    X_sorted,
    y_sorted_pred,
    color="red",
    linewidth=2,
    label="Polynomial Regression Line",
)
plt.xlabel("Median Income")
plt.ylabel("House Value")
plt.title("Polynomial Regression (Degree = 2)")
plt.legend()
plt.show()
