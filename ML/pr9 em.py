import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt


# Step 1. Simulate patient health data
np.random.seed(42)
data = {
    "Age": [25, 30, 35, np.nan, 50, 45, np.nan, 60],
    "Blood Pressure": [120, 130, np.nan, 110, 140, np.nan, 135, 150],
    "Heart Rate": [72, 75, 70, np.nan, 80, 85, 78, np.nan],
}

df = pd.DataFrame(data)
print("Original data with missing values:\n", df)

# Step 2. Handle missing values roughly (mean imputation as initialization)
imputer = SimpleImputer(strategy="mean")
X_init = imputer.fit_transform(df)

# Step 3. Apply EM using Gaussian Mixture
gmm = GaussianMixture(n_components=2, max_iter=100, random_state=42)
gmm.fit(X_init)

# Step 4. Replace missing values with EM-based estimates
X_completed = X_init.copy()
missing_mask = df.isnull().values

for i in range(len(df)):
    if missing_mask[i].any():
        # Sample from GMM to estimate missing values (approximation)
        X_completed[i] = gmm.sample(1)[0]

# Step 5. Create completed DataFrame
completed_df = pd.DataFrame(X_completed, columns=df.columns)
print("\nImputed Data (after EM):\n", completed_df.round(2))


plt.scatter(df["Age"], df["Blood Pressure"], color="red", label="Original (missing)")
plt.scatter(
    completed_df["Age"], completed_df["Blood Pressure"], color="green", label="Imputed"
)
plt.xlabel("Age")
plt.ylabel("Blood Pressure")
plt.title("EM-based Imputation Visualization")
plt.legend()
plt.show()
