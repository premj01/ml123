import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# Step 1. Load dataset
iris = load_iris()
X = pd.DataFrame(iris.data, columns=iris.feature_names)

# Step 2. Preprocessing – scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Step 3. Feature selection – use 2 main features for easy visualization
X_selected = X_scaled[:, :2]

# Step 4. K-Means clustering
k = 3  # known number of species
kmeans = KMeans(n_clusters=k, random_state=42)
clusters = kmeans.fit_predict(X_selected)

# Step 5. Evaluation – Silhouette score
score = silhouette_score(X_selected, clusters)
print("Silhouette Score:", round(score, 3))

# Step 6. Visualization
plt.scatter(X_selected[:, 0], X_selected[:, 1], c=clusters, cmap="viridis", s=50)
plt.scatter(
    kmeans.cluster_centers_[:, 0],
    kmeans.cluster_centers_[:, 1],
    color="red",
    marker="X",
    s=200,
    label="Centroids",
)
plt.title("K-Means Clustering on Iris Dataset")
plt.xlabel("Feature 1 (Scaled)")
plt.ylabel("Feature 2 (Scaled)")
plt.legend()
plt.show()
