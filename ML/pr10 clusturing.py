# Import libraries
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram, linkage

# Step 1. Load dataset
iris = load_iris()
X = pd.DataFrame(iris.data, columns=iris.feature_names)

# Standardize the data (important for clustering)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Step 2. K-Means Clustering
kmeans = KMeans(n_clusters=3, random_state=42)
kmeans_labels = kmeans.fit_predict(X_scaled)

print("K-Means Cluster Centers:\n", kmeans.cluster_centers_)
print("\nCluster Labels (K-Means):", kmeans_labels[:10])

# Step 3. Hierarchical Clustering (Agglomerative)
hier = AgglomerativeClustering(n_clusters=3)
hier_labels = hier.fit_predict(X_scaled)

# Step 4. Visualization (K-Means)
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.scatter(X_scaled[:, 0], X_scaled[:, 1], c=kmeans_labels, cmap="viridis", s=50)
plt.title("K-Means Clustering (Iris)")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")

# Step 5. Visualization (Hierarchical)
linked = linkage(X_scaled, "ward")
plt.subplot(1, 2, 2)
dendrogram(linked, truncate_mode="lastp", p=10, leaf_rotation=45)
plt.title("Hierarchical Clustering Dendrogram")
plt.xlabel("Sample groups")
plt.ylabel("Distance")

plt.tight_layout()
plt.show()
