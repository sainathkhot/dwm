# Import necessary libraries
import numpy as np
from sklearn_extra.cluster import KMedoids
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt

# Step 1: Create synthetic data using make_blobs
# Generate 2-dimensional data with 3 clusters
X, y_true = make_blobs(n_samples=300, centers=3, cluster_std=0.60, random_state=42)

# Step 2: Visualize the synthetic data
plt.scatter(X[:, 0], X[:, 1], s=50)
plt.title("Synthetic Dataset")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.show()

# Step 3: Apply K-medoids clustering
kmedoids = KMedoids(n_clusters=3, random_state=42, method='pam')  # 'pam' method for K-medoids
kmedoids.fit(X)  # Fit K-medoids to the data

# Step 4: Retrieve cluster labels and medoids
y_kmedoids = kmedoids.predict(X)
medoids = kmedoids.cluster_centers_

# Step 5: Visualize the clusters and medoids
plt.scatter(X[:, 0], X[:, 1], c=y_kmedoids, s=50, cmap='viridis')

# Plot the medoids (centers)
plt.scatter(medoids[:, 0], medoids[:, 1], c='red', s=200, alpha=0.75, marker='X', label="Medoids")
plt.title("K-medoids Clustering with 3 Clusters")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.legend()
plt.show()
