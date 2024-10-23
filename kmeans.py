# Import necessary libraries
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt

# Step 1: Create synthetic data using make_blobs
# 2-dimensional data with 3 clusters
X, y_true = make_blobs(n_samples=300, centers=3, cluster_std=0.60, random_state=42)

# Step 2: Create a DataFrame for better visualization
df = pd.DataFrame(X, columns=['Feature 1', 'Feature 2'])
df['True Label'] = y_true

# Display the first few rows of the dataset
print("Synthetic Dataset:")
print(df.head())

# Step 3: Visualize the synthetic data
plt.scatter(X[:, 0], X[:, 1], s=50, cmap='viridis')
plt.title("Synthetic Dataset")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.show()

# Step 4: Apply KMeans clustering
kmeans = KMeans(n_clusters=3, random_state=42)
kmeans.fit(X)  # Fit K-means to the data

# Step 5: Retrieve cluster centers and predictions
y_kmeans = kmeans.predict(X)
centers = kmeans.cluster_centers_

# Step 6: Visualize the clusters with centroids
plt.scatter(X[:, 0], X[:, 1], c=y_kmeans, s=50, cmap='viridis')

# Plot the cluster centers (centroids)
plt.scatter(centers[:, 0], centers[:, 1], c='red', s=200, alpha=0.75, marker='X', label="Centroids")
plt.title("K-means Clustering with 3 Clusters")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.legend()
plt.show()
