# Import libraries
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

import seaborn as sns


# Load dataset
df = pd.read_csv("Mall_Customers.csv")   # <- make sure this file is in the same directory

# Show first few rows
print(df.head())

# Select features for clustering (e.g., Annual Income and Spending Score)
X = df[['Annual Income (k$)', 'Spending Score (1-100)']]

# Elbow Method to find optimal number of clusters
wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, init='k-means++', random_state=42)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)

# Plot Elbow curve
plt.figure(figsize=(8,5))
plt.plot(range(1,11), wcss, marker='o')
plt.title("Elbow Method")
plt.xlabel("Number of Clusters")
plt.ylabel("WCSS")
plt.show()

# Apply KMeans with optimal clusters (let’s assume 5)
kmeans = KMeans(n_clusters=5, init='k-means++', random_state=42)
y_kmeans = kmeans.fit_predict(X)

# Add cluster info to dataset
df['Cluster'] = y_kmeans

# Visualize clusters
plt.figure(figsize=(8,6))
sns.scatterplot(x='Annual Income (k$)', y='Spending Score (1-100)', 
                hue='Cluster', palette='Set2', data=df, s=100)
plt.scatter(kmeans.cluster_centers_[:,0], kmeans.cluster_centers_[:,1], 
            s=300, c='red', marker='X', label='Centroids')
plt.title("Customer Segments (K-means Clustering)")
plt.legend()
plt.show()

# Print cluster summary (numeric columns only)
numeric_cols = df.select_dtypes(include='number').columns
print(df.groupby('Cluster')[numeric_cols].mean())
