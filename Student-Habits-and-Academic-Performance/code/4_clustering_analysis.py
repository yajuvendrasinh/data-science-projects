import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from pathlib import Path

# Set the style for visualizations
plt.style.use('fivethirtyeight')
sns.set_palette('viridis')

# Create directories for output
Path('../visualizations').mkdir(exist_ok=True)
Path('../output').mkdir(exist_ok=True)

# Load the preprocessed dataset
print("Loading preprocessed dataset...")
df = pd.read_csv('../output/encoded_data_for_modeling.csv')

# Define features for clustering
print("\nPreparing data for clustering analysis...")
# Exclude student_id and target variable from features
features = [col for col in df.columns if col != 'exam_score' and col != 'student_id']

# Extract features for clustering
X = df[features]

# Standardize the features
print("Standardizing features...")
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Determine optimal number of clusters using the Elbow Method
print("\nDetermining optimal number of clusters using Elbow Method...")
inertia = []
silhouette_scores = []
k_range = range(2, 11)

for k in k_range:
    print(f"Testing k={k}...")
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(X_scaled)
    inertia.append(kmeans.inertia_)
    silhouette_scores.append(silhouette_score(X_scaled, kmeans.labels_))

# Save elbow method results
elbow_results = pd.DataFrame({
    'k': list(k_range),
    'inertia': inertia,
    'silhouette_score': silhouette_scores
})
elbow_results.to_csv('../output/kmeans_elbow_results.csv', index=False)

# Plot Elbow Method results
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.plot(k_range, inertia, 'o-')
plt.xlabel('Number of Clusters (k)', fontsize=14)
plt.ylabel('Inertia', fontsize=14)
plt.title('Elbow Method for Optimal k', fontsize=16)
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(k_range, silhouette_scores, 'o-')
plt.xlabel('Number of Clusters (k)', fontsize=14)
plt.ylabel('Silhouette Score', fontsize=14)
plt.title('Silhouette Score for Optimal k', fontsize=16)
plt.grid(True)

plt.tight_layout()
plt.savefig('../visualizations/kmeans_elbow_method.png', dpi=300, bbox_inches='tight')
plt.close()

# Choose optimal k based on elbow method and silhouette score
optimal_k = k_range[np.argmax(silhouette_scores)]
print(f"\nOptimal number of clusters based on silhouette score: {optimal_k}")

# Apply K-means with optimal k
print(f"\nApplying K-means clustering with k={optimal_k}...")
kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
clusters = kmeans.fit_predict(X_scaled)

# Add cluster labels to the original dataframe
df['cluster'] = clusters

# Save clustered data
df.to_csv('../output/clustered_data.csv', index=False)

# Apply PCA for visualization
print("\nApplying PCA for dimensionality reduction and visualization...")
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# Create a dataframe for PCA results
pca_df = pd.DataFrame({
    'PCA1': X_pca[:, 0],
    'PCA2': X_pca[:, 1],
    'cluster': clusters,
    'exam_score': df['exam_score']
})

# Save PCA results
pca_df.to_csv('../output/pca_results.csv', index=False)

# Visualize clusters using PCA
plt.figure(figsize=(12, 10))
scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=clusters, cmap='viridis', alpha=0.5)
plt.colorbar(scatter, label='Cluster')
plt.xlabel('Principal Component 1', fontsize=14)
plt.ylabel('Principal Component 2', fontsize=14)
plt.title('Clusters Visualization using PCA', fontsize=16)
plt.tight_layout()
plt.savefig('../visualizations/kmeans_clusters_pca.png', dpi=300, bbox_inches='tight')
plt.close()

# Analyze clusters
print("\nAnalyzing clusters...")
cluster_analysis = df.groupby('cluster').agg({
    'exam_score': 'mean',
    'study_hours_per_day': 'mean',
    'sleep_hours': 'mean',
    'social_media_hours': 'mean',
    'netflix_hours': 'mean',
    'attendance_percentage': 'mean',
    'wellness_score': 'mean',
    'academic_engagement': 'mean',
    'stress_level': 'mean',
    'screen_time': 'mean',
    'time_management_score': 'mean'
}).reset_index()

# Save cluster analysis
cluster_analysis.to_csv('../output/cluster_analysis.csv', index=False)

# Visualize key characteristics of each cluster
plt.figure(figsize=(15, 10))
sns.heatmap(cluster_analysis.set_index('cluster').T, annot=True, cmap='viridis', fmt='.2f')
plt.title('Key Characteristics of Each Cluster', fontsize=16)
plt.tight_layout()
plt.savefig('../visualizations/cluster_characteristics_heatmap.png', dpi=300, bbox_inches='tight')
plt.close()

# Visualize exam score distribution by cluster
plt.figure(figsize=(12, 8))
sns.boxplot(x='cluster', y='exam_score', data=df)
plt.title('Exam Score Distribution by Cluster', fontsize=16)
plt.xlabel('Cluster', fontsize=14)
plt.ylabel('Exam Score', fontsize=14)
plt.tight_layout()
plt.savefig('../visualizations/exam_score_by_cluster.png', dpi=300, bbox_inches='tight')
plt.close()

# Create radar charts for each cluster
print("\nCreating radar charts for cluster profiles...")
# Select features for radar chart
radar_features = ['study_hours_per_day', 'sleep_hours', 'social_media_hours', 
                 'attendance_percentage', 'wellness_score', 'academic_engagement', 
                 'stress_level', 'time_management_score']

# Normalize the features for radar chart
radar_df = cluster_analysis[['cluster'] + radar_features].copy()
for feature in radar_features:
    max_val = radar_df[feature].max()
    min_val = radar_df[feature].min()
    radar_df[feature] = (radar_df[feature] - min_val) / (max_val - min_val)

# Create radar charts
for cluster_id in range(optimal_k):
    cluster_data = radar_df[radar_df['cluster'] == cluster_id].iloc[0, 1:].values
    
    # Create radar chart
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, polar=True)
    
    # Set the angles for each feature
    angles = np.linspace(0, 2*np.pi, len(radar_features), endpoint=False).tolist()
    angles += angles[:1]  # Close the loop
    
    # Add the cluster data
    cluster_data = np.append(cluster_data, cluster_data[0])
    ax.plot(angles, cluster_data, 'o-', linewidth=2, label=f'Cluster {cluster_id}')
    ax.fill(angles, cluster_data, alpha=0.25)
    
    # Set the labels
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(radar_features, fontsize=12)
    
    # Add title
    plt.title(f'Profile of Cluster {cluster_id}', fontsize=16)
    
    # Save the radar chart
    plt.tight_layout()
    plt.savefig(f'../visualizations/cluster_{cluster_id}_radar.png', dpi=300, bbox_inches='tight')
    plt.close()

print("\nClustering analysis completed and results saved.")
