import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold
import umap
import hdbscan
import os
import warnings
warnings.filterwarnings('ignore')

# 1. Load Data
data_path = r"C:\Users\igork\Desktop\main\ai-program-2026\project_1\data\waka_dragon_merged.csv"
print(f"Loading data from {data_path}...")
df = pd.read_csv(data_path)

# 2. Select Features
# Drop identifier and target columns
exclude_cols = ['CID', 'CAS', 'Name', 'Imax', 'Ci', 'Di_x', 'intensity_class']
feature_cols = [c for c in df.columns if c not in exclude_cols]

X = df[feature_cols].select_dtypes(include=[np.number])
print(f"Initial numeric features: {X.shape[1]}")

# 3. Preprocessing
print("Handling missing values and constants...")
# Drop columns with > 20% missing
X = X.dropna(thresh=int(X.shape[0] * 0.8), axis=1)
# Impute remaining with median
X = X.fillna(X.median())

# Remove constant features
selector = VarianceThreshold(threshold=0.01)
X_var = selector.fit_transform(X)
print(f"Features after variance threshold: {X_var.shape[1]}")

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_var)

# 4. Tune UMAP and HDBSCAN parameters
# We'll create a grid of plots
neighbors_list = [15, 30]
min_dist_list = [0.0, 0.1]
min_cluster_sizes = [5, 10]

output_dir = r"C:\Users\igork\Desktop\main\Project\plots"
os.makedirs(output_dir, exist_ok=True)

print("Running UMAP + HDBSCAN Grid...")

fig, axes = plt.subplots(len(neighbors_list) * len(min_dist_list), len(min_cluster_sizes), figsize=(15, 20))

plot_idx_row = 0
for n_neighbors in neighbors_list:
    for min_dist in min_dist_list:
        # Run UMAP
        print(f"UMAP: n_neighbors={n_neighbors}, min_dist={min_dist}")
        reducer = umap.UMAP(n_neighbors=n_neighbors, min_dist=min_dist, n_components=2, random_state=42)
        embedding = reducer.fit_transform(X_scaled)
        
        plot_idx_col = 0
        for mcs in min_cluster_sizes:
            print(f"  HDBSCAN: min_cluster_size={mcs}")
            clusterer = hdbscan.HDBSCAN(min_cluster_size=mcs, min_samples=3, gen_min_span_tree=True)
            labels = clusterer.fit_predict(embedding)
            
            # Number of clusters (excluding noise -1)
            n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
            noise_points = list(labels).count(-1)
            
            ax = axes[plot_idx_row, plot_idx_col]
            
            # Scatter plot
            scatter = ax.scatter(embedding[:, 0], embedding[:, 1], c=labels, cmap='Spectral', s=15, alpha=0.8)
            ax.set_title(f"UMAP(nn={n_neighbors}, md={min_dist})\nHDBSCAN(mcs={mcs}) -> {n_clusters} cls, {noise_points} noise")
            ax.axis('off')
            
            plot_idx_col += 1
        plot_idx_row += 1

plt.tight_layout()
out_file = os.path.join(output_dir, "umap_hdbscan_grid.png")
plt.savefig(out_file, dpi=150)
print(f"Saved 2D grid to {out_file}")

# Also let's do a 3D UMAP plot for one good setting
print("Running 3D UMAP...")
reducer_3d = umap.UMAP(n_neighbors=30, min_dist=0.0, n_components=3, random_state=42)
embedding_3d = reducer_3d.fit_transform(X_scaled)
clusterer_3d = hdbscan.HDBSCAN(min_cluster_size=10, min_samples=3)
labels_3d = clusterer_3d.fit_predict(embedding_3d)

fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
scatter_3d = ax.scatter(embedding_3d[:, 0], embedding_3d[:, 1], embedding_3d[:, 2], 
                        c=labels_3d, cmap='Spectral', s=20, alpha=0.8)
ax.set_title(f"3D UMAP (nn=30, md=0.0) + HDBSCAN (mcs=10)\nClusters: {len(set(labels_3d))-1}")
out_file_3d = os.path.join(output_dir, "umap_hdbscan_3d.png")
plt.savefig(out_file_3d, dpi=150)
print(f"Saved 3D plot to {out_file_3d}")

print("Done!")
