import pandas as pd
import numpy as np

from visualization.plot import visualize_clusters

NORM_TYPE = 'L2'  # Options: 'L1', 'L2', 'L∞'

# Load and preprocess the data
def load_clustering_data(file_path: str):
    return pd.read_csv(file_path).apply(pd.to_numeric, errors='coerce').dropna()


# Calculate the Euclidean distance between two points
def calculate_norm(point1, point2):
    if NORM_TYPE == 'L1':
        return np.sum(np.abs(np.array(point1) - np.array(point2)))
    elif NORM_TYPE == 'L2':
        return np.linalg.norm(np.array(point1) - np.array(point2))
    elif NORM_TYPE == 'L∞':
        return np.max(np.abs(np.array(point1) - np.array(point2)))

# Find neighbors within the epsilon (eps) radius
def find_neighbors(data: pd.DataFrame, point_index: int, eps: float):
    neighbors = []
    for idx in range(len(data)):
        if calculate_norm(data.iloc[point_index], data.iloc[idx]) <= eps:
            neighbors.append(idx)
    return neighbors


# Expand the cluster with density-reachable points
def expand_cluster(data: pd.DataFrame, cluster_assignment: list, point_index: int, neighbors: list, cluster_id: int,
                   eps: float, min_pts: int):
    cluster_assignment[point_index] = cluster_id
    i = 0
    while i < len(neighbors):
        neighbor_idx = neighbors[i]
        if cluster_assignment[neighbor_idx] == -1:  # If it was previously marked as noise
            cluster_assignment[neighbor_idx] = cluster_id
        elif cluster_assignment[neighbor_idx] == 0:  # Unvisited point
            cluster_assignment[neighbor_idx] = cluster_id
            new_neighbors = find_neighbors(data, neighbor_idx, eps)
            if len(new_neighbors) >= min_pts:
                neighbors.extend(new_neighbors)
        i += 1


# DBSCAN algorithm
def cluster_by_dbscan(file_path: str, eps: float, min_samples: int):
    data = load_clustering_data(file_path)
    cluster_labels = [0] * len(data)  # 0 means unvisited
    cluster_id = 0

    for point_idx in range(len(data)):
        if cluster_labels[point_idx] != 0:  # If already visited
            continue
        neighbors = find_neighbors(data, point_idx, eps)
        if len(neighbors) < min_samples:  # Not enough neighbors, mark as noise
            cluster_labels[point_idx] = -1
        else:
            cluster_id += 1  # Start a new cluster
            expand_cluster(data, cluster_labels, point_idx, neighbors, cluster_id, eps, min_samples)

    return cluster_labels


if __name__ == '__main__':
    file_path = 'data/Blobs/moons.csv'
    eps = 0.2
    min_pts = 5

    cluster_labels = cluster_by_dbscan(file_path, eps, min_pts)

    data = load_clustering_data(file_path)
    visualize_clusters(data, cluster_labels)
