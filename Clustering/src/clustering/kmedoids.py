import pandas as pd
import numpy as np
import random

from visualization.plot import visualize_clusters_dynamic


NORM_TYPE = 'L2'  # Options: 'L1', 'L2', 'L∞'

def load_clustering_data(file_path: str):
    return pd.read_csv(file_path).apply(pd.to_numeric, errors='coerce').dropna()

def compute_distance(point1, point2):
    if NORM_TYPE == 'L1':
        return np.sum(np.abs(point1 - point2))
    elif NORM_TYPE == 'L2':
        return np.linalg.norm(point1 - point2)
    elif NORM_TYPE == 'L∞':
        return np.max(np.abs(point1 - point2))

def generate_initial_random_medoids(data: pd.DataFrame, k: int):
    random_indices = random.sample(range(len(data)), k)
    return [data.iloc[idx].values.tolist() for idx in random_indices]


def assign_clusters(data: pd.DataFrame, medoids: list):
    cluster_assignment = []
    for _, row in data.iterrows():
        distances = [compute_distance(row.values, np.array(medoid)) for medoid in medoids]
        closest_medoid = np.argmin(distances)
        cluster_assignment.append(int(closest_medoid))
    return cluster_assignment


def calculate_new_medoids(data: pd.DataFrame, cluster_assignment: list, k: int):
    new_medoids = []
    for cluster_idx in range(k):
        cluster_points = data[np.array(cluster_assignment) == cluster_idx]
        if len(cluster_points) > 0:
            distances = cluster_points.apply(lambda point: np.sum(
                [compute_distance(point.values, other_point)
                 for other_point in cluster_points.values]), axis=1)
            medoid_idx = distances.idxmin()
            new_medoids.append(cluster_points.loc[medoid_idx].values.tolist())
    return new_medoids


def cluster_by_k_medoids(file_path: str, k: int, maxIter: int):
    data = load_clustering_data(file_path)
    medoids = generate_initial_random_medoids(data, k)
    cluster_assignment = assign_clusters(data, medoids)

    for _ in range(maxIter):
        medoids = calculate_new_medoids(data, cluster_assignment, k)
        new_cluster_assignment = assign_clusters(data, medoids)
        if new_cluster_assignment == cluster_assignment:
            break
        cluster_assignment = new_cluster_assignment

    return medoids, cluster_assignment


def update_centroids_and_assignments(data: pd.DataFrame, centroids: list):
    new_cluster_assignment = assign_clusters(data, centroids)
    new_centroids = calculate_new_medoids(data, new_cluster_assignment, len(centroids))
    return new_centroids, new_cluster_assignment


def cluster_by_k_medoids_visual(file_path: str, k: int):
    data = load_clustering_data(file_path)
    medoids = generate_initial_random_medoids(data, k)
    cluster_assignment = assign_clusters(data, medoids)

    visualize_clusters_dynamic(data, cluster_assignment, medoids, update_centroids_and_assignments, 'K-Medoids')


if __name__ == "__main__":
    bear_attack_path = 'data/BearAttacks/Processed_BearAttacks.csv'
    blob_path = 'data/Blobs/blob_dataset.csv'
    concentric_circles_path = 'data/Blobs/concentric_circles.csv'
    path = 'data/Blobs/challenging_data.csv'

    file_path = path
    k = 4
    cluster_by_k_medoids_visual(file_path, k)
