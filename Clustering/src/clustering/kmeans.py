import pandas as pd
import numpy as np
import random

from visualization.plot import visualize_clusters_with_centroids
from visualization.plot import visualize_clusters_dynamic


def load_clustering_data(file_path: str):
    return pd.read_csv(file_path)


def generate_initial_random_centroids(data: pd.DataFrame, k: int):
    min_first_col = data.iloc[:, 0].min()
    max_first_col = data.iloc[:, 0].max()
    min_second_col = data.iloc[:, 1].min()
    max_second_col = data.iloc[:, 1].max()

    return [(round(random.uniform(min_first_col, max_first_col), 2),
             round(random.uniform(min_second_col, max_second_col), 2)) for _ in range(k)]


def assign_clusters(data: pd.DataFrame, centroids: list):
    cluster_assignment = []
    for index, row in data.iterrows():
        distances = [np.linalg.norm(np.array(row) - np.array(centroid)) for centroid in centroids]
        closest_centroid = np.argmin(distances)
        cluster_assignment.append(closest_centroid)
    return cluster_assignment


def calculate_new_centroids(data: pd.DataFrame, centroids: list, cluster_assignment: list):
    num_centroids = len(centroids)
    new_centroids = [(0, 0) for _ in range(num_centroids)]
    counts = [0 for _ in range(num_centroids)]

    for idx, point in data.iterrows():
        cluster_idx = cluster_assignment[idx]
        new_centroids[cluster_idx] = (new_centroids[cluster_idx][0] + point.iloc[0],
                                      new_centroids[cluster_idx][1] + point.iloc[1])
        counts[cluster_idx] += 1

    for i in range(num_centroids):
        if counts[i] > 0:
            new_centroids[i] = (new_centroids[i][0] / counts[i], new_centroids[i][1] / counts[i])

    return new_centroids


def update_centroids_and_assignments(data: pd.DataFrame, centroids: list):
    new_cluster_assignment = assign_clusters(data, centroids)
    new_centroids = calculate_new_centroids(data, centroids, new_cluster_assignment)
    return new_centroids, new_cluster_assignment


def cluster_by_k_means_with_iterations(file_path: str, k: int, max_iterations: int):
    data = load_clustering_data(file_path)
    centroids = generate_initial_random_centroids(data, k)
    cluster_assignment = assign_clusters(data, centroids)

    for i in range(max_iterations):
        new_centroids = calculate_new_centroids(data, centroids, cluster_assignment)
        new_cluster_assignment = assign_clusters(data, new_centroids)

        if new_cluster_assignment == cluster_assignment:
            print(f"Converged after {i} iterations.")
            break

        cluster_assignment = new_cluster_assignment
        centroids = new_centroids

    visualize_clusters_with_centroids(data, cluster_assignment, centroids)


def cluster_by_k_means(file_path: str, k: int):
    data = load_clustering_data(file_path)
    centroids = generate_initial_random_centroids(data, k)
    cluster_assignment = assign_clusters(data, centroids)

    visualize_clusters_dynamic(data, cluster_assignment, centroids, update_centroids_and_assignments, 'K-Means')


if __name__ == "__main__":
    bear_attack_path = 'data/BearAttacks/Processed_BearAttacks.csv'
    blob_path = 'data/Blobs/blob_dataset.csv'
    concentric_circles_path = 'data/Blobs/concentric_circles.csv'

    file_path = concentric_circles_path
    k = 3
    cluster_by_k_means(file_path, k)
