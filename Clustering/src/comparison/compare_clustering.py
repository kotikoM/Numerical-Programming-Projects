from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from clustering.DBSCAN import cluster_by_dbscan
from clustering.kmedoids import cluster_by_k_medoids, load_clustering_data


def visualize_clusters(data: pd.DataFrame, cluster_assignment: list, title: str):
    unique_clusters = sorted(set(cluster_assignment))
    num_clusters = len(unique_clusters)

    colors = plt.get_cmap('plasma')(np.linspace(0, 1, num_clusters))

    first_col_name = data.columns[0]
    second_col_name = data.columns[1]

    plt.figure(figsize=(12, 6))

    for cluster in unique_clusters:
        if cluster == -1:
            color = 'k'
            label = 'Noise'
        else:
            color_idx = unique_clusters.index(cluster)
            color = colors[color_idx]
            label = f'Cluster {cluster}'

        points = data[np.array(cluster_assignment) == cluster]

        plt.scatter(points[first_col_name], points[second_col_name],
                    color=color, label=label)

    plt.title(f'{title} Clustering Visualization', fontsize=16)
    plt.xlabel(first_col_name, fontsize=14)
    plt.ylabel(second_col_name, fontsize=14)
    plt.legend()
    plt.grid(color='gray', linestyle='--', linewidth=0.5, alpha=0.7)

    plt.show(block=False)


def evaluate_clusters(data, cluster_labels):
    db_index = round(davies_bouldin_score(data, cluster_labels), 3)
    ch_score = round(calinski_harabasz_score(data, cluster_labels), 3)
    return db_index, ch_score


if __name__ == '__main__':
    # Choose dataset file
    file_path = 'data/Blobs/moons.csv'

    # K-Medoids Clustering
    n_clusters = 2
    max_iter = 100
    medoids, kmedoids_labels = cluster_by_k_medoids(file_path, n_clusters, max_iter)

    # DBSCAN Clustering
    eps = 0.2
    min_samples = 5
    dbscan_labels = cluster_by_dbscan(file_path, eps, min_samples)

    # Load data
    data = load_clustering_data(file_path)

    # Visualize clusters
    visualize_clusters(data, kmedoids_labels, "K-Medoids")
    visualize_clusters(data, dbscan_labels, "DBSCAN")

    # Evaluate K-Medoids Clusters
    db_kmedoids, ch_kmedoids = evaluate_clusters(data, kmedoids_labels)
    print(f"Davies-Bouldin Index (K-Medoids): {db_kmedoids}")
    print(f"Calinski-Harabasz Score (K-Medoids): {ch_kmedoids}")
    print("\n")

    # Evaluate DBSCAN Clusters
    db_dbscan, ch_dbscan = evaluate_clusters(data, dbscan_labels)
    print(f"Davies-Bouldin Index (DBSCAN): {db_dbscan}")
    print(f"Calinski-Harabasz Score (DBSCAN): {ch_dbscan}")

    plt.show()
