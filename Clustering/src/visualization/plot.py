import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


def visualize_processed_data(processed_file: str):
    dataset_name = processed_file.split('/')[-2]
    processed_df = pd.read_csv(processed_file)

    if processed_df.empty:
        print("The processed data file is empty.")
        return

    age_column = processed_df.columns[0]
    attack_count_column = processed_df.columns[1]

    plt.figure(figsize=(12, 6))
    plt.scatter(
        processed_df[age_column],
        processed_df[attack_count_column],
        color='red',
        alpha=0.6,
        edgecolors='black',
        s=30,
        marker='o'
    )

    plt.title(f'Bear Attacks by Age - {dataset_name}', fontsize=16, fontweight='bold')
    plt.xlabel('Age', fontsize=14)
    plt.ylabel('Number of Attacks', fontsize=14, labelpad=10)

    plt.xlim(processed_df[age_column].min() - 5, processed_df[age_column].max() + 5)
    plt.ylim(processed_df[attack_count_column].min() - 1, processed_df[attack_count_column].max() + 1)

    x_ticks = list(range(int(processed_df[age_column].min()) - 5,
                         int(processed_df[age_column].max()) + 5 + 1, 5))
    plt.xticks(x_ticks)

    plt.grid(color='gray', linestyle='--', linewidth=0.5, alpha=0.7)

    plt.show()


def visualize_clusters_with_centroids(data: pd.DataFrame, cluster_assignment: list, centroids: list):
    num_clusters = len(centroids)
    colors = plt.colormaps['plasma'](np.linspace(0, 1, num_clusters))

    first_col_name = data.columns[0]
    second_col_name = data.columns[1]

    plt.figure(figsize=(12, 6))

    for cluster in range(num_clusters):
        points = data[np.array(cluster_assignment) == cluster]
        cluster_color = colors[cluster]

        plt.scatter(points[first_col_name], points[second_col_name],
                    color=cluster_color,
                    label=f'Cluster {cluster}')

        plt.scatter(centroids[cluster][0], centroids[cluster][1],
                    color=cluster_color,
                    s=200, marker='s', edgecolor='black', label=f'Centroid {cluster}')

    plt.title('Clustering Visualization', fontsize=16)
    plt.xlabel(first_col_name, fontsize=14)
    plt.ylabel(second_col_name, fontsize=14)

    plt.legend()
    plt.grid(color='gray', linestyle='--', linewidth=0.5, alpha=0.7)

    plt.show()


def visualize_clusters(data: pd.DataFrame, cluster_assignment: list):
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

    plt.title('DBSCAN Clustering Visualization', fontsize=16)
    plt.xlabel(first_col_name, fontsize=14)
    plt.ylabel(second_col_name, fontsize=14)
    plt.legend()
    plt.grid(color='gray', linestyle='--', linewidth=0.5, alpha=0.7)
    plt.show()


def visualize_clusters_dynamic(data: pd.DataFrame, cluster_assignment: list, centroids: list, update_fn, title: str):
    num_clusters = len(centroids)
    colors = plt.colormaps['plasma'](np.linspace(0, 1, num_clusters))
    click_counter = [0]

    first_col_name = data.columns[0]
    second_col_name = data.columns[1]

    fig, ax = plt.subplots(figsize=(12, 6))

    def on_click(event):
        nonlocal centroids, cluster_assignment
        centroids, cluster_assignment = update_fn(data, centroids)
        ax.clear()
        click_counter[0] += 1

        for cluster in range(num_clusters):
            points = data[np.array(cluster_assignment) == cluster]
            cluster_color = colors[cluster]

            ax.scatter(points[first_col_name], points[second_col_name], color=cluster_color, label=f'Cluster {cluster}')
            ax.scatter(centroids[cluster][0], centroids[cluster][1], color=cluster_color, s=200, marker='s',
                       edgecolor='black', label=f'Centroid {cluster}')

        ax.set_title(f'{title} Clustering - Click to Update (Clicks: {click_counter[0]})', fontsize=16)
        ax.set_xlabel(first_col_name, fontsize=14)
        ax.set_ylabel(second_col_name, fontsize=14)
        ax.legend()
        ax.grid(color='gray', linestyle='--', linewidth=0.5, alpha=0.7)
        plt.draw()

    on_click(None)
    fig.canvas.mpl_connect('button_press_event', on_click)
    plt.show()
