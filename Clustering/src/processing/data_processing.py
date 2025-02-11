import pandas as pd
import os
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs, make_moons
from sklearn.datasets import make_circles
import numpy as np

def process_bear_attack_data(input_file: str = 'data/BearAttacks/BearAttacks.csv'):
    df = pd.read_csv(input_file)
    df.columns = df.columns.str.strip()
    df["age"] = pd.to_numeric(df["age"], errors='coerce')

    age_attack_map = df.groupby('age').size().to_dict()
    processed_df = pd.DataFrame(list(age_attack_map.items()), columns=['age', 'Number of Attacks'])

    input_dir, input_filename = os.path.split(input_file)
    processed_filename = f"Processed_{input_filename}"
    output_file = os.path.join(input_dir, processed_filename)
    processed_df.to_csv(output_file, index=False)

    print(f"Processed file saved at: {output_file}")

    return age_attack_map


def process_flight_delay_data(input_file: str = 'data/DelayedFlights/DelayedFlights.csv'):
    df = pd.read_csv(input_file)
    df.columns = df.columns.str.strip()

    df['DepDelay'] = pd.to_numeric(df['DepDelay'], errors='coerce')
    df['ArrDelay'] = pd.to_numeric(df['ArrDelay'], errors='coerce')

    transformed_data = {
        'Departure Delay': df['DepDelay'].dropna(),
        'Arrival Delay': df['ArrDelay'].dropna()
    }

    processed_df = pd.DataFrame(transformed_data)

    processed_df.dropna(inplace=True)

    input_dir, input_filename = os.path.split(input_file)
    processed_filename = f"Processed_{input_filename}"
    output_file = os.path.join(input_dir, processed_filename)
    processed_df.to_csv(output_file, index=False)

    print(f"Processed file saved at: {output_file}")

    return processed_df


def generate_and_save_blob_data(n_samples: int = 300, centers: int = 3, cluster_std: float = 1.0,
                                random_state: int = 42, filename: str = 'blob_dataset'):
    X, y = make_blobs(n_samples=n_samples, centers=centers, cluster_std=cluster_std, random_state=random_state)

    df = pd.DataFrame(X, columns=['Feature_1', 'Feature_2'])

    output_dir = "data/Blobs"
    os.makedirs(output_dir, exist_ok=True)

    output_path = os.path.join(output_dir, f"{filename}.csv")
    df.to_csv(output_path, index=False)

    plt.scatter(X[:, 0], X[:, 1], c=y, s=50, cmap='viridis')
    plt.title('Synthetic Blob-like Dataset with Clear Clusters')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.show()

    print(f"Data saved to {output_path}")


def generate_concentric_circles(n_samples=700, factor=0.5, noise=0.05, filename="concentrica_circles"):
    x, y = make_circles(n_samples=n_samples, factor=factor, noise=noise)

    df = pd.DataFrame(x, columns=['Feature_1', 'Feature_2'])

    output_dir = "data/Blobs"
    os.makedirs(output_dir, exist_ok=True)

    output_path = os.path.join(output_dir, f"{filename}.csv")
    df.to_csv(output_path, index=False)

    plt.scatter(x[:, 0], x[:, 1], c=y, s=50, cmap='viridis')
    plt.title('Concentric Circles')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.show()

    return x, y


def generate_scattered_data(n_samples=700, filename="scattered_data"):
    # Generate uniformly distributed random points
    np.random.seed(42)  # For reproducibility
    data = np.random.uniform(low=-10, high=10, size=(n_samples, 2))  # 2D points in the range [-10, 10]

    # Create a DataFrame
    df = pd.DataFrame(data, columns=['Feature_1', 'Feature_2'])

    # Save to CSV
    output_dir = "data/Blobs"
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"{filename}.csv")
    df.to_csv(output_path, index=False)

    # Visualize the dataset
    plt.figure(figsize=(10, 6))
    plt.scatter(df['Feature_1'], df['Feature_2'], alpha=0.6, edgecolors='k', s=30)
    plt.title('Completely Scattered Data with No Visual Clusters')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.grid(color='gray', linestyle='--', linewidth=0.5, alpha=0.7)
    plt.xlim(-10, 10)
    plt.ylim(-10, 10)
    plt.show()

    return df


def generate_three_moons_dataset(filename="moons"):
    # Generate the first moon using sklearn's make_moons
    X1, _ = make_moons(n_samples=400, noise=0.05)
    # Combine the three moons
    X = np.vstack((X1))

    # Convert to DataFrame for clustering purposes
    df = pd.DataFrame(X, columns=['Feature_1', 'Feature_2'])

    # Create output directory if it doesn't exist
    output_dir = "data/Blobs"
    os.makedirs(output_dir, exist_ok=True)

    # Save the dataset to a CSV file
    output_path = os.path.join(output_dir, f"{filename}.csv")
    df.to_csv(output_path, index=False)

    # Plot the dataset
    plt.scatter(X[:, 0], X[:, 1], s=10, color='b')
    plt.title('Separated Three Moons Dataset')
    plt.show()

    return df

if __name__ == "__main__":
    generate_three_moons_dataset()