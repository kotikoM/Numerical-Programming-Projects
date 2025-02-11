import numpy as np


def stochastic_gradient_descent(X, y, learning_rate=0.01, n_iterations=100):
    m, n = X.shape
    w = np.zeros((n, 1))
    b = 0

    for iteration in range(n_iterations):
        for i in range(m):
            # Randomly select a data point
            rand_index = np.random.randint(0, m)
            xi = X[rand_index:rand_index + 1]
            yi = y[rand_index:rand_index + 1]

            # Compute gradients
            predictions = xi @ w + b
            dw = 2 * xi.T @ (predictions - yi)
            db = 2 * np.sum(predictions - yi)

            # Update parameters
            w -= learning_rate * dw
            b -= learning_rate * db

        # Optionally print progress
        if iteration % 10 == 0:
            cost = (1 / m) * np.sum((X @ w + b - y) ** 2)
            print(f"Iteration {iteration}, Cost: {cost}")

    return w, b


# Example usage
if __name__ == "__main__":
    X = np.array([[1], [2], [3]])
    y = np.array([[2], [4], [6]])
    w, b = stochastic_gradient_descent(X, y)
    print(f"Optimized weights: {w.ravel()}, Optimized bias: {b}")
