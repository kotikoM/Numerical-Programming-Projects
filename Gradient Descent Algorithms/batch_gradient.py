import numpy as np


def batch_gradient_descent(X, y, learning_rate=0.01, n_iterations=10000):
    # Number of samples (m) and features (n)
    m, n = X.shape

    # Initialize weights (n x 1) and bias term
    w = np.zeros((n, 1))
    b = 0

    for iteration in range(n_iterations):
        # Compute predictions: h(x) = X @ w + b
        predictions = X @ w + b

        # Compute gradients
        dw = (2 / m) * (X.T  @ (predictions - y))
        db = (2 / m) * np.sum(predictions - y)

        # Update parameters
        w -= learning_rate * dw
        b -= learning_rate * db

        # Optionally print the cost every 100 iterations
        if iteration % 100 == 0:
            cost = (1 / m) * np.sum((predictions - y) ** 2)
            print(f"Iteration {iteration}, Cost: {cost}")

    return w, b


# Example usage
if __name__ == "__main__":
    # Example data: X (features) and y (targets)
    X = np.array([[1], [2], [3]])
    y = np.array([[2], [7], [3]])

    # Perform batch gradient descent
    learning_rate = 0.01
    n_iterations = 1000
    w, b = batch_gradient_descent(X, y, learning_rate, n_iterations)

    # Output the results
    print(f"Optimized weights: {w.ravel()}")
    print(f"Optimized bias: {b}")
