import numpy as np


# Define the function f(x1, x2, x3, x4)
def f(x1, x2, x3, x4):
    return x1 ** 2 + 2 * x2 ** 2 + 3 * x3 ** 2 + 3 * x4 ** 2 - x1 * x2 + 2 * x2 * x3 + 3 * x3 * x4 + 5


# Define the gradient of f(x1, x2, x3, x4)
def grad_f(x1, x2, x3, x4):
    dfdx1 = 2 * x1 - x2
    dfdx2 = 4 * x2 - x1 + 2 * x3
    dfdx3 = 6 * x3 + 2 * x2 + 3 * x4
    dfdx4 = 6 * x4 + 3 * x3
    return np.array([dfdx1, dfdx2, dfdx3, dfdx4])


# Initialize parameters and hyperparameters
learning_rate = 0.02
starting_point = np.array([1, -1, 0.5, 2])
batch_size = 2
data_points = [
    np.array([1, -1, 0.5, 2]),
    np.array([0.5, -1.5, 1, 0.5]),
    np.array([-1, 1, -0.5, 1.5])
]

# Number of iterations for the gradient descent
iterations = 100

# Mini-batch gradient descent loop
params = starting_point
for i in range(iterations):
    # Shuffle the data points (optional, but recommended)
    np.random.shuffle(data_points)

    # Take a mini-batch of size 2
    mini_batch = data_points[:batch_size]

    # Compute the gradient for the mini-batch
    gradients = np.zeros(4)
    for point in mini_batch:
        gradients += grad_f(*point)

    # Compute the average gradient
    avg_gradient = gradients / batch_size

    # Update the parameters using the learning rate
    params = params - learning_rate * avg_gradient

    # Print the current parameters after each iteration
    if i % 10 == 0:  # Print every 10 iterations
        print(f"Iteration {i}, Parameters: {params}")

print(f"Final parameters: {params}")
