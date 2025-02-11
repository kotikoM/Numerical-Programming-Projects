import numpy as np
import matplotlib.pyplot as plt

def chebyshev_nodes(a, b, n):
    nodes = []
    for i in range(1, n + 1):
        node = 0.5 * ((b - a) * np.cos((2 * i - 1) * np.pi / (2 * n)) + (b + a))
        nodes.append(node)
    return nodes


def piecewise_linear_interpolation(nodes, func):
    interpolated_values = []
    for i in range(len(nodes) - 1):
        x0, x1 = nodes[i], nodes[i + 1]
        y0, y1 = func(x0), func(x1)

        # Generate 50 points between each pair for interpolation
        for x in np.linspace(x0, x1, 50):
            y = y0 + (x - x0) * (y1 - y0) / (x1 - x0)
            interpolated_values.append((x, y))

    return interpolated_values


def piecewise_quadratic_interpolation(nodes, func):
    interpolated_values = []
    for i in range(len(nodes) - 1):
        x0, x1 = nodes[i], nodes[i + 1]
        y0, y1 = func(x0), func(x1)

        # Quadratic interpolation: We use the three points (x0, y0), (x1, y1), and an extra middle point.
        mid_x = (x0 + x1) / 2
        mid_y = func(mid_x)

        # Fit a quadratic polynomial to the three points: (x0, y0), (mid_x, mid_y), (x1, y1)
        coefficients = np.polyfit([x0, mid_x, x1], [y0, mid_y, y1], 2)  # Degree 2 polynomial

        # Generate 5 points between each pair for interpolation
        for x in np.linspace(x0, x1, 50):
            y = np.polyval(coefficients, x)
            interpolated_values.append((x, y))

    return interpolated_values


def piecewise_cubic_interpolation(nodes, func):
    interpolated_values = []
    for i in range(len(nodes) - 1):
        x0, x1 = nodes[i], nodes[i + 1]
        y0, y1 = func(x0), func(x1)

        # Cubic interpolation: We use the four points (x0, y0), (x1, y1), and two middle points.
        mid_x1 = x0 + (x1 - x0) / 3
        mid_x2 = x0 + 2 * (x1 - x0) / 3
        mid_y1 = func(mid_x1)
        mid_y2 = func(mid_x2)

        # Fit a cubic polynomial to the four points: (x0, y0), (mid_x1, mid_y1), (mid_x2, mid_y2), (x1, y1)
        coefficients = np.polyfit([x0, mid_x1, mid_x2, x1], [y0, mid_y1, mid_y2, y1], 3)  # Degree 3 polynomial

        # Generate 5 points between each pair for interpolation
        for x in np.linspace(x0, x1, 50):
            y = np.polyval(coefficients, x)
            interpolated_values.append((x, y))

    return interpolated_values


def func(x):
    return 1 / (1 + x ** 2)


def calculate_error(interpolated_values, func):
    # Calculate the mean absolute error (MAE) or RMSE compared to the real function
    errors = [abs(y - func(x)) for x, y in interpolated_values]
    return np.mean(errors)


def plot_interpolations(nodes, linear_values, quadratic_values, cubic_values, real_function, a, b):
    # Extract x and y values from the interpolated points
    x_linear_values, y_linear_values = zip(*linear_values)
    x_quadratic_values, y_quadratic_values = zip(*quadratic_values)
    x_cubic_values, y_cubic_values = zip(*cubic_values)

    # Plot the piecewise linear interpolation curve
    plt.plot(x_linear_values, y_linear_values, label="Piecewise Linear Interpolation", color='b')

    # Plot the piecewise quadratic interpolation curve
    plt.plot(x_quadratic_values, y_quadratic_values, label="Piecewise Quadratic Interpolation", color='m')

    # Plot the piecewise cubic interpolation curve
    plt.plot(x_cubic_values, y_cubic_values, label="Piecewise Cubic Interpolation", color='c')

    # Plot the Chebyshev nodes
    nodes_y = [func(x) for x in nodes]  # Calculate y values for Chebyshev nodes
    plt.scatter(nodes, nodes_y, color='r', label="Chebyshev Nodes")

    # Plot the real function
    x_real = np.linspace(a, b, 400)  # 400 points for the real function plot
    y_real = real_function(x_real)
    plt.plot(x_real, y_real, label="Real Function", color='g', linestyle='--')

    # Labels and title
    plt.xlabel('x')
    plt.ylabel('f(x)')
    plt.title('Interpolation Methods with Chebyshev Nodes')
    plt.legend()

    # Show the plot
    plt.grid(True)
    plt.show()


if __name__ == '__main__':
    a = -5
    b = 5
    n = 10

    # Generate Chebyshev nodes
    nodes = chebyshev_nodes(a, b, n)
    print(f"Chebyshev nodes: {nodes}")

    # Perform piecewise linear interpolation
    linear_interpolated_values = piecewise_linear_interpolation(nodes, func)

    # Perform piecewise quadratic interpolation
    quadratic_interpolated_values = piecewise_quadratic_interpolation(nodes, func)

    # Perform piecewise cubic interpolation
    cubic_interpolated_values = piecewise_cubic_interpolation(nodes, func)

    # Calculate errors
    linear_error = calculate_error(linear_interpolated_values, func)
    quadratic_error = calculate_error(quadratic_interpolated_values, func)
    cubic_error = calculate_error(cubic_interpolated_values, func)

    # Display the errors
    print(f"Linear interpolation error: {linear_error:.5f}")
    print(f"Quadratic interpolation error: {quadratic_error:.5f}")
    print(f"Cubic interpolation error: {cubic_error:.5f}")

    # Plotting
    plot_interpolations(nodes, linear_interpolated_values, quadratic_interpolated_values, cubic_interpolated_values, func, a, b)
