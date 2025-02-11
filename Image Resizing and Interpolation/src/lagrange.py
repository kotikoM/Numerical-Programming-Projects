import numpy as np
import matplotlib.pyplot as plt


def lagrange(points):
    def lagrange_basis(i, x):
        xi, _ = points[i]
        basis = 1
        for j, (xj, _) in enumerate(points):
            if i != j:
                basis *= (x - xj) / (xi - xj)
        return basis

    def interpolation_function(x):
        result = 0
        for i, (_, yi) in enumerate(points):
            result += yi * lagrange_basis(i, x)
        return result

    return interpolation_function


if __name__ == '__main__':
    points = [(1, 2), (2, 3), (3, 5), (4, 4)]

    interpolation = lagrange(points)

    x_values = np.linspace(min([x for x, _ in points]), max([x for x, _ in points]), 500)
    y_values = [interpolation(x) for x in x_values]

    original_x = [x for x, _ in points]
    original_y = [y for _, y in points]

    plt.scatter(original_x, original_y, color='red', label='Data points')
    plt.plot(x_values, y_values, label='Lagrange Interpolation', color='blue')

    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Lagrange Interpolation')
    plt.legend()

    plt.grid(True)
    plt.show()
