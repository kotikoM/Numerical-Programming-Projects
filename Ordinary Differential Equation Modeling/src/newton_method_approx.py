import numpy as np
import matplotlib.pyplot as plt


def f(x):
    return x ** 5 - 2 * x ** 4 + 3 * x ** 3 - 4 * x ** 2 + 5 * x - 6


def f_prime(x):
    return 5 * x ** 4 - 8 * x ** 3 + 9 * x ** 2 - 8 * x + 5


def newtons_method(func, func_prime, x0, tol=1e-6, max_iter=100):
    approximations = [x0]
    for _ in range(max_iter):
        f_x = func(x0)
        f_prime_x = func_prime(x0)
        if abs(f_prime_x) < 1e-10:
            print("Derivative too small, stopping iteration.")
            break

        x1 = x0 - f_x / f_prime_x
        approximations.append(x1)

        if abs(x1 - x0) < tol:  # Convergence check
            break

        x0 = x1

    return x1, approximations


x0 = 1.5
tolerance = 1e-6

root, approximations = newtons_method(f, f_prime, x0, tol=tolerance)

print(f"Root: {root}")
print(f"Approximations: {approximations}")

x_vals = np.linspace(0, 2, 500)
y_vals = f(x_vals)

plt.figure(figsize=(10, 6))
plt.plot(x_vals, y_vals, label="f(x)", color="blue")
plt.axhline(0, color="black", linewidth=0.8, linestyle="--")

for i, x_approx in enumerate(approximations):
    plt.scatter(x_approx, f(x_approx), color="red", label=f"x{i}" if i == 0 else None)

plt.scatter(root, f(root), color="green", label="Root", zorder=5)
plt.title("Newton's Method Approximations")
plt.xlabel("x")
plt.ylabel("f(x)")
plt.legend()
plt.grid()
plt.show()
