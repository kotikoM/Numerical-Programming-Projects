import numpy as np
import matplotlib.pyplot as plt


# Define the differential equations
def dK_dt(K, L, s, A, B, delta):
    return s * (A * K + B * L) - delta * K


def dL_dt(L, n):
    return n * L


# Euler's Method for solving the system of ODEs
def euler_method(K0, L0, s, A, B, delta, n, t0, tf, h):
    # Number of steps
    steps = int((tf - t0) / h)

    # Arrays to store the values of K, L, and time
    K = np.zeros(steps + 1)
    L = np.zeros(steps + 1)
    t = np.linspace(t0, tf, steps + 1)

    # Initial conditions
    K[0] = K0
    L[0] = L0

    # Euler's method loop
    for i in range(steps):
        # Compute derivatives
        f_K = dK_dt(K[i], L[i], s, A, B, delta)
        f_L = dL_dt(L[i], n)

        # Update values using Euler's method
        K[i + 1] = K[i] + h * f_K
        L[i + 1] = L[i] + h * f_L

    return t, K, L


# Main function to run the simulation
if __name__ == '__main__':
    # Parameters
    K0 = 1000  # initial capital
    L0 = 500  # initial labor force
    s = 0.2  # savings rate
    A = 1.5  # productivity coefficient for capital
    B = 0.5  # productivity coefficient for labor
    delta = 0.05  # depreciation rate of capital
    n = 0.02  # labor growth rate

    # Time span
    t0 = 0  # initial time
    tf = 100  # final time
    h = 0.1  # step size

    # Solve using Euler's method
    t, K, L = euler_method(K0, L0, s, A, B, delta, n, t0, tf, h)

    # Plot the results
    plt.figure(figsize=(10, 6))

    plt.subplot(2, 1, 1)
    plt.plot(t, K, label="Capital (K)", color='b')
    plt.xlabel('Time')
    plt.ylabel('Capital (K)')
    plt.legend()

    plt.subplot(2, 1, 2)
    plt.plot(t, L, label="Labor (L)", color='g')
    plt.xlabel('Time')
    plt.ylabel('Labor (L)')
    plt.legend()

    plt.show()
