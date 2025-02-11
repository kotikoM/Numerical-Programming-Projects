import numpy as np
import matplotlib.pyplot as plt

def dX(t, X, r, K):
    # dX/dt = r * X * (1 - X/K)
    return r * X * (1 - X / K)


def dY(t, Y, alpha, C):
    # dY/dt = alpha * Y * (1 - Y/C)
    return alpha * Y * (1 - Y / C)


# DIRK(2,1) method
def dirk_method(f, t_span, y0, r, K, alpha, C, dt):
    t0, tf = t_span
    n_steps = int((tf - t0) / dt)
    t_values = np.linspace(t0, tf, n_steps)
    y_values = np.zeros((n_steps, len(y0)))

    # Initial condition
    y_values[0] = y0

    # Butcher tableau for DIRK(2,1): a11=1, b1=1/2, b2=1/2
    a11 = 1
    b1 = 0.5
    b2 = 0.5

    for i in range(1, n_steps):
        t = t_values[i - 1]
        y = y_values[i - 1]

        # First stage
        k1 = f(t, y, r, K, alpha, C)

        # Second stage
        k2 = f(t + dt, y + dt * a11 * k1, r, K, alpha, C)

        # Update solution
        y_values[i] = y + dt * (b1 * k1 + b2 * k2)

    return t_values, y_values


def system_of_odes(t, y, r, K, alpha, C):
    X, Y = y
    dxdt = dX(t, X, r, K)
    dydt = dY(t, Y, alpha, C)
    return np.array([dxdt, dydt])


if __name__ == '__main__':
    # Parameters
    r = 0.5
    K = 1000
    a = 0.1
    C = 1000

    # Initial conditions: initial values for X and Y
    X0 = 100  # initial investment value
    Y0 = 0.1  # initial rate of return

    # Time span for solving the ODE
    t_span = (0, 50)  # from t=0 to t=50
    dt = 0.1  # time step size

    # Solve the system using DIRK method
    t_values, y_values = dirk_method(system_of_odes, t_span, [X0, Y0], r, K, a, C, dt)

    # Extract the solution
    X_values = y_values[:, 0]
    Y_values = y_values[:, 1]

    # Plot the results
    plt.figure(figsize=(10, 6))

    plt.subplot(2, 1, 1)
    plt.plot(t_values, X_values, label="Investment Value (X)")
    plt.xlabel('Time')
    plt.ylabel('Investment Value')
    plt.legend()

    plt.subplot(2, 1, 2)
    plt.plot(t_values, Y_values, label="Return Rate (Y)", color='orange')
    plt.xlabel('Time')
    plt.ylabel('Return Rate')
    plt.legend()

    plt.show()
