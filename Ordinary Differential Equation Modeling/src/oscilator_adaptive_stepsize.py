import numpy as np
import matplotlib.pyplot as plt


# System of ODEs
def system(t, y, omega, beta):
    x, v = y
    dxdt = v
    dvdt = -omega ** 2 * x - beta * v
    return np.array([dxdt, dvdt])


# Runge-Kutta 4th and 5th order method
def rk45_step(f, t, y, h, omega, beta):
    # Compute the Runge-Kutta intermediate steps
    k1 = h * f(t, y, omega, beta)
    k2 = h * f(t + h / 4, y + k1 / 4, omega, beta)
    k3 = h * f(t + 3 * h / 8, y + 3 * k1 / 32 + 9 * k2 / 32, omega, beta)
    k4 = h * f(t + 12 * h / 13, y + 1932 * k1 / 2197 - 7200 * k2 / 2197 + 7296 * k3 / 2197, omega, beta)
    k5 = h * f(t + h, y + 439 * k1 / 216 - 8 * k2 + 3680 * k3 / 513 - 845 * k4 / 4104, omega, beta)
    k6 = h * f(t + h / 2, y - 8 * k1 / 27 + 2 * k2 - 3544 * k3 / 2565 + 1859 * k4 / 4104 - 11 * k5 / 40, omega, beta)

    # Calculate the RK4 and RK5 results
    y4 = y + 25 * k1 / 216 + 1408 * k3 / 2565 + 2197 * k4 / 4104 - k5 / 5
    y5 = y + 16 * k1 / 135 + 6656 * k3 / 12825 + 28561 * k4 / 56430 - 9 * k5 / 50 + 2 * k6 / 55

    # Estimate error
    error = np.linalg.norm(y5 - y4)

    return y5, error


# Adaptive step size control
def solve_adaptive_rk45(f, t_span, y0, omega, beta, h_init=0.1, tol=1e-6):
    t0, tf = t_span
    y = np.array(y0)
    t = t0
    h = h_init
    solution = []

    while t < tf:
        # Take a step
        y_next, error = rk45_step(f, t, y, h, omega, beta)

        # Estimate the error and adjust step size
        if error < tol:
            # Accept the step
            t += h
            y = y_next
            solution.append([t, *y])  # Store time and values of x, v

        # Compute the next step size
        h_new = h * (tol / error) ** 0.25
        h = min(h_new, tf - t)  # Prevent overshooting the final time

    return np.array(solution)

if __name__ == '__main__':
    # Parameters
    omega = 2.0  # Example frequency
    beta = 0.5  # Damping factor
    x0 = 1.0  # Initial displacement
    v0 = 0.0  # Initial velocity
    y0 = [x0, v0]

    # Time span
    t_span = (0, 10)

    # Solve the system using adaptive RK45
    solution = solve_adaptive_rk45(system, t_span, y0, omega, beta)

    # Extract the results
    t_values = solution[:, 0]
    x_values = solution[:, 1]
    v_values = solution[:, 2]

    # Plot the results
    plt.figure(figsize=(10, 6))

    plt.subplot(2, 1, 1)
    plt.plot(t_values, x_values, label='Displacement (x)')
    plt.xlabel('Time (t)')
    plt.ylabel('Displacement (x)')
    plt.grid(True)

    plt.subplot(2, 1, 2)
    plt.plot(t_values, v_values, label='Velocity (v)', color='r')
    plt.xlabel('Time (t)')
    plt.ylabel('Velocity (v)')
    plt.grid(True)

    plt.tight_layout()
    plt.show()
