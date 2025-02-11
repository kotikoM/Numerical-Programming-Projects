import math

import numpy as np
import matplotlib.pyplot as plt
from finite_differences import f, df, forward_difference, backward_difference, central_difference, richardsons_extrapolation


def visualize_differences(x0, h):
    # Define x range for visualization
    x_values = np.linspace(x0 - 2, x0 + 2, 400)
    f_values = [f(x_value) for x_value in x_values]

    # Exact slope (derivative)
    exact_slope = df(x0)

    # Finite difference approximations
    forward_slope = forward_difference(x0, h)
    backward_slope = backward_difference(x0, h)
    central_slope = central_difference(x0, h)
    richardson_slope = richardsons_extrapolation(x0, h)

    # Tangent line for exact derivative
    tangent_line = exact_slope * (x_values - x0) + f(x0)

    # Approximated lines for finite difference methods
    forward_line = forward_slope * (x_values - x0) + f(x0)
    backward_line = backward_slope * (x_values - x0) + f(x0)
    central_line = central_slope * (x_values - x0) + f(x0)
    richardson_line = richardson_slope * (x_values - x0) + f(x0)

    # Plot the function f(x)
    plt.plot(x_values, f_values, label="f(x)", color="blue")

    # Plot tangent line for exact derivative
    plt.plot(x_values, tangent_line, label=f"Exact Derivative at x={x0}", color="green", linestyle='--')

    # Plot finite difference approximations
    plt.plot(x_values, forward_line, label=f"Forward Difference (h={h})", color="red", linestyle=':')
    plt.plot(x_values, backward_line, label=f"Backward Difference (h={h})", color="orange", linestyle=':')
    plt.plot(x_values, central_line, label=f"Central Difference (h={h})", color="purple", linestyle=':')
    plt.plot(x_values, richardson_line, label=f"Richardson's Extrapolation (h={h})", color="cyan", linestyle=':')

    # Vertical line at x0
    plt.axvline(x=x0, color="black", linestyle="--", label=f"x = {x0}")

    # Add legend, title, and labels
    plt.legend()
    plt.title(f"Finite Difference Visualization at x = {x0}, h = {h}")
    plt.xlabel("x")
    plt.ylabel("f(x) and Derivatives")

    # Show the plot
    plt.show()


def visualize_derivative_methods(x0, h):
    # Define x range for visualization
    x_values = np.linspace(x0 - 2, x0 + 2, 400)

    # Exact derivative values
    exact_derivatives = [df(x_value) for x_value in x_values]

    # Finite difference approximations for all x values
    forward_approx = [forward_difference(x_value, h) for x_value in x_values]
    backward_approx = [backward_difference(x_value, h) for x_value in x_values]
    central_approx = [central_difference(x_value, h) for x_value in x_values]
    richardson_approx = [richardsons_extrapolation(x_value, h) for x_value in x_values]

    # Plot the exact derivative
    plt.plot(x_values, exact_derivatives, label="Exact Derivative", color="green", linestyle='-')

    # Plot the forward difference approximation
    plt.plot(x_values, forward_approx, label=f"Forward Difference (h={h})", color="red", linestyle=':')

    # Plot the backward difference approximation
    plt.plot(x_values, backward_approx, label=f"Backward Difference (h={h})", color="orange", linestyle=':')

    # Plot the central difference approximation
    plt.plot(x_values, central_approx, label=f"Central Difference (h={h})", color="purple", linestyle=':')

    # Plot the Richardson's extrapolation approximation
    plt.plot(x_values, richardson_approx, label=f"Richardson's Extrapolation (h={h})", color="cyan", linestyle=':')

    # Add vertical line at x0 for reference
    plt.axvline(x=x0, color="black", linestyle="--", label=f"x = {x0}")

    # Add legend, title, and labels
    plt.legend()
    plt.title(f"Derivative Approximations at x = {x0}, h = {h}")
    plt.xlabel("x")
    plt.ylabel("Derivative f'(x)")

    # Show the plot
    plt.show()


def visualize_error_behavior(x0):
    # Define a range of small h values (discretization steps)
    h_values = np.logspace(-10, 0, 100)  # h values from 10^-5 to 1

    # Calculate the exact derivative at x0
    exact_derivative = df(x0)

    # Initialize lists to store errors for each method
    forward_errors = []
    backward_errors = []
    central_errors = []
    richardson_errors = []

    # Loop over h values and compute errors
    for h in h_values:
        forward_approx = forward_difference(x0, h)
        backward_approx = backward_difference(x0, h)
        central_approx = central_difference(x0, h)
        richardson_approx = richardsons_extrapolation(x0, h)

        # Compute absolute errors for each method
        forward_errors.append(abs(forward_approx - exact_derivative))
        backward_errors.append(abs(backward_approx - exact_derivative))
        central_errors.append(abs(central_approx - exact_derivative))
        richardson_errors.append(abs(richardson_approx - exact_derivative))

    # Plot the errors as a function of h
    plt.loglog(h_values, forward_errors, label="Forward Difference Error", color="red")
    plt.loglog(h_values, backward_errors, label="Backward Difference Error", color="orange")
    plt.loglog(h_values, central_errors, label="Central Difference Error", color="green")
    plt.loglog(h_values, richardson_errors, label="Richardson's Extrapolation Error", color="blue")

    # Plot settings
    plt.xlabel("Discretization Step (h)")
    plt.ylabel("Error |Approximation - Exact|")
    plt.title(f"Error Behavior as h -> 0 at x = {x0}")
    plt.legend()
    plt.grid(True, which="both", linestyle="--")

    # Show the plot
    plt.show()

if __name__ == "__main__":
    x0 = math.pi / 4
    h = 0.1
    visualize_differences(x0, h)
    # visualize_derivative_methods(x0, h)
    # visualize_error_behavior(x0)
