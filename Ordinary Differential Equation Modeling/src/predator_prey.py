import numpy as np
from src.plot import plot_results

# Derivative function for predator-prey model
def predator_prey_derivatives(state, a, b, c, d, e, f, g, h, i, j, k):
    x, y, z, w = state  # Unpack state variables

    # Derivatives
    dx_dt = a * x - b * x * y - c * x * w
    dy_dt = d * x * y - e * y - f * y * z
    dz_dt = g * y * z - h * z
    dw_dt = i - j * w - k * x * w

    return [dx_dt, dy_dt, dz_dt, dw_dt]


# Function to run the simulation
def simulate(initial_state, params, t_span, t_points):
    # Initialize time step size and time points
    dt = (t_span[1] - t_span[0]) / (t_points - 1)  # Step size
    t = np.linspace(t_span[0], t_span[1], t_points)  # Time grid
    state = np.array(initial_state)  # Convert initial state to numpy array

    # Initialize solution array to store results (time x state variables)
    num_variables = len(initial_state)  # Dynamic size of state variables
    solution = np.zeros((t_points, num_variables))  # Solution array

    # Store initial state
    solution[0] = state

    # Perform Euler integration
    for i in range(1, t_points):
        # Calculate derivatives
        derivatives = predator_prey_derivatives(state, *params)

        # Update state using Euler's method (state[i] = state[i-1] + dt * derivatives)
        for j in range(len(state)):
            state[j] = state[j] + dt * derivatives[j]  # Update all state variables

        # Store the new state in the solution array
        solution[i] = state
    return t, solution


if __name__ == '__main__':
    # Example inputs for simulation
    initial_state = [100, 10, 5, 80]  # [rabbits, foxes, wolves, resources]
    params = (
        1.2,   # a Rabbit reproduction rate
        0.1,   # b Predation rate of rabbits by foxes
        0.05,  # c Vegetation dependency rate for rabbits
        0.03,  # d Growth rate of foxes due to predation on rabbits
        0.15,  # e Natural death rate of foxes
        0.05,  # f Predation rate of foxes by wolves
        0.03,  # g Growth rate of wolves due to predation on foxes
        0.15,  # h Natural death rate of wolves
        0.5,   # e Resource replenishment rate
        0.1,   # i Natural decay rate of the resource
        0.1    # j Rate of resource depletion by rabbits
    )
    t_span = (0, 5) # Time span for simulation
    t_points = 100  # Number of time points for the simulation

    # Run the simulation
    t, solution = simulate(initial_state, params, t_span, t_points)
    print(solution)
    plot_results(t, solution)
