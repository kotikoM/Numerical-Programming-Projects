import matplotlib.pyplot as plt


def plot_results(t, solution):
    # Unpack the solution into individual variables
    x = solution[:, 0]  # Rabbits
    y = solution[:, 1]  # Foxes
    z = solution[:, 2]  # Wolves
    w = solution[:, 3]  # Resources

    # Create a single plot for all the variables
    plt.figure(figsize=(10, 6))

    # Plot rabbits (x), foxes (y), wolves (z), and resources (w) over time
    plt.plot(t, x, label='Rabbits (x)', color='tab:blue')
    plt.plot(t, y, label='Foxes (y)', color='tab:orange')
    plt.plot(t, z, label='Wolves (z)', color='tab:green')
    plt.plot(t, w, label='Resources (w)', color='tab:red')

    # Add titles and labels
    plt.title('Predator-Prey Dynamics Over Time')
    plt.xlabel('Time')
    plt.ylabel('Population')

    # Add a legend to differentiate the curves
    plt.legend()

    # Display grid for better readability
    plt.grid(True)

    # Show the plot
    plt.show()

