import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def plot_surface():
    # Define the x and y dimensions
    x = np.linspace(0, 30, 30)
    y = np.linspace(0, 30, 30)
    x, y = np.meshgrid(x, y)

    # Calculate z using the specific sine function
    z = 3 * np.sin(2 * np.pi * x / 30) * np.sin(2 * np.pi * y / 30)

    # Create a figure and a 3D axis
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    # Plot the surface
    surface = ax.plot_surface(x, y, z, cmap="viridis")

    # Add a color bar which maps values to colors
    # fig.colorbar(surface, ax=ax, shrink=0.5, aspect=5)

    # Set labels
    ax.set_xlabel("X ")
    ax.set_ylabel("Y ")
    ax.set_zlabel("Z")
    ax.set_xlim(0, 30)
    ax.set_ylim(0, 30)
    ax.set_zlim(-5, 5)

    # Show the plot
    plt.show()


def plot_surf_with_atoms(points):
    # Example periodic surface: Sine wave pattern
    x = np.linspace(0, 30, 30)
    y = np.linspace(0, 30, 30)
    x, y = np.meshgrid(x, y)

    # Calculate z using the specific sine function
    z = 3 * np.sin(2 * np.pi * x / 30) * np.sin(2 * np.pi * y / 30)

    # Create a figure and a 3D axis
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    # Plot the surface
    ax.plot_surface(x, y, z, cmap="viridis", alpha=0.5)

    # Plot the points as 'spheres'
    ax.scatter(
        points[:, 0],
        points[:, 1],
        points[:, 2],
        color="yellow",
        s=100,
        alpha=1,
        edgecolors="black",
    )  # s is the size of the point

    # Set labels
    ax.set_xlabel("X  ")
    ax.set_ylabel("Y  ")
    ax.set_zlabel("Z")

    # Show the plot
    plt.show()
