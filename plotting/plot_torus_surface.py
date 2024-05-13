import numpy as np
import matplotlib.pyplot as plt

# Parameters for the torus
R = 10  # Major radius
r = 5  # Minor radius
center = (15, 15, 15)


def plot_surface():
    # Define the u, v dimensions
    u = np.linspace(0, 2 * np.pi, 100)
    v = np.linspace(0, 2 * np.pi, 100)
    u, v = np.meshgrid(u, v)

    # Calculate the coordinates of the torus
    x = center[0] + (R + r * np.cos(v)) * np.cos(u)
    y = center[1] + (R + r * np.cos(v)) * np.sin(u)
    z = center[2] + r * np.sin(v)

    # Plot the surface of the torus with a color gradient and transparency
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    surface = ax.plot_surface(x, y, z, cmap="viridis")  # , edgecolor="none", alpha=0.6)

    # Add a color bar to enhance visualization
    # fig.colorbar(surface, ax=ax, shrink=0.5, aspect=5)

    # Set labels and title
    ax.set_xlabel("X ")
    ax.set_ylabel("Y ")
    ax.set_zlabel("Z ")
    ax.set_xlim(0, 30)
    ax.set_ylim(0, 30)
    ax.set_zlim(0, 30)

    # ax.set_title('Colored Torus with Transparency')

    # Show the plot
    plt.show()


def plot_surf_with_atoms(points):
    # Define the u, v dimensions
    u = np.linspace(0, 2 * np.pi, 100)
    v = np.linspace(0, 2 * np.pi, 100)
    u, v = np.meshgrid(u, v)

    # Calculate the coordinates of the torus
    x = center[0] + (R + r * np.cos(v)) * np.cos(u)
    y = center[1] + (R + r * np.cos(v)) * np.sin(u)
    z = center[2] + r * np.sin(v)

    # Plot the surface of the torus with a color gradient and transparency
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.plot_surface(
        x, y, z, cmap="viridis", alpha=0.6
    )  # , edgecolor="none", alpha=0.6)

    # Plot the points as 'spheres'
    ax.scatter(
        points[:, 0],
        points[:, 1],
        points[:, 2],
        color="red",
        s=100,
        alpha=1,
        edgecolors="black",
    )  # s is the size of the point

    # Set labels
    ax.set_xlabel("X  ")
    ax.set_ylabel("Y  ")
    ax.set_zlabel("Z")
    ax.set_xlim(0, 30)
    ax.set_ylim(0, 30)
    ax.set_zlim(0, 30)

    # Show the plot
    plt.show()
