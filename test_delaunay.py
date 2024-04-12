import numpy as np
import sympy as sp
import matplotlib.pyplot as plt
from scipy.spatial import Delaunay


# Define the function for the surface amplitude
def surface_amplitude(x, y):
    return 3 * sp.sin(2 * sp.pi * x / 30) * sp.sin(2 * sp.pi * y / 30)


# Define density (number of points) for triangulation
density = 30

# Create a grid of points
x = np.linspace(-30, 30, density)
y = np.linspace(-30, 30, density)
x, y = np.meshgrid(x, y)

# Compute the z-values for the grid points
z = np.array([surface_amplitude(xi, yi) for xi, yi in zip(np.ravel(x), np.ravel(y))])

# Create an array of x, y, z points for Delaunay triangulation
points = np.column_stack((x.ravel(), y.ravel(), z))

# Perform Delaunay triangulation
tri = Delaunay(points[:, :2])

# Plot the Delaunay triangulation
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection="3d")
for simplex in tri.simplices:
    simplex = np.append(simplex, simplex[0])  # Loop back to the first point
    ax.plot(
        points[simplex, 0],
        points[simplex, 1],
        points[simplex, 2],
        color="lightblue",
        linewidth=0.5,
    )
# ax.plot_trisurf(points[:, 0], points[:, 1], points[:, 2], triangles=tri.simplices)
ax.set_xlabel("X-axis")
ax.set_ylabel("Y-axis")
ax.set_zlabel("Z-axis (Amplitude)")
plt.title("Delaunay Triangulation of Surface Amplitude")
plt.show()
