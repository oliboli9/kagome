import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import Delaunay
from sklearn.neighbors import NearestNeighbors


def generate_sphere_points(num_points, radius):
    theta = np.arccos(1 - 2 * np.random.rand(num_points))
    phi = 2 * np.pi * np.random.rand(num_points)
    x = radius * np.sin(theta) * np.cos(phi)
    y = radius * np.sin(theta) * np.sin(phi)
    z = radius * np.cos(theta)
    points_array = np.vstack((x, y, z)).T
    return points_array


def generate_torus_points(num_points, R, r):
    """
    Generate points on the surface of a torus.

    Parameters:
    num_points (int): Number of points to generate.
    R (float): Major radius of the torus (distance from the center of the tube to the center of the torus).
    r (float): Minor radius of the torus (radius of the tube).

    Returns:
    numpy.ndarray: Array of points on the torus.
    """
    theta = 2 * np.pi * np.random.rand(num_points)
    phi = 2 * np.pi * np.random.rand(num_points)
    x = (R + r * np.cos(phi)) * np.cos(theta)
    y = (R + r * np.cos(phi)) * np.sin(theta)
    z = r * np.sin(phi)

    return np.column_stack((x, y, z))


def rows_appearing_more_than_once(arr):
    # Find unique rows and their counts
    arr = np.vstack(arr)
    unique_rows, counts = np.unique(arr, axis=0, return_counts=True)

    # Filter rows that appear more than once
    repeated_rows = unique_rows[counts > 1]

    return repeated_rows


def find_unique_triangles(triangles):
    # Convert the list of triangles to a 2D NumPy array
    all_triangles = np.vstack(triangles)

    # Find unique rows (triangles) and return them
    unique_triangles = np.unique(all_triangles, axis=0)
    return unique_triangles


# Parameters
num_points = 1000  # Number of points
radius = 1  # Radius of the sphere

# Generate points
points = generate_sphere_points(num_points, radius)
points = generate_torus_points(500, 10, 3)

knn = NearestNeighbors(n_neighbors=10).fit(points)
neighbours = knn.kneighbors(return_distance=False)

all_triangles = []

for point_idx, point in enumerate(points):
    neighborhood = neighbours[point_idx]
    local_points = points[neighborhood, :2]  # Project to 2D
    local_tri = Delaunay(local_points).simplices

    # Adjust indices to global index
    global_tri = neighborhood[local_tri]

    # Add to global list
    all_triangles.extend(global_tri)

# Find unique triangles
final_triangles = rows_appearing_more_than_once(all_triangles)
print(len(final_triangles))

# Plotting in 3D with lines instead of a filled surface
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection="3d")

for tri in final_triangles:
    tri_points = points[tri]
    ax.plot(
        tri_points[[0, 1, 2, 0], 0],
        tri_points[[0, 1, 2, 0], 1],
        tri_points[[0, 1, 2, 0], 2],
        color="skyblue",
    )

ax.scatter(points[:, 0], points[:, 1], points[:, 2], color="k", s=1)
ax.set_title("3D Delaunay Triangulation (Line Plot)")
plt.show()

# Plotting in 3D
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection="3d")

# Plotting the triangulated surface
ax.plot_trisurf(
    points[:, 0],
    points[:, 1],
    points[:, 2],
    triangles=final_triangles,
    color="skyblue",
    alpha=0.5,
)

# Plotting the points as black dots
ax.scatter(points[:, 0], points[:, 1], points[:, 2], color="k", s=1)

ax.set_title("3D Delaunay Triangulation on Overlapping Neighborhoods")
plt.show()

# Plotting
plt.figure(figsize=(8, 6))
plt.triplot(points[:, 0], points[:, 1], final_triangles)
plt.title("2D Delaunay Triangulation on Overlapping Neighborhoods")
plt.xlabel("X Coordinate")
plt.ylabel("Y Coordinate")
plt.gca().set_aspect("equal", adjustable="box")
plt.show()
