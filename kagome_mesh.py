from scipy.spatial import Delaunay
import numpy as np
from ase import Atoms
from ase.io.trajectory import Trajectory, TrajectoryWriter
import matplotlib.pyplot as plt


traj = Trajectory("bfgs-0-3-109-1400-150-50.traj")
atoms = traj[-1]

positions = atoms.get_positions()


points_2d = positions[:, :2]
tri = Delaunay(points_2d)
fig = plt.figure(figsize=(12, 6))
ax1 = fig.add_subplot(1, 1, 1, projection="3d")

for simplex in tri.simplices:
    simplex = np.append(simplex, simplex[0])  # Loop back to the first point
    ax1.plot(
        positions[simplex, 0],
        positions[simplex, 1],
        positions[simplex, 2],
        color="lightblue",
        linewidth=0.5,
    )
# ax1.scatter(positions[:, 0], positions[:, 1], positions[:, 2])


def midpoint(p1, p2):
    return (p1 + p2) / 2


def create_midpoint_triangles(points, triangles):
    midpoint_triangles = []
    for tri in triangles:
        pts = points[tri]

        mid1 = midpoint(pts[0], pts[1])
        mid2 = midpoint(pts[1], pts[2])
        mid3 = midpoint(pts[2], pts[0])

        # Create a new triangle from midpoints
        midpoint_triangles.append([mid1, mid2, mid3])

    return midpoint_triangles


def angle_between_vectors(v1, v2):
    """Compute the angle in degrees between two vectors."""
    unit_v1 = v1 / np.linalg.norm(v1)
    unit_v2 = v2 / np.linalg.norm(v2)
    dot_product = np.dot(unit_v1, unit_v2)
    angle_rad = np.arccos(np.clip(dot_product, -1.0, 1.0))
    return np.degrees(angle_rad)


def find_neighbor_edges(tri, edge):
    """Find neighboring edges of a given edge in the Delaunay triangulation."""
    neighbors = []
    for simplex in tri.simplices:
        if all(vertex in simplex for vertex in edge):
            # Add the other edges of the triangle
            neighbors.extend([simplex[i] for i in range(3) if simplex[i] not in edge])
    return neighbors


def classify_edges(midpoint_triangles, points, tri):
    edge_classification = {}
    color_cycle = ["red", "green", "blue"]
    color_index = 0

    for mid_tri in midpoint_triangles:
        for i in range(3):
            p1, p2 = mid_tri[i], mid_tri[(i + 1) % 3]
            edge_vector = p2 - p1
            neighbors = find_neighbor_edges(tri, [p1, p2])

            min_angle = 180
            min_edge = None

            # Find the neighboring edge that forms the closest angle to 180 degrees
            for neighbor in neighbors:
                neighbor_vector = points[neighbor] - points[(neighbor + 1) % 3]
                angle = angle_between_vectors(edge_vector, neighbor_vector)
                if abs(180 - angle) < abs(180 - min_angle):
                    min_angle = angle
                    min_edge = neighbor

            # Classify the edge
            if min_edge is not None:
                edge_classification[(p1, p2)] = color_cycle[color_index]
                edge_classification[(points[min_edge], points[(min_edge + 1) % 3])] = (
                    color_cycle[color_index]
                )

            color_index = (color_index + 1) % 3

    return edge_classification


midpoint_triangles = create_midpoint_triangles(positions, tri.simplices)
edge_classification = classify_edges(midpoint_triangles, positions, tri)
print(edge_classification)

for triangle in midpoint_triangles:
    # To ensure the triangle is closed, we repeat the first point at the end
    triangle = np.vstack([triangle, triangle[0]])
    ax1.plot(triangle[:, 0], triangle[:, 1], triangle[:, 2])  # Plot x vs y

plt.show()

# plt.savefig("surfaces/kagome-mesh")
