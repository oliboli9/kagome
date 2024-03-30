import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
from scipy.spatial import Delaunay
from mpl_toolkits.mplot3d import Axes3D
from ase.io.trajectory import Trajectory


def repeat(n, points):
    repeated_points = []
    for i in range(n):  # Repeat in x-direction
        for j in range(n):  # Repeat in y-direction
            new_points = points.copy()
            new_points[:, 0] += 30 * i
            new_points[:, 1] += 30 * j
            repeated_points.append(new_points)

    repeated_points = np.vstack(repeated_points)

    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection="3d")
    # ax.scatter(
    #     repeated_points[:, 0], repeated_points[:, 1], repeated_points[:, 2], color="blue"
    # )
    # plt.show()
    return repeated_points


def find_neighbors(tri, points_2d):
    neighbors = [[] for _ in range(len(points_2d))]
    for simplex in tri.simplices:
        for i, j in zip(simplex, simplex[[1, 2, 0]]):
            neighbors[i].append(j)
            neighbors[j].append(i)
    return [list(set(nbr)) for nbr in neighbors]


def make_delaunay_triangulation(points):
    points_2d = points[:, :2]
    tri = Delaunay(points_2d)

    neighbours = find_neighbors(tri, points_2d)
    return tri, neighbours


def plot_delaunay(tri, points, neighbours, path):
    points_2d = points[:, :2]
    nine_neighbours = np.array([len(nbrs) == 9 for nbrs in neighbours])
    eight_neighbours = np.array([len(nbrs) == 8 for nbrs in neighbours])
    seven_neighbours = np.array([len(nbrs) == 7 for nbrs in neighbours])
    six_neighbours = np.array([len(nbrs) == 6 for nbrs in neighbours])
    five_neighbours = np.array([len(nbrs) == 5 for nbrs in neighbours])
    four_neighbours = np.array([len(nbrs) == 4 for nbrs in neighbours])
    three_neighbours = np.array([len(nbrs) == 3 for nbrs in neighbours])

    fig = plt.figure(figsize=(12, 6))
    fig.suptitle(
        f"Delaunay triangulation on Sine curve",  # with amplitude {amplitude}",
        fontsize=16,
    )

    ax1 = fig.add_subplot(1, 2, 1)
    ax1.triplot(points_2d[:, 0], points_2d[:, 1], tri.simplices)
    ax1.scatter(
        points_2d[nine_neighbours, 0],
        points_2d[nine_neighbours, 1],
        color="red",
        label="9 neighbours",
    )
    ax1.scatter(
        points_2d[eight_neighbours, 0],
        points_2d[eight_neighbours, 1],
        color="orange",
        label="8 neighbours",
    )
    ax1.scatter(
        points_2d[seven_neighbours, 0],
        points_2d[seven_neighbours, 1],
        color="yellow",
        label="7 neighbours",
    )
    # ax1.scatter(points_2d[six_neighbours, 0], points_2d[six_neighbours, 1], color="green", label="6 neighbours")
    ax1.scatter(
        points_2d[five_neighbours, 0],
        points_2d[five_neighbours, 1],
        color="blue",
        label="5 neighbours",
    )
    ax1.scatter(
        points_2d[four_neighbours, 0],
        points_2d[four_neighbours, 1],
        color="purple",
        label="4 neighbours",
    )
    ax1.scatter(
        points_2d[three_neighbours, 0],
        points_2d[three_neighbours, 1],
        color="pink",
        label="3 neighbours",
    )
    ax1.set_title("2D Triangulation")
    ax1.set_xlabel("X-axis")
    ax1.set_ylabel("Y-axis")
    ax1.legend(loc="upper left", bbox_to_anchor=(1, 1))

    ax2 = fig.add_subplot(1, 2, 2, projection="3d")
    for simplex in tri.simplices:
        simplex = np.append(simplex, simplex[0])  # Loop back to the first point
        ax2.plot(
            points[simplex, 0],
            points[simplex, 1],
            points[simplex, 2],
            color="lightblue",
            linewidth=0.5,
        )
    ax2.scatter(
        points[nine_neighbours, 0],
        points[nine_neighbours, 1],
        points[nine_neighbours, 2],
        color="red",
    )
    ax2.scatter(
        points[eight_neighbours, 0],
        points[eight_neighbours, 1],
        points[eight_neighbours, 2],
        color="orange",
    )
    ax2.scatter(
        points[seven_neighbours, 0],
        points[seven_neighbours, 1],
        points[seven_neighbours, 2],
        color="yellow",
    )
    # ax2.scatter(
    #     points[six_neighbours, 0],
    #     points[six_neighbours, 1],
    #     points[six_neighbours, 2],
    #     color="green",
    # )
    ax2.scatter(
        points[five_neighbours, 0],
        points[five_neighbours, 1],
        points[five_neighbours, 2],
        color="blue",
    )
    ax2.scatter(
        points[four_neighbours, 0],
        points[four_neighbours, 1],
        points[four_neighbours, 2],
        color="purple",
    )
    ax2.scatter(
        points[three_neighbours, 0],
        points[three_neighbours, 1],
        points[three_neighbours, 2],
        color="pink",
    )
    ax2.set_title("3D Triangulation")
    ax2.set_xlabel("X-axis")
    ax2.set_ylabel("Y-axis")
    ax2.set_zlabel("Z-axis")

    plt.tight_layout()
    ax1.set_xlim(30, 60)
    ax1.set_ylim(30, 60)
    ax2.view_init(89, -90)
    # ax2.set_xlim(30, 60)
    # ax2.set_ylim(30, 60)
    # plt.show()
    plt.savefig(f"{path}.png", dpi=300)


def print_neighbour_counts(points, neighbours):
    # Step 1: Filter points based on x and y range
    range_filter = (
        (points[:, 0] >= 30)
        & (points[:, 0] <= 60)
        & (points[:, 1] >= 30)
        & (points[:, 1] <= 60)
    )
    filtered_points_indices = np.where(range_filter)[0]

    # Step 2: Get neighbors for these filtered points
    filtered_neighbors = [neighbours[i] for i in filtered_points_indices]

    # Step 3: Count the number of neighbors for each filtered point
    filtered_neighbor_counts = [len(n) for n in filtered_neighbors]

    # Step 4: Use Counter to count the frequency of each unique neighbor count among filtered points
    filtered_count_frequency = Counter(filtered_neighbor_counts)

    # Now filtered_count_frequency is a dictionary where keys are the number of neighbors,
    # and values are the frequencies of these neighbor counts for filtered points
    print(filtered_count_frequency)
