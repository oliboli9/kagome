from ase import Atoms
from ase.calculators.calculator import Calculator
from ase.cell import Cell
from ase.io.trajectory import Trajectory
from ase.optimize import BFGS

from itertools import islice
import matplotlib.pyplot as plt
from kagome import (
    find_triangle_midpoints,
    create_indices_dict,
)
from calculator import RadialPotential, KagomePotential, KagomeRadialPotential
from delaunay import repeat, make_delaunay_triangulation


def place_kagome_atoms(atoms, calculator: Calculator, repeats):
    coords = atoms.get_positions()
    points = repeat(repeats, coords)
    tri, neighbours = make_delaunay_triangulation(points)
    # print_neighbour_counts(points, neighbours)
    # plot_delaunay(tri, points, neighbours)
    kagome_positions_dict = find_triangle_midpoints(tri, points)
    kagome_neighbour_dict = create_indices_dict(kagome_positions_dict)
    neighbour_keys = list(kagome_neighbour_dict.keys())
    sorted_keys = sorted(neighbour_keys)
    sorted_neighbour_dict = {key: kagome_neighbour_dict[key] for key in sorted_keys}
    sorted_positions = [None] * len(list(kagome_positions_dict.keys()))
    for new_index, item in zip(neighbour_keys, list(kagome_positions_dict.keys())):
        sorted_positions[new_index] = item

    # make a check here that its sorted properly

    cell = Cell.fromcellpar([30 * repeats, 30 * repeats, 30 * repeats, 90, 90, 90])

    kagome_atoms = Atoms(
        "H" * len(sorted_positions),
        positions=sorted_positions,
        cell=cell,
        pbc=(1, 1, 0),
    )
    if isinstance(calculator, KagomePotential):
        kagome_atoms.calc = KagomePotential(neighbour_dict=sorted_neighbour_dict)
    else:
        kagome_atoms.calc = KagomeRadialPotential(
            r0=2, neighbour_dict=sorted_neighbour_dict
        )
    return kagome_atoms, sorted_neighbour_dict


def straighten_and_plot_weavers(kagome_atoms, neighbour_dict):
    fig, (ax1, ax2) = plt.subplots(1, 2)
    positions = kagome_atoms.get_positions()
    for point, neighbour_pairs in neighbour_dict.items():
        for neighbours in neighbour_pairs:
            for neighbour in neighbours:
                ax1.plot(
                    [positions[point][0], positions[neighbour][0]],
                    [positions[point][1], positions[neighbour][1]],
                    "b-",
                    linewidth=0.5,
                )
                # plt.plot(
                #     [point[0], neighbour[0]],
                #     [point[1], neighbour[1]],
                #     "b-",
                #     linewidth=0.5,
                # )

    ax1.set_title("Weavers")
    ax1.set_xlabel("X coordinate")
    ax1.set_ylabel("Y coordinate")

    local_minimisation = BFGS(kagome_atoms)
    local_minimisation.run(steps=20)

    positions = kagome_atoms.get_positions()
    for point, neighbour_pairs in neighbour_dict.items():
        for neighbours in neighbour_pairs:
            for neighbour in neighbours:
                ax2.plot(
                    [positions[point][0], positions[neighbour][0]],
                    [positions[point][1], positions[neighbour][1]],
                    "b-",
                    linewidth=0.5,
                )
                # plt.plot(
                #     [point[0], neighbour[0]],
                #     [point[1], neighbour[1]],
                #     "b-",
                #     linewidth=0.5,
                # )

    ax2.set_title("Weavers")
    ax2.set_xlabel("X coordinate")
    ax2.set_ylabel("Y coordinate")


def straighten_and_plot_weavers_periodic(kagome_atoms, neighbour_dict):
    fig, (ax1, ax2) = plt.subplots(1, 2)
    positions = kagome_atoms.get_positions()

    def is_periodic_neighbor(pos1, pos2):
        return abs(pos1[0] - pos2[0]) > 15 or abs(pos1[1] - pos2[1]) > 15

    for point, neighbour_pairs in neighbour_dict.items():
        for neighbours in neighbour_pairs:
            for neighbour in neighbours:
                if not is_periodic_neighbor(positions[point], positions[neighbour]):
                    ax1.plot(
                        [positions[point][0], positions[neighbour][0]],
                        [positions[point][1], positions[neighbour][1]],
                        "b-",
                        linewidth=0.5,
                    )

    ax1.set_title("Weavers")
    ax1.set_xlabel("X coordinate")
    ax1.set_ylabel("Y coordinate")

    local_minimisation = BFGS(kagome_atoms)
    local_minimisation.run(steps=20)

    positions = kagome_atoms.get_positions()
    for point, neighbour_pairs in neighbour_dict.items():
        for neighbours in neighbour_pairs:
            for neighbour in neighbours:
                if not is_periodic_neighbor(positions[point], positions[neighbour]):
                    ax2.plot(
                        [positions[point][0], positions[neighbour][0]],
                        [positions[point][1], positions[neighbour][1]],
                        "b-",
                        linewidth=0.5,
                    )

    ax2.set_title("Weavers")
    ax2.set_xlabel("X coordinate")
    ax2.set_ylabel("Y coordinate")


def straighten_and_plot_weavers_torus(kagome_atoms, neighbour_dict):
    fig = plt.figure(figsize=(7, 4))
    positions = kagome_atoms.get_positions()
    print(positions.shape)
    # Kagome structure plot on the left (subplot 1)
    ax1 = fig.add_subplot(121, projection="3d")
    for pos in positions:
        ax1.scatter(*pos, color="blue", s=5)
    for atom_idx, neighbour_pairs in neighbour_dict.items():
        atom_pos = positions[atom_idx]
        for neighbours in neighbour_pairs:
            for neighbour in neighbours:
                neighbour_pos = positions[neighbour]
                ax1.plot(
                    [atom_pos[0], neighbour_pos[0]],
                    [atom_pos[1], neighbour_pos[1]],
                    [atom_pos[2], neighbour_pos[2]],
                    color="red",
                )
    ax1.set_xlabel("X axis")
    ax1.set_ylabel("Y axis")
    ax1.set_zlabel("Z axis")
    ax1.set_title("Kagome Structure Connections")

    local_minimisation = BFGS(kagome_atoms)
    local_minimisation.run(steps=50)

    # Another subplot (subplot 2)
    ax2 = fig.add_subplot(122, projection="3d")
    for pos in positions:
        ax2.scatter(*pos, color="blue", s=5)
    for atom_idx, neighbour_pairs in neighbour_dict.items():
        atom_pos = positions[atom_idx]
        for neighbours in neighbour_pairs:
            for neighbour in neighbours:
                neighbour_pos = positions[neighbour]
                ax2.plot(
                    [atom_pos[0], neighbour_pos[0]],
                    [atom_pos[1], neighbour_pos[1]],
                    [atom_pos[2], neighbour_pos[2]],
                    color="red",
                )
    ax2.set_xlabel("X axis")
    ax2.set_ylabel("Y axis")
    ax2.set_zlabel("Z axis")
    ax2.set_title("Kagome Structure Connections")


# Your code to call the function goes here


# # trajectory_paths = ["trajectories/10001/bfgs-3-3-110-2000-500-50.traj"]
# trajectory_paths = ["bfgs-4-3-22-1500-150-50.traj"]
# # trajectory_paths = ["polar.traj"]

# for path in trajectory_paths:
#     traj = Trajectory(f"{path}")
#     atoms = traj[-1]
#     kagome_atoms, neighbour_dict = place_kagome_atoms(atoms, KagomePotential, 3)
#     straighten_and_plot_weavers(kagome_atoms, neighbour_dict)
#     plt.show()
