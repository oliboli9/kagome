from ase import Atoms
from ase.calculators.calculator import Calculator
from ase.cell import Cell
from ase.io.trajectory import Trajectory
from ase.optimize import BFGS

from itertools import islice
import numpy as np
import matplotlib.pyplot as plt
from kagome import (
    find_triangle_midpoints,
    create_indices_dict,
)
from calculator import RadialPotential, KagomePotential, KagomeRadialPotential
from delaunay import repeat, make_delaunay_triangulation, get_neighbour_counts
from weavers import straighten_and_plot_weavers, straighten_and_plot_weavers_periodic


def initialise_kagome_atoms(points):
    tri, neighbours = make_delaunay_triangulation(points)
    kagome_positions_dict = find_triangle_midpoints(tri, points)

    kagome_neighbour_dict = create_indices_dict(kagome_positions_dict)
    neighbour_keys = list(kagome_neighbour_dict.keys())
    sorted_keys = sorted(neighbour_keys)
    sorted_neighbour_dict = {key: kagome_neighbour_dict[key] for key in sorted_keys}
    sorted_positions = [None] * len(list(kagome_positions_dict.keys()))
    for new_index, item in zip(neighbour_keys, list(kagome_positions_dict.keys())):
        sorted_positions[new_index] = item
    return sorted_positions, sorted_neighbour_dict


# make a check here that its sorted properly
def adjust_periodic(coord, square_start, square_end):
    square_size = square_end - square_start
    if coord < square_start:
        return coord + square_size
    elif coord > square_end:
        return coord - square_size
    else:
        return coord


def make_periodic_connections(atoms, square_start, square_end):
    coords = atoms.get_positions()
    repeats = 5
    points = repeat(repeats, coords)
    sorted_positions, sorted_neighbour_dict = initialise_kagome_atoms(points)

    # Create a mapping from original indices to new indices in the specified range
    index_mapping = {
        old_idx: new_idx
        for new_idx, old_idx in enumerate(
            atom_idx
            for atom_idx, atom_pos in enumerate(sorted_positions)
            if square_start <= atom_pos[0] <= square_end
            and square_start <= atom_pos[1] <= square_end
        )
    }

    # Adjust neighbors and filter positions
    adjusted_neighbour_dict = {}
    for old_idx, neighbor_pairs in sorted_neighbour_dict.items():
        if old_idx in index_mapping:
            new_neighbors = []
            for neighbor_pair in neighbor_pairs:
                new_pair = []
                for neighbor_idx in neighbor_pair:
                    neighbor_pos = sorted_positions[neighbor_idx]
                    # Apply periodic boundary adjustments
                    adjusted_x = adjust_periodic(
                        neighbor_pos[0], square_start, square_end
                    )
                    adjusted_y = adjust_periodic(
                        neighbor_pos[1], square_start, square_end
                    )
                    # Find the corresponding new index
                    for idx, pos in enumerate(sorted_positions):
                        if (
                            square_start <= pos[0] <= square_end
                            and square_start <= pos[1] <= square_end
                            and np.allclose(
                                [adjusted_x, adjusted_y], [pos[0], pos[1]], atol=1e-5
                            )
                        ):
                            new_pair.append(index_mapping[idx])
                            break
                if len(new_pair) == 2:
                    new_neighbors.append(tuple(new_pair))
            adjusted_neighbour_dict[index_mapping[old_idx]] = new_neighbors

    # Shift the positions in the specified range
    shifted_positions = [
        (pos[0] - square_start, pos[1] - square_start, pos[2])
        for idx, pos in enumerate(sorted_positions)
        if idx in index_mapping
    ]

    return shifted_positions, adjusted_neighbour_dict


path = "bfgs-4-3-22-1500-150-50.traj"
traj = Trajectory(f"{path}")
atoms = traj[-1]

positions, neighbour_dict = make_periodic_connections(atoms, 30, 120)


# print(len(neighbour_dict))
def plot_to_check_neighbour_periodicity():
    x_coords = [pos[0] for pos in positions]
    y_coords = [pos[1] for pos in positions]

    # Plotting all atoms
    plt.scatter(x_coords, y_coords, c="blue", label="Atoms")

    # Highlighting the atom at index 1
    plt.scatter(
        x_coords[23],
        y_coords[23],
        c="red",
        label="Highlighted Atom (Index 23)",
        s=100,
    )
    plt.scatter(
        x_coords[28],
        y_coords[28],
        c="black",
        label="Highlighted Atom (Index 28)",
        s=100,
    )
    plt.scatter(
        x_coords[12],
        y_coords[12],
        c="black",
        label="Highlighted Atom (Index 12)",
        s=100,
    )

    # Adding labels and title
    plt.xlabel("X Coordinate")
    plt.ylabel("Y Coordinate")
    plt.title("2D Plot of Atoms with Highlighted Atom")
    plt.legend(loc="upper left", bbox_to_anchor=(1, 1))
    plt.tight_layout()

    # Display the plot
    plt.show()


# plot_to_check_neighbour_periodicity()
cell = Cell.fromcellpar([30, 30, 30, 90, 90, 90])
kagome_atoms = Atoms(
    "H" * len(positions),
    positions=positions,
    cell=cell,
    pbc=(1, 1, 0),
)
# kagome_atoms.calc = KagomeRadialPotential(r0=2, neighbour_dict=neighbour_dict)
kagome_atoms.calc = KagomePotential(neighbour_dict=neighbour_dict)


# Use the adjusted neighbor dictionary for the calculation
# if isinstance(calculator, KagomePotential):
#     kagome_atoms.calc = KagomePotential(neighbour_dict=neighbour_dict)
# else:
#     kagome_atoms.calc = KagomeRadialPotential(r0=2, neighbour_dict=neighbour_dict)

straighten_and_plot_weavers_periodic(kagome_atoms, neighbour_dict)
plt.show()
