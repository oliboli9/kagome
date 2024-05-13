from ase.atoms import Atoms
from ase.cell import Cell
from ase.optimize import BFGS

from methods.calculator import KagomePotential, KagomeRadialPotential, RadialPotential
from methods.surface import SurfaceConstraint

import numpy as np


def calculate_midpoint(point1, point2):
    return (point1 + point2) / 2


class Kagome:
    def __init__(self, simplices, points, r0, surface=None, periodic=False):
        self.simplices = simplices
        self.points = points
        self.r0 = r0
        self.surface = surface
        self.periodic = periodic
        self.neighbour_coordinate_dict = self.kagome_coordinate_dict()
        self.kagome_positions, self.neighbour_indices_dict = self.kagome_indices_dict()
        self.atoms = self.kagome_atoms()

    def kagome_coordinate_dict(self):
        midpoints_dict = {}

        for simplex in self.simplices:
            # Calculate midpoints for current triangle
            midpoints = [
                calculate_midpoint(
                    self.points[simplex[i]], self.points[simplex[(i + 1) % 3]]
                )
                for i in range(3)
            ]
            midpoint_tuples = [tuple(midpoint) for midpoint in midpoints]

            # Store opposite midpoints
            for i in range(3):
                opposite_midpoint1 = midpoint_tuples[(i + 1) % 3]
                opposite_midpoint2 = midpoint_tuples[(i + 2) % 3]
                current_midpoint = midpoint_tuples[i]

                if current_midpoint in midpoints_dict:
                    midpoints_dict[current_midpoint].add(
                        (opposite_midpoint1, opposite_midpoint2)
                    )
                else:
                    midpoints_dict[current_midpoint] = {
                        (opposite_midpoint1, opposite_midpoint2)
                    }

        # Convert sets back to lists for easier use
        for midpoint in midpoints_dict:
            midpoints_dict[midpoint] = list(midpoints_dict[midpoint])

        return midpoints_dict

    def kagome_indices_dict(self):
        # Step 1: Extract all unique coordinates
        unique_coords = set()
        for coord1, coord_pairs in self.neighbour_coordinate_dict.items():
            unique_coords.add(coord1)
            for coord2, coord3 in coord_pairs:
                unique_coords.add(coord2)
                unique_coords.add(coord3)

        # Step 2: Create a mapping of coordinates to indices
        coord_to_index = {coord: i for i, coord in enumerate(unique_coords)}

        # Step 3: Build the new dictionary with indices
        indexed_dict = {}
        for coord1, coord_pairs in self.neighbour_coordinate_dict.items():
            indexed_pairs = [
                (coord_to_index[coord2], coord_to_index[coord3])
                for coord2, coord3 in coord_pairs
            ]
            indexed_dict[coord_to_index[coord1]] = indexed_pairs

        neighbour_keys = list(indexed_dict.keys())
        sorted_keys = sorted(neighbour_keys)
        sorted_neighbour_dict = {key: indexed_dict[key] for key in sorted_keys}
        sorted_positions = [None] * len(list(self.neighbour_coordinate_dict.keys()))
        for new_index, item in zip(
            neighbour_keys, list(self.neighbour_coordinate_dict.keys())
        ):
            sorted_positions[new_index] = item
        if not self.periodic:
            return sorted_positions, sorted_neighbour_dict
        else:
            square_start = 30
            square_end = 60

            def adjust_periodic(coord, square_start, square_end):
                square_size = square_end - square_start
                if coord < square_start:
                    return coord + square_size
                elif coord > square_end:
                    return coord - square_size
                else:
                    return coord

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
                                        [adjusted_x, adjusted_y],
                                        [pos[0], pos[1]],
                                        atol=1e-5,
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

    def kagome_atoms(self):
        cell = Cell.fromcellpar([30, 30, 30, 90, 90, 90])
        kagome_atoms = Atoms(
            "H" * len(self.kagome_positions),
            positions=self.kagome_positions,
            cell=cell,
            pbc=(1, 1, 0),
        )
        # kagome_atoms.calc = KagomePotential(neighbour_dict=self.neighbour_indices_dict)
        kagome_atoms.calc = KagomeRadialPotential(
            r0=self.r0, neighbour_dict=self.neighbour_indices_dict
        )
        # kagome_atoms.calc = RadialPotential(r0=self.r0)
        if self.surface is not None:
            constraint = SurfaceConstraint(self.surface)
            kagome_atoms.set_constraint(constraint)
        return kagome_atoms

    def straighten_weavers(self, fmax=0.275, steps=1000):
        local_minimisation = BFGS(self.atoms)
        local_minimisation.run(steps=steps, fmax=fmax)

    def plot_weavers(self, ax):
        positions = self.atoms.get_positions()
        for pos in positions:
            ax.scatter(*pos, color="blue", s=5)
        for atom_idx, neighbour_pairs in self.neighbour_indices_dict.items():
            atom_pos = positions[atom_idx]
            for neighbours in neighbour_pairs:
                for neighbour in neighbours:
                    neighbour_pos = positions[neighbour]
                    ax.plot(
                        [atom_pos[0], neighbour_pos[0]],
                        [atom_pos[1], neighbour_pos[1]],
                        [atom_pos[2], neighbour_pos[2]],
                        color="red",
                    )
        ax.set_xlabel("X axis")
        ax.set_ylabel("Y axis")
        ax.set_zlabel("Z axis")
        ax.set_xlim(0, 30)
        ax.set_ylim(0, 30)
        ax.set_zlim(0, 30)
        ax.set_title("Kagome Structure Connections")

    def plot_weavers_top_half(self, ax):
        positions = self.atoms.get_positions()
        center_z = 15  # Assuming the sphere is centered at (15, 15, 15)

        # Plotting only atoms that are in the top half of the sphere
        # for pos in positions:
        #     if pos[2] > center_z:  # Check if z-coordinate is in the top half
        #         ax.scatter(*pos, color="blue", s=5)

        # Plotting connections only if both atoms are in the top half
        for atom_idx, neighbour_pairs in self.neighbour_indices_dict.items():
            atom_pos = positions[atom_idx]
            if atom_pos[2] > center_z:  # Check if the atom itself is in the top half
                for neighbours in neighbour_pairs:
                    for neighbour in neighbours:
                        neighbour_pos = positions[neighbour]
                        if (
                            neighbour_pos[2] > center_z
                        ):  # Check if the neighbour is also in the top half
                            ax.plot(
                                [atom_pos[0], neighbour_pos[0]],
                                [atom_pos[1], neighbour_pos[1]],
                                # [atom_pos[2], neighbour_pos[2]],
                                "b-",
                                linewidth=0.5,
                            )

        ax.set_xlim(0, 30)
        ax.set_ylim(0, 30)
        # ax.set_xlabel("X axis")
        # ax.set_ylabel("Y axis")
        # ax.set_zlabel("Z axis")
        # ax.set_title("Kagome Structure Connections - Top Half")

    def plot_weavers_yz_projection(self, ax):
        positions = self.atoms.get_positions()
        # Filter points where the x-component is greater than 15
        filtered_positions = [pos for pos in positions if pos[0] > 15]

        # Plot each point in YZ plane
        # for pos in filtered_positions:
        #     ax.scatter(
        #         pos[1], pos[2], color="blue", s=5
        #     )  # Plot only y and z components

        # Iterate over connections in the structure
        for atom_idx, neighbour_pairs in self.neighbour_indices_dict.items():
            atom_pos = positions[atom_idx]
            if atom_pos[0] > 15:  # Check if the current atom's x-component is > 15
                for neighbours in neighbour_pairs:
                    for neighbour in neighbours:
                        neighbour_pos = positions[neighbour]
                        if (
                            neighbour_pos[0] > 15
                        ):  # Check if the neighbour's x-component is > 15
                            # Plot the connection in the YZ plane
                            ax.plot(
                                [atom_pos[1], neighbour_pos[1]],
                                [atom_pos[2], neighbour_pos[2]],
                                "b-",  # Blue line
                                linewidth=0.5,
                            )

        # ax.set_xlabel("Y axis")
        # ax.set_ylabel("Z axis")
        ax.set_xlim(0, 30)
        ax.set_ylim(0, 30)
        # ax.set_title("Filtered Kagome Structure Connections on YZ Plane")

    def plot_weavers_half(self, ax):
        """Plots the YZ plane with a front filter."""
        # Get positions from the atoms object
        positions = self.atoms.get_positions()

        # Define a filter for the front half (e.g., x ≥ 0)
        front_filter = [pos[0] >= 15 for pos in positions]

        # Plot points that meet the front filter in the YZ plane
        for pos, is_front in zip(positions, front_filter):
            if is_front:
                ax.scatter(pos[1], pos[2], color="blue", s=5)  # YZ plane

        # Plot connections for atoms in the front filter on the YZ plane
        for atom_idx, neighbour_pairs in self.neighbour_indices_dict.items():
            if not front_filter[atom_idx]:
                continue  # Skip atoms not in the front filter

            atom_pos = positions[atom_idx]
            for neighbours in neighbour_pairs:
                for neighbour in neighbours:
                    if front_filter[
                        neighbour
                    ]:  # Only plot connections within the front filter
                        neighbour_pos = positions[neighbour]
                        ax.plot(
                            [atom_pos[1], neighbour_pos[1]],  # Y-axis
                            [atom_pos[2], neighbour_pos[2]],  # Z-axis
                            color="red",
                        )

        # Set plot labels and title
        ax.set_xlabel("Y Axis")
        ax.set_ylabel("Z Axis")
        ax.set_box_aspect([1, 1, 1])
        ax.set_title("Connections on the YZ Plane")

    def plot_weavers_periodic(self, ax, projection_2d=False):
        if projection_2d:
            positions = self.atoms.get_positions()

            def is_periodic_neighbor(pos1, pos2):
                return abs(pos1[0] - pos2[0]) > 15 or abs(pos1[1] - pos2[1]) > 15

            for point, neighbour_pairs in self.neighbour_indices_dict.items():
                for neighbours in neighbour_pairs:
                    for neighbour in neighbours:
                        if not is_periodic_neighbor(
                            positions[point], positions[neighbour]
                        ):
                            ax.plot(
                                [positions[point][0], positions[neighbour][0]],
                                [positions[point][1], positions[neighbour][1]],
                                "b-",  # Blue line
                                linewidth=0.5,
                            )

            ax.set_xlabel("X ")
            ax.set_ylabel("Y ")
            ax.set_xlim(0, 30)
            ax.set_ylim(0, 30)
            # ax.set_title("Weavers 2D Projection on XY Plane")

        else:

            positions = self.atoms.get_positions()

            def is_periodic_neighbor(pos1, pos2):
                return abs(pos1[0] - pos2[0]) > 15 or abs(pos1[1] - pos2[1]) > 15

            for point, neighbour_pairs in self.neighbour_indices_dict.items():
                for neighbours in neighbour_pairs:
                    for neighbour in neighbours:
                        if not is_periodic_neighbor(
                            positions[point], positions[neighbour]
                        ):
                            ax.plot(
                                [positions[point][0], positions[neighbour][0]],
                                [positions[point][1], positions[neighbour][1]],
                                [positions[point][2], positions[neighbour][2]],
                                "b-",
                                linewidth=0.5,
                            )

            # ax.set_title("Weavers")
            ax.set_xlabel("X ")
            ax.set_ylabel("Y ")
            ax.set_zlabel("Z ")
