from ase.atoms import Atoms
from ase.cell import Cell
from ase.optimize import BFGS

from calculator import KagomePotential, KagomeRadialPotential
from surface import SurfaceConstraint


def calculate_midpoint(point1, point2):
    return (point1 + point2) / 2


class Kagome:
    def __init__(self, simplices, points, surface=None):
        self.simplices = simplices
        self.points = points
        self.surface = surface
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
        return sorted_positions, sorted_neighbour_dict

    def kagome_atoms(self):
        cell = Cell.fromcellpar([30, 30, 30, 90, 90, 90])
        kagome_atoms = Atoms(
            "H" * len(self.kagome_positions),
            positions=self.kagome_positions,
            cell=cell,
            pbc=(1, 1, 0),
        )
        # kagome_atoms.calc = KagomeRadialPotential(r0=2, neighbour_dict=neighbour_dict)
        kagome_atoms.calc = KagomePotential(neighbour_dict=self.neighbour_indices_dict)
        if self.surface is not None:
            constraint = SurfaceConstraint(self.surface)
            kagome_atoms.set_constraint(constraint)
        return kagome_atoms

    def straighten_weavers(self):
        local_minimisation = BFGS(self.atoms)
        local_minimisation.run(steps=50)

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
        ax.set_title("Kagome Structure Connections")

    def plot_weavers_periodic(self, ax):
        positions = self.atoms.get_positions()

        def is_periodic_neighbor(pos1, pos2):
            return abs(pos1[0] - pos2[0]) > 15 or abs(pos1[1] - pos2[1]) > 15

        for point, neighbour_pairs in self.neighbour_dict.items():
            for neighbours in neighbour_pairs:
                for neighbour in neighbours:
                    if not is_periodic_neighbor(positions[point], positions[neighbour]):
                        ax.plot(
                            [positions[point][0], positions[neighbour][0]],
                            [positions[point][1], positions[neighbour][1]],
                            "b-",
                            linewidth=0.5,
                        )

        ax.set_title("Weavers")
        ax.set_xlabel("X coordinate")
        ax.set_ylabel("Y coordinate")
