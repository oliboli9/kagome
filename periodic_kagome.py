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
from delaunay import repeat, make_delaunay_triangulation, get_neighbour_counts

calculator = KagomeRadialPotential()
path = ["bfgs-4-3-22-1500-150-50.traj"]
traj = Trajectory(f"{path}")
atoms = traj[-1]
coords = atoms.get_positions()
repeats = 3
points = repeat(repeats, coords)

tri, neighbours = make_delaunay_triangulation(points)
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
