from itertools import product
import math
import os
import numpy as np
import sympy as sp
from numpy import transpose
from numpy.random import SeedSequence, default_rng
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from multiprocessing import Pool
from scipy.integrate import dblquad
from functools import partial
from collections import Counter
from scipy.spatial import Delaunay

from ase import Atoms, units
from ase.calculators.calculator import Calculator
from ase.calculators.emt import EMT
from ase.calculators.lj import LennardJones
from ase.calculators.morse import MorsePotential
from ase.constraints import IndexedConstraint
from ase.cell import Cell
from ase.constraints import FixedPlane
from ase.io.trajectory import Trajectory
from ase.md.langevin import Langevin
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from ase.md.verlet import VelocityVerlet
from ase.neighborlist import neighbor_list
from ase.optimize import BFGS
from ase.parallel import paropen
from ase.visualize import view

from calculator import RadialPotential, KagomePotential
from delaunay import make_delaunay_triangulation, print_neighbour_counts, plot_delaunay
from surface import Surface, SurfaceConstraint

trajectory_paths = [
    "trajectories/10001/langevin-0-5-130-1500-150-50.traj"
]

def calculate_midpoint(point1, point2):
    return (point1 + point2) / 2

def find_triangle_midpoints(tri, points):
    midpoints_dict = {}

    for simplex in tri.simplices:
        # Calculate midpoints for current triangle
        midpoints = [calculate_midpoint(points[simplex[i]], points[simplex[(i + 1) % 3]]) for i in range(3)]
        midpoint_tuples = [tuple(midpoint) for midpoint in midpoints]

        # Store opposite midpoints
        for i in range(3):
            opposite_midpoint1 = midpoint_tuples[(i + 1) % 3]
            opposite_midpoint2 = midpoint_tuples[(i + 2) % 3]
            current_midpoint = midpoint_tuples[i]

            if current_midpoint in midpoints_dict:
                midpoints_dict[current_midpoint].add((opposite_midpoint1, opposite_midpoint2))
            else:
                midpoints_dict[current_midpoint] = {(opposite_midpoint1, opposite_midpoint2)}

    # Convert sets back to lists for easier use
    for midpoint in midpoints_dict:
        midpoints_dict[midpoint] = list(midpoints_dict[midpoint])

    return midpoints_dict

for path in trajectory_paths:
    atoms, tri, points, points_2d, neighbours, amplitude = make_delaunay_triangulation(
        path
    )
    print_neighbour_counts(points, neighbours)
    # plot_delaunay(tri, points, points_2d, neighbours, amplitude)

    kagome_positions = find_triangle_midpoints(tri, points)
    print(len(kagome_positions))
    # print(kagome_positions.keys())
    cell = Cell.fromcellpar([30, 30, 30, 90, 90, 90])

    kagome_atoms = Atoms(
        "H" * len(kagome_positions),
        positions=list(kagome_positions.keys()),
        cell=cell,
        pbc=(1, 1, 0),
    )

    # kagome_atoms.calc = KagomePotential(neighbour_dict=kagome_positions)
    kagome_atoms.calc = RadialPotential(r0=2)
    surface = Surface(
        lambda x, y: 3 * sp.sin(2 * sp.pi * x / 30) * sp.sin(2 * sp.pi * y / 30),
        n=len(kagome_atoms),
    )
    constraint = SurfaceConstraint(surface)
    kagome_atoms.set_constraint(constraint)

    potential_energy = kagome_atoms.get_potential_energy()
    print("Potential energy:", potential_energy, "eV")

    local_minimisation = BFGS(kagome_atoms)
    # traj_bfgs = Trajectory(
    #     f"trajectories-surfaces/{seed}/bfgs-{n}-{amp}-{surface.n}-{start_temp}-{cooling_rate}-{end_temp}.traj",
    #     "w",
    #     atoms,
    # )

    # local_minimisation.attach(wrap_and_write_bfgs)
    local_minimisation.run(steps=1000, fmax=0.015)



