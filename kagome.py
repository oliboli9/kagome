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


class RadialPotential(Calculator):
    """Radial potential."""

    implemented_properties = ["energy", "forces"]
    default_parameters = {"V0": 1.0, "r0": 4}

    nolabel = True

    def __init__(self, **kwargs):
        """
        Parameters
        ----------
        V0: float
        Energy scale, default 1.0
        r0: float
        Range of quadratic potential, default 1.0
        """
        Calculator.__init__(self, **kwargs)

    def calculate(
        self,
        atoms=None,
        properties=["energy", "forces"],
        system_changes=["positions", "numbers", "cell", "pbc", "charges", "magmoms"],
    ):
        Calculator.calculate(self, atoms, properties, system_changes)

        V0 = self.parameters.V0
        r0 = self.parameters.r0

        forces = np.zeros((len(self.atoms), 3))
        preF = -2 * V0 / r0

        i, j, d, D = neighbor_list("ijdD", atoms, r0)
        dhat = (D / d[:, None]).T

        dd = 1 - d / r0
        E = V0 * dd**2
        dE = preF * dd * dhat
        energy = 0.5 * E.sum()

        F = dE.T
        for dim in range(3):
            forces[:, dim] = np.bincount(i, weights=F[:, dim], minlength=len(atoms))

        self.results["energy"] = energy
        self.results["forces"] = forces


class Surface:
    def __init__(self, f, density=None, n=None):
        x, y = sp.symbols("x y")
        self.f = f(x, y)
        self.density = density
        self.df_dx = sp.diff(self.f, x)
        self.df_dy = sp.diff(self.f, y)
        """
        The number of atoms n is such that the density of atoms on the surface is 1/9 atoms/Å^2
        (optimal density is based on # of Au atoms that make a lattice with Au bond distance)
        """
        area_expr = sp.lambdify(
            (x, y), sp.sqrt(1 + self.df_dx**2 + self.df_dy**2), "numpy"
        )
        if n is None:
            assert (
                density is not None
            ), "Pass either desired density or desired number of atoms"
            area, error = dblquad(area_expr, 0, 30, lambda x: 0, lambda x: 30)
            self.n = int(area / density)
        else:
            self.n = n

    def elevate_to_surface(self, x_vals, y_vals):
        x, y = sp.symbols("x y")
        f_num = sp.lambdify((x, y), self.f, "numpy")
        return f_num(x_vals, y_vals)

    def dx(self, x_vals, y_vals):
        x, y = sp.symbols("x y")
        df_dx_num = sp.lambdify((x, y), self.df_dx, "numpy")
        return df_dx_num(x_vals, y_vals)

    def dy(self, x_vals, y_vals):
        x, y = sp.symbols("x y")
        df_dy_num = sp.lambdify((x, y), self.df_dy, "numpy")
        return df_dy_num(x_vals, y_vals)


class SurfaceConstraint:
    def __init__(self, surface):
        self.surface: Surface = surface

    def adjust_positions(self, atoms, newpositions):
        x = newpositions[:, 0]
        y = newpositions[:, 1]

        z = self.surface.elevate_to_surface(x, y)

        newpositions[:, 2] = z

    def adjust_forces(self, atoms, forces):
        # Modify the forces to be tangential to the surface
        positions = atoms.get_positions()
        x = positions[:, 0]
        y = positions[:, 1]

        dz_dx = self.surface.dx(x, y)
        dz_dy = self.surface.dy(x, y)
        dz_dz = -1 * np.ones(self.surface.n)

        print
        normals = np.stack((dz_dx, dz_dy, dz_dz), axis=-1)
        normals /= np.linalg.norm(normals, axis=1).reshape(-1, 1)
        # forces -= np.dot(forces, normals) * normals
        forces -= np.einsum("ij,ij->ij", forces, normals) * normals


def setup_atoms(stream, surface, r0=4):
    cell = Cell.fromcellpar([30, 30, 30, 90, 90, 90])
    random_positions = stream.random((surface.n, 3))
    cartesian_positions = np.dot(random_positions, cell)

    atoms = Atoms(
        "Au" * surface.n, positions=cartesian_positions, cell=cell, pbc=(1, 1, 0)
    )
    atoms.calc = RadialPotential(r0=r0)
    constraint = SurfaceConstraint(surface)
    atoms.set_constraint(constraint)

    potential_energy = atoms.get_potential_energy()
    print("Potential energy:", potential_energy, "eV")
    return atoms


def annealing(n, amp, surface, atoms, start_temp, cooling_rate, end_temp):
    # dyn = VelocityVerlet(atoms, timestep=2 * units.fs)
    dyn = Langevin(
        atoms,
        timestep=2 * units.fs,
        temperature_K=10,
        friction=0.001 / units.fs,
    )
    traj_annealing = Trajectory(
        f"trajectories-surfaces/{seed}/langevin-{n}-{amp}-{surface.n}-{start_temp}-{cooling_rate}-{end_temp}.traj",
        "a",
        atoms,
    )

    def wrap_and_write_annealing():
        # atoms.wrap()
        traj_annealing.write(atoms)

    dyn.attach(wrap_and_write_annealing, interval=500)

    iters = round((start_temp - end_temp) / cooling_rate)
    for i in range(iters):
        temp = round(start_temp - cooling_rate * i)
        print(str(n) + ", " + str(i) + f": Setting temp {temp}")
        MaxwellBoltzmannDistribution(
            atoms, temperature_K=(start_temp - cooling_rate * i)
        )
        atoms.wrap()
        dyn.run(10000)
        # printenergy(n, i, atoms)
    print(str(n) + ", " + str(iters + 1) + f": Setting temp {end_temp}")
    MaxwellBoltzmannDistribution(atoms, temperature_K=end_temp)
    dyn.run(15000)
    # printenergy(n, i, atoms)

    local_minimisation = BFGS(atoms)
    traj_bfgs = Trajectory(
        f"trajectories-surfaces/{seed}/bfgs-{n}-{amp}-{surface.n}-{start_temp}-{cooling_rate}-{end_temp}.traj",
        "w",
        atoms,
    )

    def wrap_and_write_bfgs():
        # atoms.wrap()
        traj_bfgs.write(atoms)

    local_minimisation.attach(wrap_and_write_bfgs)
    local_minimisation.run(steps=1000, fmax=0.015)

    optimised_energy = atoms.get_potential_energy()
    print(f"Optimised energy: {optimised_energy}")

    return optimised_energy


def calculate_midpoint(point1, point2):
    return (point1 + point2) / 2


def find_triangle_midpoints(tri, points):
    midpoints = []
    for simplex in tri.simplices:
        for i in range(3):
            point1 = points[simplex[i]]
            point2 = points[
                simplex[(i + 1) % 3]
            ]  # Ensures the last point connects to the first
            midpoint = calculate_midpoint(point1, point2)
            midpoints.append(midpoint)
    return midpoints


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


def make_delaunay_triangulation(traj_path):
    traj = Trajectory(f"/Users/olive/Desktop/bsc thesis/{traj_path}")
    atoms = traj[-1]

    coords = atoms.get_positions()
    points = repeat(3, coords)
    points_2d = points[:, :2]
    tri = Delaunay(points_2d)

    neighbours = find_neighbors(tri, points_2d)
    amplitude = traj_path[13:16]
    return atoms, tri, points, points_2d, neighbours, amplitude


def plot_delaunay(tri, points, points_2d, neighbours, amplitude):
    nine_neighbours = np.array([len(nbrs) == 9 for nbrs in neighbours])
    eight_neighbours = np.array([len(nbrs) == 8 for nbrs in neighbours])
    seven_neighbours = np.array([len(nbrs) == 7 for nbrs in neighbours])
    six_neighbours = np.array([len(nbrs) == 6 for nbrs in neighbours])
    five_neighbours = np.array([len(nbrs) == 5 for nbrs in neighbours])
    four_neighbours = np.array([len(nbrs) == 4 for nbrs in neighbours])
    three_neighbours = np.array([len(nbrs) == 3 for nbrs in neighbours])

    fig = plt.figure(figsize=(12, 6))
    fig.suptitle(
        f"Delaunay triangulation on Sine curve with amplitude {amplitude}",
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
    plt.savefig(f"delaunay-triangulations.png", dpi=300)


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


trajectory_paths = [
    # "30011/bfgs-0-1.0-2000-200-50.traj",
    # "30011/bfgs-0-3.0-2000-200-50.traj",
    # "30011/bfgs-0-5.0-2000-200-50.traj",
    # "20011/bfgs-0-2.0-2000-200-50.traj",
    # "20011/bfgs-1-2.0-2000-200-50.traj",
    # "20011/bfgs-2-2.0-2000-200-50.traj",
    # "30011/bfgs-0-7.0-2000-200-50.traj",
    # "40012/bfgs-28-3-2000-200-30.traj"
    "99993/bfgs-0-3-2000-500-50.traj",
    # "99993/bfgs-1-3-2000-500-50.traj",
    # "99993/bfgs-2-3-2000-500-50.traj",
    # "99993/bfgs-3-3-2000-500-50.traj",
]

for path in trajectory_paths:
    atoms, tri, points, points_2d, neighbours, amplitude = make_delaunay_triangulation(
        path
    )
    print_neighbour_counts(points, neighbours)
    plot_delaunay(tri, points, points_2d, neighbours, amplitude)

    kagome_positions = find_triangle_midpoints(tri, points)
    cell = Cell.fromcellpar([30, 30, 30, 90, 90, 90])

    kagome_atoms = Atoms(
        "H" * len(kagome_positions),
        positions=kagome_positions,
        cell=cell,
        pbc=(1, 1, 0),
    )

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


# n_streams = 50
# n_processes = 40
# seed = 99992
# ss = SeedSequence(seed)
# child_seeds = ss.spawn(n_streams)
# streams = [default_rng(s) for s in child_seeds]

# start_temps = np.arange(600, 3001, 200)
# end_temps = np.arange(10, 71, 10)
# cooling_rates = np.arange(50,201,50)
# densities = [7,8,9]
# # start_temps = [2000]
# # end_temps = [50]
# # cooling_rates = [500]

# if not os.path.exists(f"trajectories-surfaces/{seed}"):
#     os.makedirs(f"trajectories-surfaces/{seed}")


# def sine_surfaces(amplitude, density):
#     return (amplitude, Surface(lambda x,y: amplitude*sp.sin(2 * sp.pi * x / 30) * sp.sin(2 * sp.pi * y / 30), density))

# # surfaces = [sine_surfaces(a) for a in np.arange(1, 10.5, 0.5)]
# surfaces = [sine_surfaces(3, density) for density in densities]

# args = [
#     (nstream, surface, start_temp, cooling_rate, end_temp,)
#     for nstream, surface, start_temp, cooling_rate, end_temp, in product(
#         enumerate(streams), surfaces, start_temps, cooling_rates, end_temps
#     )
# ]

# def launch_parallel(nstream, surface, start_temp, cooling_rate, end_temp):
#     n, stream = nstream
#     amp, surf = surface
#     atoms = setup_atoms(stream, surf)
#     energy = annealing(n, amp, surf, atoms, start_temp, cooling_rate, end_temp)
#     with paropen(f"results-surfaces/surfaces-{seed}.txt", "a") as resfile:
#         print(n, amp, surf.density, start_temp, cooling_rate, end_temp, energy, file=resfile)


# def pool_handler():
#     p = Pool(n_processes)
#     p.starmap(launch_parallel, args)


# if __name__ == "__main__":
#     pool_handler()

#     # 40011 112 streams amp 3
#     # 40012 diff start end temps
#     # 40013 area/8
#     # 40014 diff start end temps again
#     # 40015 same but less cores
#     # 40016 amp 3 area 9
#     # 40017 amp3 area9 get hist
#     # 40018 amp3 bfgs 0.15
