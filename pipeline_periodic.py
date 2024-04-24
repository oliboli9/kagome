import numpy as np
import sympy as sp
import spglib
import os
import matplotlib.pyplot as plt
from numpy.random import SeedSequence, default_rng
from itertools import product
from multiprocessing import Pool

from ase.io import Trajectory
from ase.optimize import BFGS
from ase.spacegroup.symmetrize import check_symmetry
from ase.visualize import view
from ase.parallel import paropen

from annealing import AnnealingSimulator
from calculator import RadialPotential
from delaunay import find_neighbors, get_neighbour_counts
from kagome import Kagome
from surface import PeriodicSurface
from triangulation import SurfaceTriangulation

# surface = PeriodicSurface(
#     lambda x, y: 0.1 * sp.sin(2 * sp.pi * x / 30) * sp.sin(2 * sp.pi * y / 30),
#     n=100,
# )
# r0 = surface.density / 2
# print(r0)
# calculator = RadialPotential(r0=r0)

# n_streams = 1
# n_processes = 1
# seed = 10020
# ss = SeedSequence(seed)
# child_seeds = ss.spawn(n_streams)
# streams = [default_rng(s) for s in child_seeds]

# filename = "periodic.traj"
# # #### Annealing
# annealing = AnnealingSimulator(surface, calculator)
# atoms = annealing.setup_atoms(streams[0])
# with open(filename, "w") as file:
#     pass
# annealing.anneal(atoms, 2500, 100, 300, traj_path_md=f"{filename}", fmax=0.001)
# atoms.wrap()
# view(atoms)
# symmetry_dataset = spglib.get_symmetry_dataset(
#     (atoms.cell, atoms.positions, atoms.numbers), symprec=1
# )
# space_group = symmetry_dataset["international"]
# space_group_number = symmetry_dataset["number"]
# print(f"The space group is: {space_group} (No. {space_group_number})")

# # view(atoms)
# # local_minimisation = BFGS(atoms, trajectory="polar2.traj")

# # local_minimisation.run(steps=100, fmax=0.01)

# coords = atoms.get_positions()
# np.savetxt("torus_points.csv", coords, delimiter=",", fmt="%f")


# #### Triangulation - matlab
# # simplices = np.loadtxt("simplices2.csv", delimiter=",", dtype="int") - 1
# # coords = np.loadtxt("alphapoints2.csv", delimiter=",", dtype="float")
# # coords = np.loadtxt("torus_points.csv", delimiter=",", dtype="float")
# def repeat(n, points):
#     repeated_points = []
#     for i in range(n):  # Repeat in x-direction
#         for j in range(n):  # Repeat in y-direction
#             new_points = points.copy()
#             new_points[:, 0] += 30 * i
#             new_points[:, 1] += 30 * j
#             repeated_points.append(new_points)

#     repeated_points = np.vstack(repeated_points)
#     return repeated_points


# traj = Trajectory(f"{filename}")
# atoms = traj[-1]
# coords = atoms.get_positions()
# repeats = 3
# coords = repeat(repeats, coords)
# triangulation = SurfaceTriangulation()
# simplices = triangulation.triangulate(coords)

# # fig = plt.figure()
# # ax = fig.add_subplot(111, projection="3d")
# # ax.scatter(coords[:, 0], coords[:, 1], coords[:, 2], c="b", marker="o")

# # # Plot triangles
# # for simplex in simplices:
# #     simplex_points = coords[simplex]
# #     simplex_points = np.append(
# #         simplex_points, [simplex_points[0]], axis=0
# #     )  # Connect back to the first point
# #     ax.plot(simplex_points[:, 0], simplex_points[:, 1], simplex_points[:, 2], "k-")

# # ax.set_xlabel("X")
# # ax.set_ylabel("Y")
# # ax.set_zlabel("Z")
# # ax.set_title("Delaunay Triangulation")
# # plt.show()


# #### Kagome
# kagome = Kagome(simplices, coords, surface=surface, r0=r0 / 2, periodic=True)
# fig = plt.figure()
# ax = fig.add_subplot(121, projection="3d")
# ax.view_init(elev=90, azim=0)
# ax2 = fig.add_subplot(122, projection="3d")
# ax2.view_init(elev=90, azim=0)

# kagome.plot_weavers_periodic(ax)
# kagome.straighten_weavers()
# kagome.plot_weavers_periodic(ax2)
# check_symmetry(kagome.atoms, symprec=0.001, verbose=True)
# plt.show()


n_streams = 3
n_processes = 56
seed = 10000
ss = SeedSequence(seed)
child_seeds = ss.spawn(n_streams)
streams = [default_rng(s) for s in child_seeds]


ns = np.linspace(10, 200, 191)

args = [
    (
        nstream,
        int(no_atoms),
    )
    for nstream, no_atoms, in product(enumerate(streams), ns)
]

if not os.path.exists(f"periodic/trajectories/{seed}"):
    os.makedirs(f"periodic/trajectories/{seed}")
if not os.path.exists(f"periodic/weave_patterns/{seed}"):
    os.makedirs(f"periodic/weave_patterns/{seed}")

results_filename = f"periodic/results/{seed}.txt"


def launch_parallel(nstream, no_atoms):
    n, stream = nstream
    traj_filename = f"periodic/trajectories/{seed}/stream_{n}_atoms_{no_atoms}"
    fig_filename = f"periodic/weave_patterns/{seed}/stream_{n}_atoms_{no_atoms}"
    surface = PeriodicSurface(
        lambda x, y: 0.1 * sp.sin(2 * sp.pi * x / 30) * sp.sin(2 * sp.pi * y / 30),
        n=no_atoms,
    )
    r0 = surface.density / 2
    calculator = RadialPotential(r0=r0)
    annealing = AnnealingSimulator(surface, calculator)
    atoms = annealing.setup_atoms(stream)
    with open(traj_filename, "w"):
        pass
    energy = annealing.anneal(atoms, 2500, 100, 500, traj_path_md=f"{traj_filename}")
    coords = atoms.get_positions()
    triangulation = SurfaceTriangulation()
    simplices = triangulation.triangulate(coords)  # 1/r0 + a little bit (0.05?)
    neighbours = find_neighbors(simplices, coords)
    neighbour_counts = get_neighbour_counts(neighbours)

    with paropen(results_filename, "a") as resfile:
        print(
            n,
            surface.density,
            surface.n,
            r0,
            energy,
            neighbour_counts,
            file=resfile,
        )
    kagome = Kagome(simplices, coords)
    fig = plt.figure()
    ax = fig.add_subplot(121, projection="3d")
    ax.view_init(elev=90, azim=0)
    ax2 = fig.add_subplot(122, projection="3d")
    ax2.view_init(elev=90, azim=0)

    kagome.plot_weavers(ax)
    kagome.straighten_weavers()
    kagome.plot_weavers(ax2)
    plt.savefig(f"{fig_filename}")


def pool_handler():
    p = Pool(n_processes)
    with open(results_filename, "a") as resfile:
        print(
            "n surf.density surf.n r0 energy neighbours",
            file=resfile,
        )
    p.starmap(launch_parallel, args)


if __name__ == "__main__":
    pool_handler()
