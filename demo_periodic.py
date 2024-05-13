import numpy as np
import sympy as sp
import os
import matplotlib.pyplot as plt
from numpy.random import SeedSequence, default_rng
from itertools import product
from multiprocessing import Pool

from ase.io import Trajectory
from ase.optimize import BFGS
from ase.visualize import view
from ase.parallel import paropen

from methods.annealing import AnnealingSimulator
from methods.calculator import RadialPotential
from methods.delaunay import (
    find_neighbours,
    get_neighbour_counts,
    repeat,
    plot_delaunay,
)
from methods.kagome import Kagome
from methods.surface import PeriodicSurface
from methods.triangulation import SurfaceTriangulation

from plotting.plot_periodic_surface import plot_surf_with_atoms

n_streams = 1
n_processes = 1
seed = 30001
ss = SeedSequence(seed)
child_seeds = ss.spawn(n_streams)
streams = [default_rng(s) for s in child_seeds]

ns = np.linspace(10, 200, 191)
# ns = [200]

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
        lambda x, y: 3 * sp.sin(2 * sp.pi * x / 30) * sp.sin(2 * sp.pi * y / 30),
        n=no_atoms,
    )
    r0 = 30 * np.sqrt(2) / np.sqrt(no_atoms)
    calculator = RadialPotential(r0=r0)
    annealing = AnnealingSimulator(surface, calculator)
    atoms = annealing.setup_atoms(stream)
    with open(traj_filename, "w"):
        pass
    energy = annealing.anneal(
        atoms,
        2000,
        10,
        traj_path_md=f"{traj_filename}",
        fmax=0.0005,
    )

    coords = atoms.get_positions()
    points = repeat(3, coords)
    triangulation = SurfaceTriangulation()
    simplices = triangulation.triangulate(points)
    neighbours = find_neighbours(simplices, points)
    neighbour_counts = get_neighbour_counts(points, neighbours)

    plot_delaunay(simplices, points, neighbours)
    plt.savefig("periodic_delaunay")

    with paropen(results_filename, "a") as resfile:
        print(
            n,
            surface.density,
            surface.n,
            r0,
            energy,
            file=resfile,
        )
    kagome = Kagome(simplices, points, r0=r0 / 2, surface=surface, periodic=True)
    fig = plt.figure()
    # ax = fig.add_subplot(131, projection="3d")
    # ax.view_init(elev=90, azim=0)
    # ax2 = fig.add_subplot(132, projection="3d")
    # ax2.view_init(elev=90, azim=0)
    # ax3 = fig.add_subplot(133, projection="3d")
    # ax3.view_init(elev=90, azim=0)
    ax = fig.add_subplot(131)
    ax2 = fig.add_subplot(132)
    ax3 = fig.add_subplot(133)

    kagome.plot_weavers_periodic(ax, projection_2d=True)
    kagome.straighten_weavers(steps=15, fmax=0.01)
    kagome.plot_weavers_periodic(ax2, projection_2d=True)
    kagome.straighten_weavers(steps=15, fmax=0.01)
    kagome.plot_weavers_periodic(ax3, projection_2d=True)


def pool_handler():
    p = Pool(n_processes)
    with open(results_filename, "a") as resfile:
        print(
            "n surf.density surf.n r0 energy",
            file=resfile,
        )
    p.starmap(launch_parallel, args)


if __name__ == "__main__":
    pool_handler()
