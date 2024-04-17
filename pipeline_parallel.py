import numpy as np
import os
import matplotlib.pyplot as plt
from numpy.random import SeedSequence, default_rng
from multiprocessing import Pool
from itertools import product

from ase.parallel import paropen

from annealing import AnnealingSimulator
from calculator import RadialPotential
from delaunay import get_neighbour_counts_3d, find_neighbours_3d
from kagome import Kagome
from surface import TorusSurface
from triangulation import NonConvexTriangulation


n_streams = 4
n_processes = 4
seed = 99999
ss = SeedSequence(seed)
child_seeds = ss.spawn(n_streams)
streams = [default_rng(s) for s in child_seeds]


ns = np.linspace(10, 200, 39)

args = [
    (
        nstream,
        int(no_atoms),
    )
    for nstream, no_atoms, in product(enumerate(streams), ns)
]

if not os.path.exists(f"torus/trajectories/{seed}"):
    os.makedirs(f"torus/trajectories/{seed}")
if not os.path.exists(f"torus/results/{seed}"):
    os.makedirs(f"torus/results/{seed}")
if not os.path.exists(f"torus/weave_patterns/{seed}"):
    os.makedirs(f"torus/weave_patterns/{seed}")

results_filename = f"torus/results/{seed}.txt"


def launch_parallel(nstream, no_atoms):
    n, stream = nstream
    traj_filename = f"torus/trajectories/{seed}/stream_{n}_atoms_{no_atoms}"
    fig_filename = f"torus/weave_patterns/{seed}/stream_{n}_atoms_{no_atoms}"
    surface = TorusSurface(5, 2, (15, 15, 15), n=no_atoms)
    r0 = surface.density / 2
    calculator = RadialPotential(r0=r0)
    annealing = AnnealingSimulator(surface, calculator)
    atoms = annealing.setup_atoms(stream)
    with open(traj_filename, "w"):
        pass
    energy = annealing.anneal(atoms, 2500, 100, 500, traj_path_md=f"{traj_filename}")
    coords = atoms.get_positions()
    triangulation = NonConvexTriangulation()
    simplices = triangulation.triangulate(coords, 0.3)
    neighbours = find_neighbours_3d(simplices, coords)
    neighbour_counts = get_neighbour_counts_3d(neighbours)
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
