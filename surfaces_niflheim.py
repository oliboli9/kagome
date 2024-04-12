from itertools import product
import os
import numpy as np
import sympy as sp
from numpy.random import SeedSequence, default_rng
from multiprocessing import Pool

from ase.parallel import paropen

from annealing import setup_atoms, anneal
from delaunay import (
    make_delaunay_triangulation,
    get_neighbour_counts,
    plot_delaunay,
    repeat,
)
from surface import PeriodicSurface, SurfaceConstraint


n_streams = 1
n_processes = 1
seed = 10020
ss = SeedSequence(seed)
child_seeds = ss.spawn(n_streams)
streams = [default_rng(s) for s in child_seeds]

# start_temps = np.arange(600, 3001, 200)
# end_temps = np.arange(10, 71, 10)
# cooling_rates = np.arange(50,201,50)
# densities = [7,8,9]
start_temps = [2000]
end_temps = [50]
cooling_rates = [500]
densities = np.arange(10, 125, 1)
# densities = [110]

if not os.path.exists(f"trajectories/{seed}"):
    os.makedirs(f"trajectories/{seed}")

# if not os.path.exists(f"results/{seed}"):
#     os.makedirs(f"results/{seed}")

if not os.path.exists(f"delaunay/{seed}"):
    os.makedirs(f"delaunay/{seed}")


def sine_surfaces(amplitude, density):
    return (
        amplitude,
        PeriodicSurface(
            lambda x, y: amplitude
            * sp.sin(2 * sp.pi * x / 30)
            * sp.sin(2 * sp.pi * y / 30),
            n=density,
        ),
    )


# surfaces = [sine_surfaces(a) for a in np.arange(1, 10.5, 0.5)]
surfaces = [sine_surfaces(5, density) for density in densities]
# surfaces = [sine_surfaces(4,  9)]
file = f"results/{seed}.txt"

args = [
    (
        nstream,
        surface,
        start_temp,
        cooling_rate,
        end_temp,
    )
    for nstream, surface, start_temp, cooling_rate, end_temp, in product(
        enumerate(streams), surfaces, start_temps, cooling_rates, end_temps
    )
]


def launch_parallel(nstream, surface, start_temp, cooling_rate, end_temp):
    n, stream = nstream
    amp, surf = surface
    r0 = 32 / surf.density  # in flat cell found optimal to be density 1/9 and r0=4
    atoms = setup_atoms(stream, surf, r0=r0)
    energy = anneal(seed, n, amp, surf, atoms, start_temp, cooling_rate, end_temp)
    coords = atoms.get_positions()
    points = repeat(3, coords)
    tri, neighbours = make_delaunay_triangulation(points)
    neighbour_counts = get_neighbour_counts(points, neighbours)
    with paropen(file, "a") as resfile:
        print(
            n,
            amp,
            surf.density,
            surf.n,
            r0,
            start_temp,
            cooling_rate,
            end_temp,
            energy,
            neighbour_counts,
            file=resfile,
        )
    plot_delaunay(
        tri,
        points,
        neighbours,
        f"delaunay/{seed}/{n}-{amp}-{surf.n}-{start_temp}-{cooling_rate}-{end_temp}",
    )


def pool_handler():
    p = Pool(n_processes)
    with open(file, "a") as resfile:
        print(
            f"Description: seed={seed} density adjusted r0, {n_streams} streams, fmax0.001",
            file=resfile,
        )
        print(
            "n amp surf.density surf.n r0 start_temp cooling_rate end_temp energy neighbours",
            file=resfile,
        )
    p.starmap(launch_parallel, args)


if __name__ == "__main__":
    pool_handler()

# amp1=101
# amp3=109
# amp4=116
# amp5=124
# amp7 fmax 0.065

# 10020 amp3 10-124atoms 10streams r0 32/density (3.55)
