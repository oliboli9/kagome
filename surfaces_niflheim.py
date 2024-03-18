from itertools import product
import math
import os
import numpy as np
import sympy as sp
from numpy.random import SeedSequence, default_rng
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from multiprocessing import Pool

from functools import partial

from ase import Atoms, units
from ase.cell import Cell
from ase.constraints import FixedPlane
from ase.io.trajectory import Trajectory
from ase.md.langevin import Langevin
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from ase.md.verlet import VelocityVerlet
from ase.optimize import BFGS
from ase.parallel import paropen
from ase.visualize import view

from calculator import RadialPotential
from surface import Surface, SurfaceConstraint
from delaunay import make_delaunay_triangulation, print_neighbour_counts, plot_delaunay


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
        f"trajectories/{seed}/langevin-{n}-{amp}-{surface.n}-{start_temp}-{cooling_rate}-{end_temp}.traj",
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
        f"trajectories/{seed}/bfgs-{n}-{amp}-{surface.n}-{start_temp}-{cooling_rate}-{end_temp}.traj",
        "w",
        atoms,
    )

    def wrap_and_write_bfgs():
        # atoms.wrap()
        traj_bfgs.write(atoms)

    local_minimisation.attach(wrap_and_write_bfgs)
    local_minimisation.run(steps=10000, fmax=0.001)

    optimised_energy = atoms.get_potential_energy()
    print(f"Optimised energy: {optimised_energy}")

    return optimised_energy


n_streams = 10
n_processes = 10
seed = 10005
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
densities = np.arange(108, 115, 1)
# densities=[100]

if not os.path.exists(f"trajectories/{seed}"):
    os.makedirs(f"trajectories/{seed}")

# if not os.path.exists(f"results/{seed}"):
#     os.makedirs(f"results/{seed}")

if not os.path.exists(f"delaunay/{seed}"):
    os.makedirs(f"delaunay/{seed}")


def sine_surfaces(amplitude, density):
    return (
        amplitude,
        Surface(
            lambda x, y: amplitude
            * sp.sin(2 * sp.pi * x / 30)
            * sp.sin(2 * sp.pi * y / 30),
            n=density,
        ),
    )


# surfaces = [sine_surfaces(a) for a in np.arange(1, 10.5, 0.5)]
surfaces = [sine_surfaces(3, density) for density in densities]
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
    r0 = 36 / surf.density  # in flat cell found optimal to be density 1/9 and r0=4
    atoms = setup_atoms(stream, surf, r0=r0)
    energy = annealing(n, amp, surf, atoms, start_temp, cooling_rate, end_temp)
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
            file=resfile,
        )
    atoms, tri, points, points_2d, neighbours, amplitude = make_delaunay_triangulation(
        f"trajectories/{seed}/bfgs-{n}-{amp}-{surf.n}-{start_temp}-{cooling_rate}-{end_temp}.traj"
    )
    print_neighbour_counts(points, neighbours)
    plot_delaunay(
        tri,
        points,
        points_2d,
        neighbours,
        amplitude,
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
            "n amp surf.density surf.n r0 start_temp cooling_rate end_temp energy",
            file=resfile,
        )
    p.starmap(launch_parallel, args)


if __name__ == "__main__":
    pool_handler()

#amp1=101
#amp3=109
#amp5=124