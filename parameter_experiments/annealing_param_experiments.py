import numpy as np
import os
import matplotlib.pyplot as plt
from numpy.random import SeedSequence, default_rng
from itertools import product
from multiprocessing import Pool

from ase.io import Trajectory
from ase.optimize import BFGS
from ase import Atoms, units
from ase.constraints import FixedPlane
from ase.cell import Cell
from ase.md.langevin import Langevin
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from ase.parallel import paropen

from methods.calculator import Calculator, RadialPotential
from methods.delaunay import (
    find_neighbours,
    get_neighbour_counts,
    repeat,
    plot_delaunay,
)
from methods.kagome import Kagome
from methods.surface import PeriodicSurface
from methods.triangulation import SurfaceTriangulation

# from calculator import Calculator, RadialPotential
# from delaunay import (
#     find_neighbours,
#     get_neighbour_counts,
#     repeat,
#     plot_delaunay,
# )
# from kagome import Kagome
# from surface import PeriodicSurface
# from triangulation import SurfaceTriangulation


class AnnealingSimulator:
    def __init__(self, calculator):
        self.calculator: Calculator = calculator

    def setup_atoms(self, stream):
        cell = Cell.fromcellpar([30, 30, 0, 90, 90, 90])
        random_positions = stream.random((100, 3))
        cartesian_positions = np.dot(random_positions, cell)
        atoms = Atoms(
            "Au" * 100,
            positions=cartesian_positions,
            cell=cell,
            pbc=(1, 1, 0),
        )
        atoms.calc = self.calculator

        c = FixedPlane(  # outdated docs??
            indices=[atom.index for atom in atoms],
            direction=[0, 0, 1],
        )
        atoms.set_constraint(c)

        # potential_energy = atoms.get_potential_energy()
        # print("Potential energy:", potential_energy, "eV")
        return atoms

    def anneal(
        self,
        atoms: Atoms,
        start_temp: int,
        cooling_rate: int,
        fmax: float = 0.05,
        timestep: float = 2 * units.fs,
        friction: float = 0.01 / units.fs,
    ):
        """
        Perform simulated annealing with Langevin molecular dynamics on an ASE Atoms object.
        The atoms should have a calculator object attached.
        After the final simulation round, a local minimisation is carried out with BFGS.

        Parameters:

        atoms: ASE Atoms object
            Atoms to anneal
        start_temp: int
            Starting temperature (K) of first simulation
        cooling_rate: int
            Number of cooling steps to take
        fmax: float
            Convergence criterion for BFGS minimisation - maximum force per atom. Default 0.5

        """
        MaxwellBoltzmannDistribution(atoms, temperature_K=start_temp)

        for i in range(1, cooling_rate + 1):
            temperature_K = start_temp * (1 - i / cooling_rate)
            dyn = Langevin(
                atoms,
                timestep=timestep,
                temperature_K=temperature_K,
                friction=friction,
            )
            print(f"Heat bath temp: {temperature_K}")
            dyn.run(5000)

        local_minimisation = BFGS(
            atoms,
            trajectory=(Trajectory("flat_bfgs.traj", mode="w", atoms=atoms)),
        )
        local_minimisation.run(steps=1000, fmax=fmax)
        optimised_energy = atoms.get_potential_energy()
        # print("Optimised energy:", potential_energy, "eV")
        return optimised_energy


n_streams = 1
n_processes = 1
seed = 10003
ss = SeedSequence(seed)
child_seeds = ss.spawn(n_streams)
streams = [default_rng(s) for s in child_seeds]


# ns = np.linspace(10, 200, 191)
ns = [100]
# frictions = [i / units.fs for i in np.linspace(0.001, 0.1, 100)]
frictions = [2 / units.fs]

args = [
    (
        nstream,
        int(no_atoms),
        friction,
    )
    for nstream, no_atoms, friction, in product(enumerate(streams), ns, frictions)
]

if not os.path.exists(f"periodic/trajectories/{seed}"):
    os.makedirs(f"periodic/trajectories/{seed}")
if not os.path.exists(f"periodic/weave_patterns/{seed}"):
    os.makedirs(f"periodic/weave_patterns/{seed}")

results_filename = f"periodic/results/{seed}.txt"


def launch_parallel(nstream, no_atoms, friction):
    n, stream = nstream
    r0 = 30 * np.sqrt(2) / np.sqrt(no_atoms)
    print(r0)
    calculator = RadialPotential(r0=r0)
    annealing = AnnealingSimulator(calculator)
    atoms = annealing.setup_atoms(stream)
    energy = annealing.anneal(atoms, 1500, 10, fmax=0.0005, friction=friction)
    # traj = Trajectory("flat_bfgs.traj")
    # atoms = traj[-1]
    coords = atoms.get_positions()
    points = repeat(3, coords)
    triangulation = SurfaceTriangulation()
    simplices = triangulation.triangulate(points)
    neighbours = find_neighbours(simplices, points)
    plot_delaunay(simplices, points, neighbours)
    kagome = Kagome(simplices, points, r0=r0 / 2, periodic=True)
    fig = plt.figure()
    ax = fig.add_subplot(121)
    # ax.view_init(elev=90, azim=0)
    ax2 = fig.add_subplot(122)
    # ax2.view_init(elev=90, azim=0)
    # ax3 = fig.add_subplot(133, projection="3d")
    # ax3.view_init(elev=90, azim=0)

    kagome.plot_weavers_periodic(ax, projection_2d=True)
    kagome.straighten_weavers(steps=50, fmax=0.01)
    kagome.plot_weavers_periodic(ax2, projection_2d=True)
    # kagome.straighten_weavers(steps=15, fmax=0.01)
    # kagome.plot_weavers_periodic(ax3)
    plt.show()

    neighbours = find_neighbours(simplices, points)
    neighbour_counts = get_neighbour_counts(points, neighbours)

    with paropen(results_filename, "a") as resfile:
        print(
            n,
            r0,
            friction,
            energy,
            neighbour_counts,
            file=resfile,
        )


def pool_handler():
    p = Pool(n_processes)
    with open(results_filename, "a") as resfile:
        print(
            "n r0 friction energy neighbours",
            file=resfile,
        )
    p.starmap(launch_parallel, args)


if __name__ == "__main__":
    pool_handler()
