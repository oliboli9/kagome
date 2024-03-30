import numpy as np

from ase import Atoms, units
from ase.cell import Cell
from ase.io.trajectory import Trajectory
from ase.md.langevin import Langevin
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from ase.md.verlet import VelocityVerlet
from ase.optimize import BFGS

from calculator import RadialPotential
from surface import PeriodicSurface, SurfaceConstraint


def setup_periodic_atoms(stream, surface, r0=4):
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


def anneal(seed, n, amp, surface, atoms, start_temp, cooling_rate, end_temp, fmax=0.05):
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

    # dyn.attach(wrap_and_write_annealing, interval=500)

    iters = round((start_temp - end_temp) / cooling_rate)
    for i in range(iters):
        temp = round(start_temp - cooling_rate * i)
        print(str(n) + ", " + str(i) + f": Setting temp {temp}")
        MaxwellBoltzmannDistribution(
            atoms, temperature_K=(start_temp - cooling_rate * i)
        )
        atoms.wrap()
        dyn.run(10000)
    print(str(n) + ", " + str(iters + 1) + f": Setting temp {end_temp}")
    MaxwellBoltzmannDistribution(atoms, temperature_K=end_temp)
    dyn.run(15000)

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
    local_minimisation.run(steps=10000, fmax=fmax)

    optimised_energy = atoms.get_potential_energy()
    print(f"Optimised energy: {optimised_energy}")

    return optimised_energy
