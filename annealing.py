import numpy as np
from typing import Optional

from ase import Atoms, units
from ase.cell import Cell
from ase.io.trajectory import Trajectory
from ase.md.langevin import Langevin
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from ase.md.verlet import VelocityVerlet
from ase.optimize import BFGS

from calculator import Calculator
from surface import Surface, SurfaceConstraint


class AnnealingSimulator:
    def __init__(self, surface, calculator):
        self.surface: Surface = surface
        self.calculator: Calculator = calculator

    def setup_atoms(self, stream):
        cell = Cell.fromcellpar([30, 30, 30, 90, 90, 90])
        random_positions = stream.random((self.surface.n, 3))
        cartesian_positions = np.dot(random_positions, cell)
        atoms = Atoms(
            "Au" * self.surface.n,
            positions=cartesian_positions,
            cell=cell,
            pbc=(0, 0, 0),
        )
        atoms.calc = self.calculator
        constraint = SurfaceConstraint(self.surface)
        surface_positions = self.surface.elevate_to_surface(atoms.get_positions())
        atoms.set_positions(surface_positions)
        atoms.set_constraint(constraint)

        # potential_energy = atoms.get_potential_energy()
        # print("Potential energy:", potential_energy, "eV")
        return atoms

    def anneal(
        self,
        atoms: Atoms,
        start_temp: int,
        end_temp: int,
        cooling_rate: int,
        fmax: float = 0.05,
        traj_path_md: Optional[str] = None,
        traj_path_bfgs: Optional[str] = None,
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
        end_temp: int
            Starting temperature (K) of final simulation
        cooling_rate: int
            Temperature decrease (K) between each round of annealing
        fmax: float
            Convergence criterion for BFGS minimisation - maximum force per atom. Default 0.5
        traj_path_md: str
            Path to save .traj file of molecular dynamics
        traj_path_bfgs: str
            Path to save .traj file of BFGS minimisation

        """
        MaxwellBoltzmannDistribution(atoms, temperature_K=start_temp)

        iters = round((start_temp - end_temp) / cooling_rate)
        for i in range(1, iters + 1):
            temperature_K = start_temp - cooling_rate * i
            dyn = Langevin(
                atoms,
                timestep=2 * units.fs,
                temperature_K=temperature_K,
                friction=0.01 / units.fs,
            )
            if traj_path_md is not None:
                dyn.attach(
                    Trajectory(traj_path_md, mode="a", atoms=atoms), interval=100
                )
            print(f"Heat bath temp: {temperature_K}")
            atoms.wrap()
            dyn.run(1000)

        local_minimisation = BFGS(
            atoms,
            trajectory=(
                Trajectory(traj_path_bfgs, mode="w", atoms=atoms)
                if traj_path_bfgs is not None
                else None
            ),
        )
        local_minimisation.run(steps=100, fmax=fmax)
        optimised_energy = atoms.get_potential_energy()
        # print("Optimised energy:", potential_energy, "eV")
        return optimised_energy
