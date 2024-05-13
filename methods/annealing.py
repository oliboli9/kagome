import numpy as np
from typing import Optional

from ase import Atoms, units
from ase.cell import Cell
from ase.io.trajectory import Trajectory
from ase.md.langevin import Langevin
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from ase.optimize import BFGS
from ase.visualize import view

from methods.calculator import Calculator
from methods.surface import Surface, SurfaceConstraint


class AnnealingSimulator:
    def __init__(self, surface, calculator):
        self.surface: Surface = surface
        self.calculator: Calculator = calculator

    def setup_atoms(self, stream):
        """
        Initialise atoms in 30x30x30 cube cell for annealing simulation.

        Parameters:
        ----------
        stream: numpy Random Number Generator
            Seeds initial random positions of atoms

        Returns:
        ----------
        ASE Atoms object projected and constrained to the surface defined in the class, with a Calculator object attached
        """
        cell = Cell.fromcellpar([30, 30, 30, 90, 90, 90])
        random_positions = stream.random((self.surface.n, 3))
        cartesian_positions = np.dot(random_positions, cell)
        atoms = Atoms(
            "Au" * self.surface.n,
            positions=cartesian_positions,
            cell=cell,
            pbc=(1, 1, 0),
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
        ----------
        atoms: ASE Atoms object
            Atoms to anneal
        start_temp: int
            Starting temperature (K) of first simulation
        cooling_rate: int
            Number of cooling steps to take
        fmax: float
            Convergence criterion for BFGS minimisation - maximum force per atom. Default 0.05
        traj_path_md: str
            Path to save .traj file of molecular dynamics
        traj_path_bfgs: str
            Path to save .traj file of BFGS minimisation

        Returns:
        ----------
        energy: float
            Potential energy of atoms after annealing

        """
        MaxwellBoltzmannDistribution(atoms, temperature_K=start_temp)

        for i in range(1, cooling_rate + 1):
            temperature_K = start_temp * (1 - i / cooling_rate)
            dyn = Langevin(
                atoms,
                timestep=2 * units.fs,
                temperature_K=temperature_K,
                friction=0.01 / units.fs,
            )
            if traj_path_md is not None:
                dyn.attach(
                    Trajectory(traj_path_md, mode="a", atoms=atoms), interval=50
                )  # write every 50th frame to prevent .traj file becoming too large
            print(f"Heat bath temp: {temperature_K}")
            atoms.wrap()
            for i in range(100):
                dyn.run(50)
                atoms.wrap()  # wrap atoms to cell size so periodic surfaces are visualised accurately

        local_minimisation = BFGS(
            atoms,
            trajectory=(
                Trajectory(traj_path_bfgs, mode="w", atoms=atoms)
                if traj_path_bfgs is not None
                else None
            ),
        )
        local_minimisation.run(steps=1000, fmax=fmax)
        optimised_energy = atoms.get_potential_energy()
        # print("Optimised energy:", potential_energy, "eV")
        return optimised_energy
