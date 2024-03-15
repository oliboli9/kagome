import numpy as np

from ase.calculators.calculator import Calculator
from ase.neighborlist import neighbor_list


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
        properties=["energy"],
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
