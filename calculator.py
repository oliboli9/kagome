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
        Range of quadratic potential, default 4.0
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
        # dhat = (D / d[:, None]).T
        epsilon = 1e-10
        d_safe = np.where(d == 0, epsilon, d)
        dhat = (D / d_safe[:, None]).T

        dd = 1 - d / r0
        E = V0 * dd**2
        dE = preF * dd * dhat
        energy = 0.5 * E.sum()

        F = dE.T
        for dim in range(3):
            forces[:, dim] = np.bincount(i, weights=F[:, dim], minlength=len(atoms))

        self.results["energy"] = energy
        self.results["forces"] = forces


class KagomeRadialPotential(Calculator):
    """
    Kagome potential:
    The kagome atoms are points along kagome weavers. (Midpoints of edges in triangulation of desired surface)
    Energy is minimised for straight weavers and increases as weavers are made to bend
    """

    implemented_properties = ["energy", "forces"]
    default_parameters = {
        "V0": 1.0,
        "r0": 1,
        "K0": 10000.0,
    }  # this config makes the radial and kagome force have the same order of magnitude

    nolabel = True

    def __init__(self, **kwargs):
        """
        Parameters
        ----------
        V0: float
            Energy scale, default 1.0
        r0: float
            Range of quadratic potential, default 1.0
        K0: float
            Kagome energy scale, default 10000.0
        neighbour_dict: dict (required)
            A dictionary where for each kagome atom its 4 kagome neighbours are stored as pairs that lie on the same weaver
        """
        Calculator.__init__(self, **kwargs)

        if "neighbour_dict" not in kwargs:
            raise ValueError(
                "The 'neighbour_dict' parameter must be provided for KagomePotential"
            )
        self.neighbour_dict = kwargs["neighbour_dict"]

    def calculate(
        self,
        atoms=None,
        properties=["energy", "forces"],
        system_changes=["positions", "numbers", "cell", "pbc", "charges", "magmoms"],
    ):
        Calculator.calculate(self, atoms, properties, system_changes)

        V0 = self.parameters.V0
        r0 = self.parameters.r0
        K0 = self.parameters.K0

        forces = np.zeros((len(self.atoms), 3))
        preF = -2 * V0 / r0

        i, j, d, D = neighbor_list("ijdD", atoms, r0)
        # dhat = (D / d[:, None]).T
        epsilon = 1e-10
        d_safe = np.where(d == 0, epsilon, d)
        dhat = (D / d_safe[:, None]).T

        dd = 1 - d / r0
        E = V0 * dd**2
        dE = preF * dd * dhat
        energy = 0.5 * E.sum()

        F = dE.T
        for dim in range(3):
            forces[:, dim] = np.bincount(i, weights=F[:, dim], minlength=len(atoms))

        neighbour_dict = self.neighbour_dict
        positions = atoms.get_positions()
        # forces = np.zeros((len(atoms), 3))
        # energy = 0
        # print(f"radial force:{forces[0]}")

        for atom_index, neighbours in neighbour_dict.items():
            if len(neighbours) == 2:
                n1, n2 = neighbours[0]
                n3, n4 = neighbours[1]
                pairs = [(n1, n3), (n2, n4)]
                for n1, n2 in pairs:
                    vec1 = positions[n1] - positions[atom_index]
                    vec2 = positions[n2] - positions[atom_index]

                    norm_vec1 = np.linalg.norm(vec1)
                    norm_vec2 = np.linalg.norm(vec2)
                    vec1_normalized = vec1 / norm_vec1
                    vec2_normalized = vec2 / norm_vec2

                    dot_product = np.dot(vec1_normalized, vec2_normalized)
                    energy_contribution = K0 * (1 - dot_product)
                    energy += energy_contribution

                    # Calculate force contributions
                    # Derivative of dot product w.r.t. positions
                    force_direction1 = (
                        vec2_normalized - dot_product * vec1_normalized
                    ) / norm_vec1
                    force_direction2 = (
                        vec1_normalized - dot_product * vec2_normalized
                    ) / norm_vec2

                    forces[atom_index] += K0 * (force_direction1 + force_direction2)
                    forces[n1] -= K0 * force_direction1
                    forces[n2] -= K0 * force_direction2

                    # print(f"kagome force:{[force_direction1, force_direction2]}")

        self.results["energy"] = energy
        self.results["forces"] = forces


class KagomePotential(Calculator):
    """
    Kagome potential:
    The kagome atoms are points along kagome weavers. (Midpoints of edges in triangulation of desired surface)
    Energy is minimised for straight weavers and increases as weavers are made to bend
    """

    implemented_properties = ["energy", "forces"]
    default_parameters = {"V0": 1.0}

    nolabel = True

    def __init__(self, **kwargs):
        """
        Parameters
        ----------
        V0: float
            Energy scale, default 1.0
        neighbour_dict: dict (required)
            A dictionary where for each kagome atom its 4 kagome neighbours are stored as pairs that lie on the same weaver
        """
        Calculator.__init__(self, **kwargs)

        if "neighbour_dict" not in kwargs:
            raise ValueError(
                "The 'neighbour_dict' parameter must be provided for KagomePotential"
            )
        self.neighbour_dict = kwargs["neighbour_dict"]

    def calculate(
        self,
        atoms=None,
        properties=["energy", "forces"],
        system_changes=["positions", "numbers", "cell", "pbc", "charges", "magmoms"],
    ):
        Calculator.calculate(self, atoms, properties, system_changes)

        V0 = self.parameters.V0
        neighbour_dict = self.neighbour_dict
        positions = atoms.get_positions()
        forces = np.zeros((len(atoms), 3))
        energy = 0

        for atom_index, neighbours in neighbour_dict.items():
            if len(neighbours) == 2:
                n1, n2 = neighbours[0]
                n3, n4 = neighbours[1]
                pairs = [(n1, n3), (n2, n4)]
                for n1, n2 in pairs:
                    vec1 = positions[n1] - positions[atom_index]
                    vec2 = positions[n2] - positions[atom_index]

                    norm_vec1 = np.linalg.norm(vec1)
                    norm_vec2 = np.linalg.norm(vec2)
                    vec1_normalized = vec1 / norm_vec1
                    vec2_normalized = vec2 / norm_vec2

                    dot_product = np.dot(vec1_normalized, vec2_normalized)
                    energy_contribution = V0 * (1 - dot_product)
                    energy += energy_contribution

                    # Calculate force contributions
                    # Derivative of dot product w.r.t. positions
                    force_direction1 = (
                        vec2_normalized - dot_product * vec1_normalized
                    ) / norm_vec1
                    force_direction2 = (
                        vec1_normalized - dot_product * vec2_normalized
                    ) / norm_vec2

                    forces[atom_index] += V0 * (force_direction1 + force_direction2)
                    forces[n1] -= V0 * force_direction1
                    forces[n2] -= V0 * force_direction2

        self.results["energy"] = energy
        self.results["forces"] = forces
