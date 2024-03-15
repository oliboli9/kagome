from ase import Atoms, units
from ase.calculators.emt import EMT
from ase.calculators.lj import LennardJones
from ase.calculators.morse import MorsePotential
from ase.cell import Cell
from ase.constraints import FixedPlane
from ase.md.langevin import Langevin
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from ase.md.verlet import VelocityVerlet
from ase.optimize import BFGS
from ase.visualize import view

import numpy as np

from calculator import RadialPotential

n = 100

cell = Cell.fromcellpar([25, 25, 0, 60, 60, 60])

random_positions = np.random.rand(n, 3)  # n random positions in [0, 1) range

# Convert the random positions to Cartesian coordinates within the unit cell
cartesian_positions = np.dot(random_positions, cell)

# Create an ASE Atoms object for hydrogen atoms
atoms = Atoms("Au" * n, positions=cartesian_positions, cell=cell, pbc=(1, 1, 0))

# print("Atom positions:")
# print(atoms.get_positions())
view(atoms)

atoms.calc = RadialPotential(r0=4.5)
##4.5 -


# Get the potential energy by accessing the calculator's 'get_potential_energy' method
potential_energy = atoms.get_potential_energy()

print("Potential energy:", potential_energy, "eV")

c = FixedPlane(
    indices=[atom.index for atom in atoms],
    direction=[0, 0, 1],
)

atoms.set_constraint(c)

# Optimize the atomic positions using the BFGS algorithm
# optimizer = BFGS(atoms, trajectory="optimisation.traj")  # Save trajectory to file
# optimizer.run(steps=1000)  # Run optimization for 1000 steps


# We want to run MD with constant energy using the VelocityVerlet algorithm.

MaxwellBoltzmannDistribution(atoms, temperature_K=2000)
# dyn = VelocityVerlet(atoms, 2 * units.fs)

"""
1500 - 19.19
500 - 17.59
300 - 18.19
100 - 15.44
50 - 16.40
"""

dyn = Langevin(
    atoms,
    timestep=2.0 * units.fs,
    temperature_K=11,  # temperature in K
    friction=0.001 / units.fs,
)

"""
1000 - 10.45
900 - 11.39, 11.01
800 - 10.06, 11.27, 10.75
700 - 10.67
600 - 9.19
500 - 8.65
400 - 7.96
300 - 7.16
200 - 6.18
100 - 4.83 / 3.64
50 - 4.44 / 3.17
40 - 4.68 / 3.01
30 - 4.39 / 2.98
20 - 4.49 / 3.09
15 - / 2.83
11 - 2.79, 2.45
10 - 4.3 / 1000dyn 2.82
8 - / 2.90
5 - / 2.86
"""


def printenergy(a):
    """Function to print the potential, kinetic and total energy"""
    epot = a.get_potential_energy() / len(a)
    ekin = a.get_kinetic_energy() / len(a)
    print(
        "Energy per atom: Epot = %.3feV  Ekin = %.3feV (T=%3.0fK)  "
        "Etot = %.3feV" % (epot, ekin, ekin / (1.5 * units.kB), epot + ekin)
    )


# Now run the dynamics
printenergy(atoms)
for i in range(20):
    MaxwellBoltzmannDistribution(atoms, temperature_K=(2000 - i * 100))
    dyn.run(1000)
    printenergy(atoms)


atoms.wrap()
# Get the optimized atomic positions and potential energy
optimised_positions = atoms.get_positions()
optimised_energy = atoms.get_potential_energy()

# print("Optimised positions:", optimised_positions)
print("Optimised potential energy:", optimised_energy)

optimised_atoms = Atoms("Au" * n, positions=optimised_positions, cell=cell)
view(optimised_atoms)

# from scipy.spatial import Delaunay

# points = optimised_positions
# tri = Delaunay(points)

# import matplotlib.pyplot as plt

# plt.triplot(points[:, 0], points[:, 1], tri.simplices)
# plt.plot(points[:, 0], points[:, 1], "o")
# plt.show()
