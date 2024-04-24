from ase import Atoms
from ase.io.trajectory import Trajectory
from pyscf import gto
import spglib

# Define an ASE Atoms object
traj = Trajectory("torus.traj")
atoms = traj[-1]
print(atoms)
# Get the symmetry dataset
symmetry_dataset = spglib.get_symmetry_dataset(
    (atoms.cell, atoms.positions, atoms.numbers), symprec=1
)

# Retrieve the point group
point_group = symmetry_dataset["pointgroup"]

print(f"The point group is: {point_group}")
