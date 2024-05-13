import matplotlib.pyplot as plt
from ase.io import read
import numpy as np

# Load the trajectory file
traj = read("bfgs.traj", ":")

# Extract data
energies = [atoms.get_potential_energy() for atoms in traj]
fmax = [np.sqrt((atoms.get_forces() ** 2).sum(axis=1)).max() for atoms in traj]

# Plotting
plt.figure(figsize=(8, 5))
# plt.scatter(energies, fmax, color="blue")
plt.plot(energies, fmax, linestyle="-", color="blue")
# plt.title("Maximum Force vs. Potential Energy")
plt.xlabel("Potential Energy (eV)")
plt.ylabel("Maximum Force (eV/Å)")
# plt.ylim(0, 0.25)
plt.grid(True)
plt.show()
