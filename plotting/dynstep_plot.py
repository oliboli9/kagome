import matplotlib.pyplot as plt
from ase.io import read

# Read the trajectory file
traj = read("99998/md_100.traj", ":")

# Extract total energy and frame number
reference_image_number = 400
reference_energy = traj[reference_image_number].get_total_energy()
energies = [frame.get_total_energy() - reference_energy for frame in traj]
energies = [frame.get_total_energy() - reference_energy for frame in traj]

frame_numbers = list(range(len(traj)))

# Plotting
plt.figure(figsize=(10, 6))
plt.plot(frame_numbers, energies, "b-")
plt.xlabel("Frame Number")
plt.ylabel("Total Energy Difference (eV)")
# plt.title(f'Total Energy Difference vs Frame Number (Reference: Image {reference_image_number})')
plt.grid(True)
plt.tight_layout()
plt.show()
