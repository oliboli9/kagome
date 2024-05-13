import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.colors import Normalize

# Parameters for the cylinder
radius = 10
height = 10
center = (15, 15, 15)

# Define the theta for the circular part and z for the height
theta = np.linspace(0, 2 * np.pi, 100)
z = np.linspace(center[2] - height / 2, center[2] + height / 2, 50)
theta, z = np.meshgrid(theta, z)

# Calculate the coordinates for the sides of the cylinder
x = center[0] + radius * np.cos(theta)
y = center[1] + radius * np.sin(theta)

# Parameters for the hemisphere caps
phi = np.linspace(0, np.pi / 2, 50)
theta_cap, phi = np.meshgrid(theta, phi)

# Calculate the coordinates for the top hemisphere cap
x_cap_top = center[0] + radius * np.sin(phi) * np.cos(theta_cap)
y_cap_top = center[1] + radius * np.sin(phi) * np.sin(theta_cap)
z_cap_top = center[2] + height / 2 + radius * np.cos(phi)

# Calculate the coordinates for the bottom hemisphere cap
x_cap_bottom = center[0] + radius * np.sin(phi) * np.cos(theta_cap)
y_cap_bottom = center[1] + radius * np.sin(phi) * np.sin(theta_cap)
z_cap_bottom = center[2] - height / 2 - radius * np.cos(phi)

# Create a figure and a 3D axis for plotting
fig, ax = plt.subplots(subplot_kw={"projection": "3d"})

# Normalizer for colors
norm = Normalize(
    vmin=center[2] - height / 2 - radius, vmax=center[2] + height / 2 + radius
)

# Plot the sides of the cylinder
ax.plot_surface(x, y, z, facecolors=plt.cm.viridis(norm(z)), edgecolor="none")

# Plot the top and bottom hemisphere caps
ax.plot_surface(
    x_cap_top,
    y_cap_top,
    z_cap_top,
    facecolors=plt.cm.viridis(norm(z_cap_top)),
    edgecolor="none",
)
ax.plot_surface(
    x_cap_bottom,
    y_cap_bottom,
    z_cap_bottom,
    facecolors=plt.cm.viridis(norm(z_cap_bottom)),
    edgecolor="none",
)

# Set labels and title
ax.set_xlabel("X ")
ax.set_ylabel("Y ")
ax.set_zlabel("Z ")
ax.set_xlim(0, 30)
ax.set_ylim(0, 30)
ax.set_zlim(0, 30)
# ax.set_title("Uniformly Colored Cylinder with Hemispherical Caps")

# Show the plot
plt.show()
