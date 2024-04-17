import numpy as np
import matplotlib.pyplot as plt
from numpy.random import SeedSequence, default_rng

from ase.io import Trajectory
from ase.optimize import BFGS
from ase.spacegroup.symmetrize import check_symmetry
from ase.visualize import view

from annealing import AnnealingSimulator
from calculator import RadialPotential
from kagome import Kagome
from surface import SphereSurface
from triangulation import ConvexTriangulation

surface = SphereSurface(r=5, centre=(15, 15, 15), n=22)
r0 = surface.density / 2
calculator = RadialPotential(r0=r0)

n_streams = 1
n_processes = 1
seed = 10020
ss = SeedSequence(seed)
child_seeds = ss.spawn(n_streams)
streams = [default_rng(s) for s in child_seeds]
filename = "sphere.traj"

# #### Annealing
# annealing = AnnealingSimulator(surface, calculator)
# atoms = annealing.setup_atoms(streams[0])
# with open(filename, "w") as file:
#     pass
# annealing.anneal(atoms, 2500, 100, 100, traj_path_md=f"{filename}")

# view(atoms)
# local_minimisation = BFGS(atoms, trajectory="polar2.traj")

# local_minimisation.run(steps=100, fmax=0.01)

# coords = atoms.get_positions()
# np.savetxt("torus_points.csv", coords, delimiter=",", fmt="%f")

#### Triangulation - matlab
# simplices = np.loadtxt("simplices2.csv", delimiter=",", dtype="int") - 1
# coords = np.loadtxt("alphapoints2.csv", delimiter=",", dtype="float")
# coords = np.loadtxt("torus_points.csv", delimiter=",", dtype="float")
traj = Trajectory(f"{filename}")
atoms = traj[-1]
coords = atoms.get_positions()
triangulation = ConvexTriangulation()
simplices = triangulation.triangulate(coords)


def plot_top_half(simplices, coords):
    # Filter simplices where all points have z > 15
    valid_simplices = []
    for simplex in simplices:
        if np.all(coords[simplex, 2] > 15):
            valid_simplices.append(simplex)

    # Convert valid simplices to a numpy array
    valid_simplices = np.array(valid_simplices)

    # Orthographic projection onto the x-y plane for visualization
    points_2d = coords[:, :2]

    # Plotting
    plt.figure(figsize=(8, 8))
    for simplex in valid_simplices:
        for i, j in zip(simplex, np.roll(simplex, 1)):
            plt.plot(*zip(points_2d[i], points_2d[j]), color="blue")

    plt.scatter(*points_2d.T, color="red")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.title("Top Half of Delaunay Triangulation on a Sphere Projected to 2D")
    plt.show()


# fig = plt.figure()
# ax = fig.add_subplot(111, projection="3d")
# ax.scatter(coords[:, 0], coords[:, 1], coords[:, 2], c="b", marker="o")

# # Plot triangles
# for simplex in simplices:
#     simplex_points = coords[simplex]
#     simplex_points = np.append(
#         simplex_points, [simplex_points[0]], axis=0
#     )  # Connect back to the first point
#     ax.plot(simplex_points[:, 0], simplex_points[:, 1], simplex_points[:, 2], "k-")

# ax.set_xlabel("X")
# ax.set_ylabel("Y")
# ax.set_zlabel("Z")
# ax.set_title("Delaunay Triangulation")
# plt.show()

#### Kagome
print(r0)
kagome = Kagome(simplices, coords, surface=surface, r0=r0 * 2)
fig = plt.figure()
ax = fig.add_subplot(121, projection="3d")
ax.view_init(elev=90, azim=0)
ax2 = fig.add_subplot(122, projection="3d")
ax2.view_init(elev=90, azim=0)
kagome.plot_weavers(ax)
kagome.straighten_weavers()
kagome.plot_weavers(ax2)
# check_symmetry(kagome.atoms, symprec=0.1, verbose=True)
plt.show()
