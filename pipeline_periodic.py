import numpy as np
import sympy as sp
import matplotlib.pyplot as plt
from numpy.random import SeedSequence, default_rng

from ase.io import Trajectory
from ase.optimize import BFGS
from ase.spacegroup.symmetrize import check_symmetry
from ase.visualize import view

from annealing import AnnealingSimulator
from calculator import RadialPotential
from kagome import Kagome
from surface import PeriodicSurface
from triangulation import SurfaceTriangulation

surface = PeriodicSurface(
    lambda x, y: 3 * sp.sin(2 * sp.pi * x / 30) * sp.sin(2 * sp.pi * y / 30),
    n=100,
)
r0 = surface.density / 2
calculator = RadialPotential(r0=r0)

n_streams = 1
n_processes = 1
seed = 10020
ss = SeedSequence(seed)
child_seeds = ss.spawn(n_streams)
streams = [default_rng(s) for s in child_seeds]

filename = "periodic.traj"
# #### Annealing
annealing = AnnealingSimulator(surface, calculator)
atoms = annealing.setup_atoms(streams[0])
with open(filename, "w") as file:
    pass
annealing.anneal(atoms, 2000, 100, 500, traj_path_md=f"{filename}")
check_symmetry(atoms, symprec=0.1, verbose=True)

# # view(atoms)
# # local_minimisation = BFGS(atoms, trajectory="polar2.traj")

# # local_minimisation.run(steps=100, fmax=0.01)

# coords = atoms.get_positions()
# np.savetxt("torus_points.csv", coords, delimiter=",", fmt="%f")


# #### Triangulation - matlab
# # simplices = np.loadtxt("simplices2.csv", delimiter=",", dtype="int") - 1
# # coords = np.loadtxt("alphapoints2.csv", delimiter=",", dtype="float")
# # coords = np.loadtxt("torus_points.csv", delimiter=",", dtype="float")
# def repeat(n, points):
#     repeated_points = []
#     for i in range(n):  # Repeat in x-direction
#         for j in range(n):  # Repeat in y-direction
#             new_points = points.copy()
#             new_points[:, 0] += 30 * i
#             new_points[:, 1] += 30 * j
#             repeated_points.append(new_points)

#     repeated_points = np.vstack(repeated_points)
#     return repeated_points


# traj = Trajectory(f"{filename}")
# atoms = traj[-1]
# coords = atoms.get_positions()
# repeats = 3
# coords = repeat(repeats, coords)
# triangulation = SurfaceTriangulation()
# simplices = triangulation.triangulate(coords)

# # fig = plt.figure()
# # ax = fig.add_subplot(111, projection="3d")
# # ax.scatter(coords[:, 0], coords[:, 1], coords[:, 2], c="b", marker="o")

# # # Plot triangles
# # for simplex in simplices:
# #     simplex_points = coords[simplex]
# #     simplex_points = np.append(
# #         simplex_points, [simplex_points[0]], axis=0
# #     )  # Connect back to the first point
# #     ax.plot(simplex_points[:, 0], simplex_points[:, 1], simplex_points[:, 2], "k-")

# # ax.set_xlabel("X")
# # ax.set_ylabel("Y")
# # ax.set_zlabel("Z")
# # ax.set_title("Delaunay Triangulation")
# # plt.show()


# #### Kagome
# kagome = Kagome(simplices, coords, surface=surface, r0=r0 / 2, periodic=True)
# fig = plt.figure()
# ax = fig.add_subplot(121, projection="3d")
# ax.view_init(elev=90, azim=0)
# ax2 = fig.add_subplot(122, projection="3d")
# ax2.view_init(elev=90, azim=0)

# kagome.plot_weavers_periodic(ax)
# kagome.straighten_weavers()
# kagome.plot_weavers_periodic(ax2)
# check_symmetry(kagome.atoms, symprec=0.001, verbose=True)
# plt.show()
