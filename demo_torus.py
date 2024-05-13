import numpy as np
import matplotlib.pyplot as plt
from numpy.random import SeedSequence, default_rng

from ase.io import Trajectory
from ase.optimize import BFGS
from ase.visualize import view

from methods.annealing import AnnealingSimulator
from methods.calculator import RadialPotential
from methods.kagome import Kagome
from methods.surface import TorusSurface
from methods.triangulation import NonConvexTriangulation

from plotting.plot_torus_surface import plot_surf_with_atoms

no_atoms = 100
surface = TorusSurface(5, 2, (15, 15, 15), n=no_atoms)
r0 = np.sqrt(surface.area) * np.sqrt(2) / np.sqrt(no_atoms)
calculator = RadialPotential(r0=r0)

n_streams = 1
n_processes = 1
seed = 10020
ss = SeedSequence(seed)
child_seeds = ss.spawn(n_streams)
streams = [default_rng(s) for s in child_seeds]

#### Annealing
annealing = AnnealingSimulator(surface, calculator)
atoms = annealing.setup_atoms(streams[0])
annealing.anneal(
    atoms,
    2000,
    10,
    traj_path_md="torus_md.traj",
    traj_path_bfgs="torus_bfgs.traj",
    fmax=0.12,
)
# plot_surf_with_atoms(atoms.get_positions())
# view(atoms)


# #### Triangulation
coords = atoms.get_positions()
triangulation = NonConvexTriangulation()
simplices = triangulation.triangulate(coords, 0.4)


# # #### Kagome
kagome = Kagome(simplices, coords, surface=surface, r0=r0 / 2)
fig = plt.figure()
# ax = fig.add_subplot(121, projection="3d")
# ax.view_init(elev=90, azim=0)
# ax2 = fig.add_subplot(122, projection="3d")
# ax2.view_init(elev=90, azim=0)
ax = fig.add_subplot(131)
ax2 = fig.add_subplot(132)
ax3 = fig.add_subplot(133)


kagome.plot_weavers_top_half(ax)
kagome.straighten_weavers(steps=15)
kagome.plot_weavers_top_half(ax2)
kagome.straighten_weavers(steps=15)
kagome.plot_weavers_top_half(ax3)
plt.show()
