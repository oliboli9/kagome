import numpy as np
import matplotlib.pyplot as plt
from numpy.random import SeedSequence, default_rng

from ase.io import Trajectory

from methods.annealing import AnnealingSimulator
from methods.calculator import RadialPotential
from methods.kagome import Kagome
from methods.surface import CappedCylinderSurface
from methods.triangulation import NonConvexTriangulation

surface = CappedCylinderSurface(7, 10, (15, 15, 15), n=100)
r0 = np.sqrt(surface.area) * np.sqrt(2) / np.sqrt(100)
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
    fmax=0.09,
)

#### Triangulation
# view(atoms)
coords = atoms.get_positions()
triangulation = NonConvexTriangulation()
simplices = triangulation.triangulate(coords, alpha=0.1)

#### Kagome
kagome = Kagome(simplices, coords, surface=surface, r0=r0 / 2)
fig = plt.figure()
# ax = fig.add_subplot(121, projection="3d")
# ax.view_init(elev=90, azim=0)
# ax2 = fig.add_subplot(122, projection="3d")
# ax2.view_init(elev=90, azim=0)
ax = fig.add_subplot(131)
ax2 = fig.add_subplot(132)
ax3 = fig.add_subplot(133)

kagome.plot_weavers_yz_projection(ax)
kagome.straighten_weavers(steps=10)
kagome.plot_weavers_yz_projection(ax2)
kagome.straighten_weavers(steps=40)
kagome.plot_weavers_yz_projection(ax3)
plt.show()
