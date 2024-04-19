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
from surface import TorusSurface
from triangulation import NonConvexTriangulation

surface = TorusSurface(5, 2, (15, 15, 15), n=150)
r0 = surface.density / 2
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
filename = "torus.traj"
with open(filename, "w") as file:
    pass
annealing.anneal(atoms, 2000, 100, 500, traj_path_md=filename)

#### Triangulation - matlab
# traj = Trajectory(f"torus.traj")
# atoms = traj[-1]
coords = atoms.get_positions()
triangulation = NonConvexTriangulation()
simplices = triangulation.triangulate(coords, 0.3)

#### Kagome
kagome = Kagome(simplices, coords, surface=surface, r0=r0 / 2)
fig = plt.figure()
ax = fig.add_subplot(121, projection="3d")
ax.view_init(elev=90, azim=0)
ax2 = fig.add_subplot(122, projection="3d")
ax2.view_init(elev=90, azim=0)

kagome.plot_weavers(ax)
kagome.straighten_weavers()
kagome.plot_weavers(ax2)
plt.savefig("torus")
