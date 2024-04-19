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
from surface import CapsuleSurface
from triangulation import ConvexTriangulation

surface = CapsuleSurface(5, 2, (15, 15, 15), n=50)
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
pos = atoms.get_positions()
print(pos[0])
print(surface.normals(pos[:1]))