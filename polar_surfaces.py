from itertools import product
import os
import numpy as np
import sympy as sp
from numpy.random import SeedSequence, default_rng
from multiprocessing import Pool

from ase.parallel import paropen

from annealing import setup_periodic_atoms, anneal
from delaunay import (
    make_delaunay_triangulation,
    get_neighbour_counts,
    plot_delaunay,
    repeat,
)
from surface import PeriodicSurface, SurfaceConstraint, TorusSurface


n_streams = 1
n_processes = 1
seed = 10020
ss = SeedSequence(seed)
child_seeds = ss.spawn(n_streams)
streams = [default_rng(s) for s in child_seeds]

start_temp = 2000
end_temp = 50
cooling_rate = 500
densities = np.arange(10, 125, 1)


stream = streams[0]
surf = TorusSurface(5, 2, (15, 15, 15), 500)
r0 = 5  # in flat cell found optimal to be density 1/9 and r0=4
atoms = setup_periodic_atoms(stream, surf, r0=r0)
energy = anneal(seed, 1, 1, surf, atoms, start_temp, cooling_rate, end_temp)
coords = atoms.get_positions()


# amp1=101
# amp3=109
# amp4=116
# amp5=124
# amp7 fmax 0.065

# 10020 amp3 10-124atoms 10streams r0 32/density (3.55)
