from pointgroup import PointGroup
from ase.io.trajectory import Trajectory
import matplotlib.pyplot as plt

traj = Trajectory("99999/stream_2_atoms_120")
atoms = traj[-1]

fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')
pos = atoms.get_positions()
x = pos[:, 0]
y = pos[:, 1]
z = pos[:, 2]
ax.scatter(x, y, z)
ax.set_xlabel('X Position')
ax.set_ylabel('Y Position')
ax.set_zlabel('Z Position')

plt.savefig('120atoms')


pg = PointGroup(positions=atoms.get_positions(), 
                symbols=atoms.symbols)

print('Point group: ', pg.get_point_group())

pg = PointGroup(positions=[[ 0.000000,  0.000000,  0.000000],
                           [ 0.000000,  0.000000,  1.561000],
                           [ 0.000000,  1.561000,  0.000000],
                           [ 0.000000,  0.000000, -1.561000],
                           [ 0.000000, -1.561000,  0.000000],
                           [ 1.561000,  0.000000,  0.000000],
                           [-1.561000,  0.000000,  0.000000]], 
                symbols=['S', 'F', 'F', 'F', 'F', 'F', 'F'])

print('Point group: ', pg.get_point_group())