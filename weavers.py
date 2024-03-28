from ase import Atoms
from ase.cell import Cell
from ase.io.trajectory import Trajectory
from ase.optimize import BFGS

from itertools import islice
import matplotlib.pyplot as plt
from kagome import (
    find_triangle_midpoints,
    create_indices_dict,
)
from calculator import RadialPotential, KagomePotential, KagomeRadialPotential
from delaunay import make_delaunay_triangulation, print_neighbour_counts


# trajectory_paths = ["trajectories/10001/bfgs-3-3-110-2000-500-50.traj"]
# trajectory_paths = ["bfgs-4-3-22-1500-150-50.traj"]
trajectory_paths = ["polar.traj"]


for path in trajectory_paths:
    au_atoms, tri, points, points_2d, neighbours, amplitude = (
        make_delaunay_triangulation(path, 3)
    )
    print_neighbour_counts(points, neighbours)
    # plot_delaunay(tri, points, points_2d, neighbours, amplitude)

    kagome_positions_dict = find_triangle_midpoints(tri, points)
    # kagome_positions_dict = {
    #     key: kagome_positions_dict[key] for key in islice(kagome_positions_dict, 5)
    # }
    # print(f"Kagome positions dict {kagome_positions_dict}")
    kagome_neighbour_dict = create_indices_dict(kagome_positions_dict)
    # print(f"Kagome neighbour dict {kagome_neighbour_dict}")
    neighbour_keys = list(kagome_neighbour_dict.keys())
    # print(f"Neighbour keys {neighbour_keys}")
    sorted_keys = sorted(neighbour_keys)
    # print(f"Sorted keys {sorted_keys}")
    sorted_neighbour_dict = {key: kagome_neighbour_dict[key] for key in sorted_keys}
    # print(f"Sorted neighbour dict {sorted_neighbour_dict}")
    sorted_positions = [None] * len(list(kagome_positions_dict.keys()))
    for new_index, item in zip(neighbour_keys, list(kagome_positions_dict.keys())):
        sorted_positions[new_index] = item

    # print(kagome_positions_dict)
    # print(list(kagome_positions_dict.keys()))
    # print(
    #     sorted_positions.index(
    #         (0.08468293185389564, 12.357850985387854, 0.017378947364130117)
    #     ),
    #     sorted_positions.index(
    #         (-0.4424465259399222, 19.0444367063661, 0.2889780494148348)
    #     ),
    #     sorted_positions.index(
    #         (-0.44407465474827446, 14.893366284649261, 0.3232106709650871)
    #     ),
    # )

    # make a check here that its sorted properly

    cell = Cell.fromcellpar([30, 30, 30, 90, 90, 90])

    kagome_atoms = Atoms(
        "H" * len(sorted_positions),
        positions=sorted_positions,
        cell=cell,
        pbc=(1, 1, 0),
    )

    # plt.figure(figsize=(8, 6))
    fig, (ax1, ax2) = plt.subplots(1, 2)

    # Iterate over the dictionary and plot lines between each point and its neighbors
    positions = kagome_atoms.get_positions()
    # print(positions[0], positions[231], positions[184])
    # plt.scatter(
    #     [positions[0][0]],
    #     [positions[0][1]],
    #     marker="X",
    #     color="blue",
    #     s=100,
    # )
    # plt.scatter(
    #     [positions[231][0], positions[184][0]],
    #     [positions[231][1], positions[184][1]],
    #     marker="X",
    #     color="red",
    #     s=100,
    # )
    for point, neighbour_pairs in sorted_neighbour_dict.items():
        for neighbours in neighbour_pairs:
            for neighbour in neighbours:
                ax1.plot(
                    [positions[point][0], positions[neighbour][0]],
                    [positions[point][1], positions[neighbour][1]],
                    "b-",
                    linewidth=0.5,
                )
                # plt.plot(
                #     [point[0], neighbour[0]],
                #     [point[1], neighbour[1]],
                #     "b-",
                #     linewidth=0.5,
                # )

    ax1.set_title("Weavers")
    ax1.set_xlabel("X coordinate")
    ax1.set_ylabel("Y coordinate")
    # ax1.set_grid(True)

    # print(au_atoms)
    # print(kagome_atoms)
    au_atoms.calc = RadialPotential(r0=2)
    kagome_atoms.calc = KagomePotential(neighbour_dict=sorted_neighbour_dict)
    # kagome_atoms.calc = KagomeRadialPotential(
    #     r0=2, neighbour_dict=sorted_neighbour_dict
    # )
    # kagome_atoms.extend(au_atoms)
    # print(f"calc{kagome_atoms.calc}")
    # kagome_atoms.calc = RadialPotential(r0=3)
    # traj_bfgs = Trajectory(
    #     f"kagome.traj",
    #     "w",
    #     kagome_atoms,
    # )

    # def wrap_and_write_bfgs():
    #     # atoms.wrap()
    #     traj_bfgs.write(kagome_atoms)

    local_minimisation = BFGS(kagome_atoms)
    # local_minimisation.attach(wrap_and_write_bfgs)
    local_minimisation.run(steps=20)

    positions = kagome_atoms.get_positions()
    for point, neighbour_pairs in sorted_neighbour_dict.items():
        for neighbours in neighbour_pairs:
            for neighbour in neighbours:
                ax2.plot(
                    [positions[point][0], positions[neighbour][0]],
                    [positions[point][1], positions[neighbour][1]],
                    "b-",
                    linewidth=0.5,
                )
                # plt.plot(
                #     [point[0], neighbour[0]],
                #     [point[1], neighbour[1]],
                #     "b-",
                #     linewidth=0.5,
                # )

    ax2.set_title("Weavers")
    ax2.set_xlabel("X coordinate")
    ax2.set_ylabel("Y coordinate")
    # ax2.set_grid(True)

    # Show the plot
    plt.show()
