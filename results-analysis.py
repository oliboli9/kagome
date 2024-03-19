import os
import pandas as pd

from delaunay import make_delaunay_triangulation, print_neighbour_counts

directory_path = "trajectories/10001"


def lowest_energies():
    df = pd.read_csv("results/10001.txt", sep=" ", skiprows=1)
    lowest_energies = df.sort_values(by="energy").head(10)
    print(lowest_energies)

    grouped = df.groupby("surf.n")
    lowest_energy_per_group = grouped.apply(
        lambda x: x.nsmallest(1, "energy")
    ).reset_index(drop=True)
    print(lowest_energy_per_group)


for file_name in os.listdir(directory_path):
    file_path = os.path.join(directory_path, file_name)

    atoms, tri, points, points_2d, neighbours, amplitude = make_delaunay_triangulation(
        file_path
    )
    print_neighbour_counts(points, neighbours)
