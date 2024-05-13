import pandas as pd
import matplotlib.pyplot as plt


def read_and_plot_energy_data(file_path):
    # Read the data, specifying columns for timestep and energy
    data = pd.read_csv(
        file_path,
        delim_whitespace=True,
        usecols=[2, 3],
        names=["timestep", "energy"],
        header=0,
    )

    # Calculate the average energy for each timestep
    average_energy_per_timestep = (
        data.groupby("timestep")["energy"].mean().reset_index()
    )

    # Plotting the average energy per timestep
    plt.figure(figsize=(12, 6))
    plt.plot(
        average_energy_per_timestep["timestep"],
        average_energy_per_timestep["energy"],
        marker="o",
        linestyle="-",
        color="b",
    )
    # plt.title("Average of Potential Energy per Timestep")
    plt.xlabel("Timestep * fs")
    plt.ylabel("Average Potential Energy [eV]")
    plt.grid(True)
    plt.show()


file_path = "parameter_experiments/timestep.txt"
read_and_plot_energy_data(file_path)
