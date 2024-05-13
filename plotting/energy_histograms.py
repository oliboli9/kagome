import pandas as pd
import matplotlib.pyplot as plt

file_name = "plotting/reproducability.txt"

df = pd.read_csv(file_name, delim_whitespace=True)

# energy = df.iloc[:, -1]
energy = df["energy/atom"]


def plot_histogram():
    plt.hist(energy, bins=50)
    # plt.title("Histogram of Energy")
    plt.xlabel("Potential Energy per Atom [eV]")
    # plt.xticks([0.18, 0.185, 0.19, 0.195, 0.2, 0.205, 0.21, 0.215, 0.22])
    plt.ylabel("Frequency")
    plt.show()


plot_histogram()

lowest_five = df.sort_values(by=df.columns[-1]).head(10)
print(lowest_five)
