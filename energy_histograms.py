import pandas as pd
import matplotlib.pyplot as plt

file_name = "surfaces-99988.txt"

df = pd.read_csv(file_name, sep=" ", header=None)

energy = df.iloc[:, -1]


def plot_histogram():
    plt.hist(energy, bins=30)
    plt.title("Histogram of Energy")
    plt.xlabel("Values")
    plt.ylabel("Frequency")
    plt.show()


plot_histogram()

lowest_five = df.sort_values(by=df.columns[-1]).head(10)
print(lowest_five)
