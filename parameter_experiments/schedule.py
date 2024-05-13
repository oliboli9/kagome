import pandas as pd
import matplotlib.pyplot as plt


file_path = "parameter_experiments/schedule.txt"
df = pd.read_csv(file_path, delim_whitespace=True)

lowest_energies = df.sort_values(by="energy").head(10)
print(lowest_energies)

average_values = lowest_energies.mean()
print(average_values)

plt.figure(figsize=(10, 6))
grouped = df.groupby("cooling_rate")["energy"].mean().reset_index()
plt.scatter(grouped["cooling_rate"], grouped["energy"], color="blue")  # Scatter plot
plt.title("Energy vs Start Temp")
plt.xlabel("Start Temp")
plt.ylabel("Energy")
plt.grid(True)
plt.show()
