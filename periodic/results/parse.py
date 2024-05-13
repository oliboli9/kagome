import pandas as pd
import matplotlib.pyplot as plt

# Load the data from a text file
df = pd.read_csv("periodic/results/10004.txt", delim_whitespace=True)

# # Sorting the DataFrame by the 'energy' column
# sorted_df = df.sort_values('energy')

# # Selecting the top 5 rows with the lowest energy
# lowest_energy_rows = sorted_df.head(5)

# # Printing the result
# print(lowest_energy_rows)

# ############### plotting energy vs cooling rate

# # Grouping by 'cooling_rate' and calculating the mean energy
# grouped = df.groupby("cooling_rate")["energy"].mean().reset_index()

# # Plotting
# plt.figure(figsize=(10, 6))
# plt.plot(
#     grouped["cooling_rate"], grouped["energy"], marker="o", linestyle="-", color="b"
# )
# plt.title("Energy vs. Cooling Rate")
# plt.xlabel("Cooling Rate")
# plt.ylabel("Average Energy")
# plt.grid(True)
# plt.show()

############### plotting energy vs starting temp

# # Grouping by 'cooling_rate' and calculating the mean energy
# grouped = df.groupby("start_temp")["energy"].mean().reset_index()

# # Plotting
# plt.figure(figsize=(10, 6))
# plt.plot(grouped["start_temp"], grouped["energy"], marker="o", linestyle="-", color="b")
# plt.title("Energy vs. Starting Temp")
# plt.xlabel("Starting Temp")
# plt.ylabel("Average Energy")
# plt.grid(True)
# plt.show()

################# both
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection="3d")

# Scatter plot
sc = ax.scatter(
    df["cooling_rate"], df["start_temp"], df["energy"], c=df["energy"], cmap="viridis"
)

# Labels and title
ax.set_xlabel("Cooling Rate")
ax.set_ylabel("Starting Temperature")
ax.set_zlabel("Energy")
ax.set_title("Energy as a Function of Cooling Rate and Starting Temperature")

# Color bar
plt.colorbar(sc, ax=ax, label="Energy")

# Show the plot
plt.show()
