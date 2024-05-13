import pandas as pd
import matplotlib.pyplot as plt


file_path = "parameter_experiments/friction.txt"
df = pd.read_csv(file_path, delim_whitespace=True)
plt.figure(figsize=(10, 6))
plt.scatter(df["friction"], df["energy"], color="blue")  # Scatter plot
plt.title("Energy vs Friction")
plt.xlabel("Friction")
plt.ylabel("Energy")
plt.grid(True)
plt.show()
