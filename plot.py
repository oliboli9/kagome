import numpy as np
import matplotlib.pyplot as plt

# Generate x values
x = np.linspace(0, 3, 100)

# Calculate y values for (1 - x/3)^2
y_new = (1 - x / 3) ** 2

# Create the plot
plt.figure(figsize=(8, 6))
plt.plot(x, y_new, label="y = (1 - x/3)^2")
plt.title("Graph of y = (1 - x/3)^2")
plt.xlabel("x")
plt.ylabel("y")
plt.grid(True)
plt.legend()
plt.xlim(0, 3)
plt.ylim(0, 1)
plt.show()  # Display the plot
