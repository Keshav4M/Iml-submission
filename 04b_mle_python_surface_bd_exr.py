import numpy as np
import matplotlib.pyplot as plt

N = 50
S = np.arange(1, N, 0.1)
o = np.linspace(0.1, 0.9, 100)

# MLE function
def L(S, o):
    return S * np.log(o) + (N - S) * np.log(1. - o)

# Create a figure with two subplots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
fig.suptitle("Maximum Likelihood Estimation")

# Plot the likelihood as a heatmap
heatmap = ax1.imshow(
    L(
        np.repeat(S[:, np.newaxis], len(o), axis=1),
        np.repeat(o[np.newaxis, :], len(S), axis=0)
    ),
    cmap='jet',
    origin='lower',
    aspect='auto',
    extent=[S.min(), S.max(), o.min(), o.max()]
)
ax1.set_xlabel('S')
ax1.set_ylabel('θ')
ax1.set_title("Bird's Eye View")

# Add a vertical line at S = 25
ax1.axvline(x=25, color='black')

# Plot L(o|S=25)
ax2.plot(o, L(25, o), color='blue')
ax2.set_xlabel('o')
ax2.set_title("L(o|S=25)")

# Adjust spacing between subplots
plt.subplots_adjust(wspace=0.5)

# Save the figure
plt.savefig("./plots/sThetaL25")
