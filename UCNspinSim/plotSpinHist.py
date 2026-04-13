"""
Plots a histogram of spin values extracted from a csv

Code by Rylan Stutters
Sep 2025

"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

sim = pd.read_csv("UCNspinSim/taperV2spins.csv")

# fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(10, 4))
plt.rcParams.update({'font.size': 18})

# bins = np.linspace(0.75, 1, 51)
# axes[0].hist(sim["Ending Spin"], bins=bins)
# axes[0].set_title("Ending Vertical Spin Componenet (Sz)")
# axes[0].set_xlabel("Sz", fontsize=15)
# axes[0].set_ylabel("Number of UCN", fontsize=15)
# axes[0].grid()

# axes[1].scatter(sim["Theta"], sim["Ending Spin"], s=6)
# axes[1].set_title("Ending Vertical Spin Componenet (Sz) \nvs Initial Path Angle (theta)")
# axes[1].set_xlabel("Theta", fontsize=15)
# axes[1].set_ylabel("Sz", fontsize=15)
# axes[1].grid()

fig, ax = plt.subplots(figsize=(10,8))

bins = np.linspace(0, 1, 51)
ax.hist(sim["Ending Spin"], bins=bins)
ax.set_xlabel("Ending Vertical Spin Component ($S_z$)", fontsize=24)
ax.set_ylabel("Number of UCN", fontsize=24)
ax.grid()

print(np.mean(sim["Ending Spin"]))

plt.show()
