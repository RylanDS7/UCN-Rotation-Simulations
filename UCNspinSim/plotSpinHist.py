"""
Plots a histogram of spin values extracted from a csv

Code by Rylan Stutters
Sep 2025

"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

sim = pd.read_csv("UCNspinSim/sim.csv")

sim["Probability"] = (1 + sim["Ending Spin"])/2
print(sim["Probability"].mean())

fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(10, 4))
plt.rcParams.update({'font.size': 14})

bins = np.linspace(0.75, 1, 51)
axes[0].hist(sim["Ending Spin"], bins=bins)
axes[0].set_title("Ending Vertical Spin Componenet (Sz)")
axes[0].set_xlabel("Sz", fontsize=15)
axes[0].set_ylabel("Number of UCN", fontsize=15)
axes[0].grid()

axes[1].scatter(sim["Theta"], sim["Ending Spin"], s=6)
axes[1].set_title("Ending Vertical Spin Componenet (Sz) \nvs Initial Path Angle (theta)")
axes[1].set_xlabel("Theta", fontsize=15)
axes[1].set_ylabel("Sz", fontsize=15)
axes[1].grid()

plt.show()

