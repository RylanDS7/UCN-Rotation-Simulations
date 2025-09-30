"""
Plots a histogram of spin values extracted from a csv

Code by Rylan Stutters
Sep 2025

"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

sim = pd.read_csv("UCNspinSim/sim.csv")

fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(10, 4))

bins = np.linspace(0.75, 1, 51)
axes[0].hist(sim["Ending Spin"], bins=bins)
axes[0].set_title("Ending Vertical Spin Componenet (Sz)")
axes[0].set_xlabel("Sz")
axes[0].set_ylabel("Number of UCN")

axes[1].scatter(sim["Theta"], sim["Ending Spin"], s=6)
axes[1].set_title("Ending Vertical Spin Componenet (Sz) vs Initial Path Angle (theta)")
axes[1].set_xlabel("Theta")
axes[1].set_ylabel("Sz")
axes[1].grid()

plt.show()
