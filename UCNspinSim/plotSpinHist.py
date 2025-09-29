"""
Plots a histogram of spin values extracted from a csv

Code by Rylan Stutters
Sep 2025

"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

sim = pd.read_csv("UCNspinSim/sim.csv")

bins = np.linspace(0.85, 1, 16)
print(sim["Ending Spin"])
plt.hist(sim["Ending Spin"], bins=bins)
plt.show()
