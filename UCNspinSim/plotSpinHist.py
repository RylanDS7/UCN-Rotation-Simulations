"""
Plots a histogram of spin values extracted from a csv

Code by Rylan Stutters
Sep 2025

"""

import numpy as np
import matplotlib.pyplot as plt

spins = np.loadtxt("UCNspinSim/spins.csv", delimiter=",")

plt.hist(spins, 5)
plt.show()
