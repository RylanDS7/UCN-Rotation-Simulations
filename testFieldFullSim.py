"""
Instance of UCNspinRotSim for test field

Code by Rylan Stutters
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import UCNspinRotSim


# Constants
v = 7  # Speed of neutrons in m/s
gamma = 1.832e8  # Gyromagnetic ratio for neutrons in rad/s/T
Bo = 1e-6  # Constant magnetic field in T in the z direction
D = 0.095 # Diameter of tube
yo = -0.4

# TODO: import field into B and pos
pos = []
B = []

sim = UCNspinRotSim(v, gamma, [np.array(pos), np.array(B)], D, yo)

sim.plot_path(sim.simulate_path())

