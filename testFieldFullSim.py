"""
Instance of UCNspinRotSim for test field data

Code by Rylan Stutters
"""

import numpy as np
import UCNspinRotSim as ucn


# Constants
v = 7  # Speed of neutrons in m/s
gamma = 1.832e8  # Gyromagnetic ratio for neutrons in rad/s/T
Bo = 1e-6  # Constant magnetic field in T in the z direction
D = 0.095 # Diameter of tube
yo = -1.87000 # Starting y value
yf = -1.05 # Ending y value
upsample_factor = 40
S0 = np.array([0, 0.001, 1])
num_paths = 3

# import field data into B and pos
pos = []
B = []
lines = []

with open('field_data/gradDescentField.txt', 'r') as file:
    for line in file:
        elements = line.split(",")
        lines.append(elements)

for line in lines:
    for x in np.linspace(-0.95, 0.95, 41):
        for z in np.linspace(-0.95, 0.95, 41):
            y = float(line[0])
            Bz = float(line[1])

            pos.append([x, y, z])
            B.append([0, 1, Bz])

    
# run simulation
sim = ucn.UCNspinRotSim(gamma, [np.array(pos), np.array(B)], num_paths, v, D, yo, yf, upsample_factor)
sim.solve_spins(S0)

# plot output data
sim.plot_spin_set()
