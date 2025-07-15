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
yo = -3.6 # Starting y value
yf = -1.3 # Ending y value
upsample_factor = 40
S0 = np.array([0, 0, 1])
num_paths = 3

# import field data into B and pos
pos = []
B = []
lines = []

# read space seperated data from txt file
with open('field_data/FieldGrid-2.txt', 'r') as file:
    for line in file:
        elements = line.split()
        if elements[0] != '%': 
            lines.append(elements)

# convert data to set of 2 np arrays 
for line in lines:
    x = float(line[0])
    y = float(line[1])
    z = float(line[2])
    Bx = float(line[3])
    By = float(line[4])
    Bz = float(line[5])

    # artifically add 1uT uniform field
    if Bz < 1E-6:
        Bz = 1E-6

    pos.append([x, y, z])
    B.append([Bx, By, Bz])
    

# run simulation
sim = ucn.UCNspinRotSim(gamma, [np.array(pos), np.array(B)], num_paths, v, D, yo, yf, upsample_factor)
sim.solve_spins(S0)

# plot output data
sim.plot_spin_set()
