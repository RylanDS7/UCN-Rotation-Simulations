"""
Instance of UCNspinRotSim for test field

Code by Rylan Stutters
"""

import numpy as np
import matplotlib.pyplot as plt
import csv
from scipy.integrate import solve_ivp
import UCNspinRotSim as ucn


# Constants
v = 7  # Speed of neutrons in m/s
gamma = 1.832e8  # Gyromagnetic ratio for neutrons in rad/s/T
Bo = 1e-6  # Constant magnetic field in T in the z direction
D = 0.095 # Diameter of tube
yo = -4.4

# import field data into B and pos
pos = []
B = []
lines = []

# read space seperated data from txt file
with open('FieldGrid-2.txt', 'r') as file:
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

    pos.append([x, y, z])
    B.append([Bx, By, Bz])
    

# run simulation
sim = ucn.UCNspinRotSim(v, gamma, [np.array(pos), np.array(B)], D, yo)

# sim.plot_Field()
sim.plot_path(sim.simulate_path())
