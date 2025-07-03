"""
Code to parallelize the UCNspinRotSim class to simulate large collections of UCN

Code by Rylan Stutters
"""

import numpy as np
import UCNspinRotSim as ucn
import multiprocessing as mp

# Constants
v = 7  # Speed of neutrons in m/s
gamma = 1.832e8  # Gyromagnetic ratio for neutrons in rad/s/T
Bo = 1e-6  # Constant magnetic field in T in the z direction
D = 0.095 # Diameter of tube
yo = -3.6 # Starting y value
yf = -1.3 # Ending y value
S0 = np.array([0,0,1]) # initial spin vector
num_paths = 1 # number of UCN paths to simulate

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

def solve_spins_wrapper(args):
    sim_params, S0, path, steps = args
    sim = ucn.UCNspinRotSim(*sim_params)
    return sim.solve_spins(S0, path, steps)

if __name__ == '__main__':
    mp.freeze_support()

    # Simulation parameters
    sim_params = (v, gamma, [np.array(pos), np.array(B)], D, yo, yf)

    # Create sim object to generate paths
    sim = ucn.UCNspinRotSim(*sim_params)

    paths = []
    collisions_set = []
    thetas = []
    for i in range(num_paths):
        path, collisions, theta = sim.simulate_path()
        paths.append(path)
        collisions_set.append(collisions)
        thetas.append(theta)

    print("Finished generating paths")

    with mp.Pool(mp.cpu_count()) as pool:
        results = pool.map(solve_spins_wrapper, [(sim_params, S0, path, 40) for path in paths])

    # Unzip the list of (path, spin) tuples
    paths_out, spins_out = zip(*results)

    # Plot the results
    sim.plot_spin_set(paths_out, spins_out, collisions_set, np.array(thetas))
