"""
Instance of UCNspinRotSim to test the optimal tapering field

Code by Rylan Stutters
Sep 2025
"""

import numpy as np
import UCNspinRotSim as ucn


# Constants
v = 7  # Speed of neutrons in m/s
gamma = 1.832e8  # Gyromagnetic ratio for neutrons in rad/s/T
Bo = 1e-6  # Constant magnetic field in T in the z direction
D = 0.095 # Diameter of tube
yo = -0.4 # Starting y value
yf = 0.9 # Ending y value
upsample_factor = 40
S0 = np.array([0, 0, 1])
num_paths = 5

# B function coefficients
a_1 = -20.146
a_2 = 32.337
a_3 = 6.276
a_4 = -44.399
a_5 = 37.489
a_6 = -15.768
A = 30

# define B field as function of axial position
def B_z(y):
    if y < 0:
        return A
    elif y <= 0.75397:
        poly = (a_1 * y**6) + (a_2 * y**5) + (a_3 * y**4) + (a_4 * y**3) + (a_5 * y**2) + (a_6 * y)
        return A * np.exp(poly)
    else:
        return 1


# define x,y,z domains
x_dom = np.linspace(-0.1, 0.1, 10)
y_dom = np.linspace(-0.5, 1, 500)
z_dom = np.linspace(-0.1, 0.1, 10)

pos = []
B = []

for y in y_dom:
    for x in x_dom:
        for z in z_dom:
            pos.append([x, y, z])
            r = np.sqrt(x**2 + z**2)
            signx = x / np.abs(x)
            signz = z / np.abs(z)
            B.append([np.random.normal(-signx * r, 0.1), np.random.normal(signz * r, 0.1), B_z(y)])


# run simulation
sim = ucn.UCNspinRotSim(gamma, [np.array(pos), np.array(B)], num_paths, v, D, yo, yf, upsample_factor)
sim.solve_spins(S0)

# plot output data
sim.plot_spin_set(pdf_name="OptimalSpinEvo.pdf", display=True)

# save the ending spins
# sim.save_end_state()
