"""
Code to maximze plot the tapering fields

Code by Rylan Stutters
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize

A = 30 # initial field value in uT
v = 7  # Speed of neutrons in m/s
gamma = 1.832e8  # Gyromagnetic ratio for neutrons in rad/s/T
B0 = 1  # Constant magnetic field in uT in the z direction

# MSR layer distances
L0 = - 4 / 2
L1 = - 3.5 / 2
L2 = - 3 / 2
L3 = - 2.6 / 2
L4 = - 2.55 / 2
L5 = - 2.4 / 2
L6 = - 2.25 / 2
Lf = - 2 / 2
flge = L1 - 0.12897 # flange location

# Shift to set zero at flange
L0 -= flge
L1 -= flge
L2 -= flge
L3 -= flge
L4 -= flge
L5 -= flge
L6 -= flge
Lf -= flge
flge = 0


def generate_field(parameters):
    """Generates function from set of parameters

    Args:
        parameter (list[float]): list of function parameters

    Returns:
        ndarray[float]: The field values at each point

    """
    N = 2000

    function_x = np.linspace(0, L6-flge, N)
    y = np.zeros(len(function_x))

    poly = 0
    for p in range(len(parameters)):
        poly += parameters[p] * (function_x[-1]**(p+2))
    p1 = (np.log(1 / A) - poly) / function_x[-1]
    params = list(parameters)
    params.insert(0, p1)

    for i in range(len(y)):
        poly = 0
        for p in range(len(params)):
            poly += params[p] * (function_x[i]**(p+1))
        y[i] = A * np.exp(poly)

    sample_rate = N / (L6 - flge)
    front_y = A * np.ones(int((flge - L0) * (sample_rate)))
    back_y = np.ones(int((Lf - L6) * (sample_rate)))

    x_vals = np.linspace(L0, Lf, len(y) + len(front_y) + len(back_y))
    y_vals = np.concatenate((front_y, y, back_y))

    return x_vals, y_vals



def plot_field(x, B):
    plt.plot(x, B, label="Bz", color="blue")
    plt.ylim((-10, 40))
    plt.vlines([L1], -10, 50, label="L1", colors='green', linestyles='dotted')
    plt.vlines([L2], -10, 50, label="L2", colors='yellow', linestyles='dotted')
    plt.vlines([L3], -10, 50, label="L3", colors='orange', linestyles='dotted')
    plt.vlines([L6], -10, 50, label="L6", colors='brown', linestyles='dotted')
    plt.legend()
    plt.grid(True)
    plt.ylabel("Magnetic Field (μT)", fontsize=20)
    plt.xlabel("Axial Position along the Path (m)", fontsize=20)

    plt.rcParams['figure.figsize'] = [6.4, 1]
    plt.show()
    


x = [-20.146, 32.337, 6.276, -44.399, 37.489, -15.768]

x_vals, y_vals = generate_field(x)
plot_field(x_vals, y_vals)

