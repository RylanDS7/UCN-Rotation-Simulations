"""
Script to iterate through parameters for guiding magnetic fields
to determine the field that best preserves UCN spin

Code by Rylan Stutters
"""

import numpy as np
import matplotlib.pyplot as plt
from sympy import *

A = 0.00005 # init amplitude of field
v = 7  # Speed of neutrons in m/s
gamma = 1.832e8  # Gyromagnetic ratio for neutrons in rad/s/T
B0 = 10**(-6)  # Constant magnetic field in T in the z direction

Lo = - 3.75 / 2
L1 = - 3.5 / 2
L2 = - 3 / 2
L3 = - 2.6 / 2
L4 = - 2.55 / 2
L5 = - 2.4 / 2
L6 = - 2.25 / 2
Lf = - 2 / 2

def exp_polynomial_field(A, B0, Po, P1, P2, P3, Pf, a1, b1, c1, d1, e1, a2, b2, c2, d2, e2, dt):
    """sets up the exponential polynomial function for the given parameters

    Args:
        A (float): Initial vertical field magnitude
        B0 (float): Constant magnetic field in T in the z direction
        Po (float): The x value that the field starts at
        P1 (float): The x value that the first function starts at
        P2 (float): The x value that the second function starts at
        P2 (float): The x value that the second function ends at - constant B0
        Pf (float): The x value that the field ends at
        a1 (float): parameter 1 of the first function
        b1 (float): parameter 2 of the first function
        c1 (float): parameter 3 of the first function
        d1 (float): parameter 4 of the first function
        e1 (float): parameter 5 of the first function
        a2 (float): parameter 1 of the second function
        b2 (float): parameter 2 of the second function
        c2 (float): parameter 3 of the second function
        d2 (float): parameter 4 of the second function
        e2 (float): parameter 5 of the second function
        dt (float): the time step over which the condsider the field

    Returns:
        np.array[float]: The x values along the field
        np.array[float]: The vertical field values along the path
        np.array[float]: The derivative of the field at every point

    """
    # compile x values
    x_vals = np.linspace(Po, Pf, (int((Pf - Po) / dt)))

    # init y values
    y_vals = np.zeros(len(x_vals))

    for i in range(len(x_vals)):
        if (Po <= x_vals[i] <= P1): # first section
            y_vals[i] = A
        elif (P1 < x_vals[i] <= P2): # second section
            x = x_vals[i] - P1
            poly = a1 * x**4 + b1 * x**3 + c1 * x**2 + d1 * x + e1
            y_vals[i] = A * np.exp(-poly)
            P2_index = i # tracks the switching index to P2
        elif (P2 < x_vals[i] <= P3): # third section
            x = x_vals[i] - P2
            poly = a2 * x**4 + b2 * x**3 + c2 * x**2 + d2 * x + e2
            y_vals[i] = y_vals[P2_index] * np.exp(-poly)
        elif (P3 < x_vals[i] <= Pf): # fourth section
            y_vals[i] = B0

    return x_vals, y_vals, np.gradient(y_vals, x_vals)


def calc_kappa(gamma, v, B, dB_dx):
    """Calculates the adiabaticity for a given field

    Args:
        gamma (float): gyromagnetic ratio
        v (float): speed of UCNs
        B (np.array[float]): magnetic field values
        dB_dx (np.array[float]): magnetic field derivative values

    Returns:
        np.array[float]: The adiabaticities at each point

    """
    kappa = np.zeros(len(B))
    for i in range(len(B)):
        if (np.abs(dB_dx[i]) == 0 or B[i] == A):
            kappa[i] = 10**7
        else:
            kappa[i] = (gamma * B[i]) / (v * np.abs(dB_dx[i]) / B[i])
            if kappa[i] > 10**7:
                kappa[i] = 10**7

    return kappa


def plot_field(x, B, kappa):
    fig, ax = plt.subplots(2, 1, figsize=(12, 12))

    ax[0].plot(x, B * 1e6, label="Bz", color="blue")
    ax[0].vlines([L1], -10, 50, label="L1", colors='green', linestyles='dotted')
    ax[0].vlines([L3], -10, 50, label="L3", colors='orange', linestyles='dotted')
    ax[0].vlines([L6], -10, 50, label="L6", colors='brown', linestyles='dotted')
    ax[0].legend()
    ax[0].grid(True)
    ax[0].set_ylabel("Magnetic Field (μT)", fontsize=20)

    ax[1].plot(x, kappa, label="Adiabaticity", color="purple")
    ax[1].vlines([L1], -10, 10**8, label="L1", colors='green', linestyles='dotted')
    ax[1].vlines([L3], -10, 10**8, label="L3", colors='orange', linestyles='dotted')
    ax[1].vlines([L6], -10, 10**8, label="L6", colors='brown', linestyles='dotted')
    ax[1].set_yscale("log")
    ax[1].set_xlabel("Axial Position along the Path (m)", fontsize=20)
    ax[1].set_ylabel("Adiabaticity κ", fontsize=20)
    ax[1].legend()
    ax[1].grid(True)

    plt.show()

results = []

a1s, b1s, c1s, d1s = symbols('a1s b1s c1s d1s')

x = 0.45
x4, x3, x2 = x**4, x**3, x**2

exponent = x**4 * a1s + x**3 * b1s + x**2 * c1s + x * d1s
exp_expr = A * ln(-exponent)
P1eq1 = Eq(exp_expr, A / 5)
P1eq2 = Eq(A * (4*x3*a1s + 3*x2*b1s + 2*x*c1s + d1s) / exponent, -0.000005)

solution = solve((P1eq1, P1eq2), (c1s, d1s), dict=True)[0]

a1 = 0.01
b1 = 0.01
c1 = solution[c1s].subs({a1s: a1, b1s: b1})
d1 = solution[d1s].subs({a1s: a1, b1s: b1})

c1 = float(c1.evalf())
d1 = float(d1.evalf())


a2s, b2s, c2s, d2s = symbols('a2s b2s c2s d2s')

x = 0.175
x4, x3, x2 = x**4, x**3, x**2

exponent = x**4 * a2s + x**3 * b2s + x**2 * c2s + x * d2s
exp_expr = A / 5 * exp(-exponent)
P2eq1 = Eq(exp_expr, A / 50)
P2eq2 = Eq(-(4*x3*a2s + 3*x2*b2s + 2*x*c2s + d2s) * exp_expr, 0)

solution = solve((P2eq1, P2eq2), (c2s, d2s), dict=True)[0]

increment = 0
for a2 in np.linspace(0.1, 0.01, 10):
    for b2 in np.linspace(0.1, 0.001, 10):
        c2 = solution[c2s].subs({a2s: a2, b2s: b2})
        d2 = solution[d2s].subs({a2s: a2, b2s: b2})
        c2 = float(c2.evalf())
        d2 = float(d2.evalf())

        x, B, dB_dx = exp_polynomial_field(A, B0, Lo, L1, L3, L6, Lf, a1, b1, c1, d1, 0, a2, b2, c2, d2, 0, 0.001)
        kappa = calc_kappa(gamma, v, B, dB_dx)
        results.append(np.array([a1, b1, c1, d1, a2, b2, c2, d2, np.min(kappa)]))
        increment += 1
        print(f"Done {increment} iterations")

results = np.array(results)

max = np.max(results[:, 8])
for index in np.where(results[:, 8] == max)[0]:
    print(results[index])
    a1 = results[index][0]
    b1 = results[index][1]
    c1 = results[index][2]
    d1 = results[index][3]
    a2 = results[index][4]
    b2 = results[index][5]
    c2 = results[index][6]
    d2 = results[index][7]
    x, B, dB_dx = exp_polynomial_field(A, B0, Lo, L1, L3, L6, Lf, a1, b1, c1, d1, 0, a2, b2, c2, d2, 0, 0.001)
    kappa = calc_kappa(gamma, v, B, dB_dx)
    print(np.min(kappa))
    plot_field(x, B, kappa)










