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

def def_field(A, B0, Po, P1, P2, P3, Pf, a1, b1, c1, d1, e1, a2, b2, c2, d2, e2, dt):
    """sets up the exponential polynomial function for the given parameters

    Args:
        A (float): Initial vertical field magnitude
        B0 (float): Constant magnetic field in T in the z direction
        Po (float): The x value that the field starts at
        P1 (float): The x value that the first function starts at
        P2 (float): The x value that the second function starts at
        P3 (float): The x value that the second function ends at - constant B0
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
            poly = a1 * x**5 + b1 * x**4 + c1 * x**3 + d1 * x**2 + e1 * x
            y_vals[i] = A * exp(-poly)
            P2_index = i # tracks the switching index to P2
        elif (P2 < x_vals[i] <= P3): # third section
            x = x_vals[i] - P2
            poly = a2 * x**5 + b2 * x**4 + c2 * x**3 + d2 * x**2 + e2 * x + 1
            y_vals[i] = y_vals[P2_index] * poly
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


def plot_field(x, B, kappa, lowest_kappa):
    fig, ax = plt.subplots(2, 1, figsize=(12, 12))

    ax[0].plot(x, B * 1e6, label="Bz", color="blue")
    ax[0].vlines([L1], -10, 50, label="L1", colors='green', linestyles='dotted')
    ax[0].vlines([L2], -10, 50, label="L2", colors='yellow', linestyles='dotted')
    ax[0].vlines([L3], -10, 50, label="L3", colors='orange', linestyles='dotted')
    ax[0].vlines([L6], -10, 50, label="L6", colors='brown', linestyles='dotted')
    ax[0].legend()
    ax[0].grid(True)
    ax[0].set_ylabel("Magnetic Field (μT)", fontsize=20)

    ax[1].plot(x, kappa, label="Adiabaticity", color="purple")
    ax[1].vlines([L1], -10, 10**8, label="L1", colors='green', linestyles='dotted')
    ax[1].vlines([L2], -10, 10**8, label="L2", colors='yellow', linestyles='dotted')
    ax[1].vlines([L3], -10, 10**8, label="L3", colors='orange', linestyles='dotted')
    ax[1].vlines([L6], -10, 10**8, label="L6", colors='brown', linestyles='dotted')
    ax[1].set_yscale("log")
    ax[1].text(-1.7, 10**4, f"Lowest Adiabaticity = {lowest_kappa:.3f}", fontsize=12, color='blue')
    ax[1].set_xlabel("Axial Position along the Path (m)", fontsize=20)
    ax[1].set_ylabel("Adiabaticity κ", fontsize=20)
    ax[1].legend()
    ax[1].grid(True)

    plt.show()

results = []
dis1 = L2 - L1
dis2 = L6 - L2

for i in np.linspace(-41, -39, 5):
    a1s, b1s, c1s, d1s, e1s, x = symbols('a1s b1s c1s d1s e1s x')

    f1 = exp(-(x**5 * a1s + x**4 * b1s + x**3 * c1s + x**2 * d1s + x * e1s))

    P1eq1 = Eq(f1.subs(x, dis1), 2 / 50)
    P1eq2 = Eq(diff(f1, x).subs(x, dis1), -0.1)
    P1eq3 = Eq(diff(diff(f1, x), x).subs(x, dis1), 0)
    P1eq4 = Eq(diff(f1, x).subs(x, 0), i)
    P1eq5 = Eq(diff(diff(diff(f1, x), x), x).subs(x, dis1), 0)

    solution = solve((P1eq1, P1eq2, P1eq3, P1eq4, P1eq5), (a1s, b1s, c1s, d1s, e1s), dict=True)[0]

    a1 = solution[a1s]
    b1 = solution[b1s]
    c1 = solution[c1s]
    d1 = solution[d1s]
    e1 = solution[e1s]

    a1 = float(a1.evalf())
    b1 = float(b1.evalf())
    c1 = float(c1.evalf())
    d1 = float(d1.evalf())
    e1 = float(e1.evalf())


    a2s, b2s, c2s, d2s, e2s, x = symbols('a2s b2s c2s d2s e2s x')

    f2 = 2 / 50 * (x**5 * a2s + x**4 * b2s + x**3 * c2s + x**2 * d2s + x * e2s + 1)

    P2eq1 = Eq(f2.subs(x, dis2), 1 / 50)
    P2eq2 = Eq(diff(f2, x).subs(x, dis2), 0)
    P2eq3 = Eq(diff(diff(f2, x), x).subs(x, dis2), 0)
    P2eq4 = Eq(diff(f2, x).subs(x, 0), -0.1)
    P2eq5 = Eq(diff(diff(diff(f2, x), x), x).subs(x, dis2), 0)

    solution = solve((P2eq1, P2eq2, P2eq3, P2eq4, P2eq5), (a2s, b2s, c2s, d2s, e2s), dict=True)[0]

    a2 = solution[a2s]
    b2 = solution[b2s]
    c2 = solution[c2s]
    d2 = solution[d2s]
    e2 = solution[e2s]

    a2 = float(a2.evalf())
    b2 = float(b2.evalf())
    c2 = float(c2.evalf())
    d2 = float(d2.evalf())
    e2 = float(e2.evalf())


    x, B, dB_dx = def_field(A, B0, Lo, L1, L2, L6, Lf, a1, b1, c1, d1, e1, a2, b2, c2, d2, e2, 0.001)
    kappa = calc_kappa(gamma, v, B, dB_dx)
    results.append(np.array([a1, b1, c1, d1, e1, a2, b2, c2, d2, e2, np.min(kappa), i]))

results = np.array(results)

max = np.max(results[:, 10])
for index in np.where(results[:, 10] == max)[0]:
    print(results[index])
    a1 = results[index][0]
    b1 = results[index][1]
    c1 = results[index][2]
    d1 = results[index][3]
    e1 = results[index][4]
    a2 = results[index][5]
    b2 = results[index][6]
    c2 = results[index][7]
    d2 = results[index][8]
    e2 = results[index][9]
    x, B, dB_dx = def_field(A, B0, Lo, L1, L2, L6, Lf, a1, b1, c1, d1, e1, a2, b2, c2, d2, e2, 0.001)
    kappa = calc_kappa(gamma, v, B, dB_dx)
    plot_field(x, B, kappa, np.min(kappa))


