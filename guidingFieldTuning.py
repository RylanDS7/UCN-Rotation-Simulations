"""
Script to iterate through parameters for guiding magnetic fields
to determine the field that best preserves UCN spin

Code by Rylan Stutters
"""

import numpy as np
import matplotlib.pyplot as plt

A = 0.00005 # init amplitude of field
v = 7  # Speed of neutrons in m/s
gamma = 1.832e8  # Gyromagnetic ratio for neutrons in rad/s/T
B0 = 10**(-6)  # Constant magnetic field in T in the z direction
a = 10**(-6)

def polynomial_field(A, B0, Po, P1, P2, Pf, a1, b1, c1, a2, b2, c2, dt):
    """sets up the polynomial for the given parameters

    Args:
        A (float): Initial vertical field magnitude
        B0 (float): Constant magnetic field in T in the z direction
        Po (float): The x value that the field starts at
        P1 (float): The x value that the first polynomial starts at
        P2 (float): The x value that the second polynomial starts at
        Pf (float): The x value that the field ends at
        a1 (float): parameter 1 of the first polynomial
        b1 (float): parameter 2 of the first polynomial
        c1 (float): parameter 3 of the first polynomial
        a2 (float): parameter 1 of the second polynomial
        b2 (float): parameter 2 of the second polynomial
        c2 (float): parameter 3 of the second polynomial
        dt (float): the time step over which the condsider the field

    Returns:
        np.array[float]: The x values along the field
        np.array[float]: The vertical field values along the path
        np.array[float]: The derivative of the field at every point

    """
    # compile x values
    x_vals = np.linspace(Po, Pf * 1.2, int(((Pf * 1.2) - Po) / dt))

    # init y values
    y_vals = np.zeros(len(x_vals))

    for i in range(len(x_vals)):
        if (Po <= x_vals[i] <= P1): # first section
            y_vals[i] = A
        elif (P1 < x_vals[i] <= P2): # second section
            x = x_vals[i] - P1
            y_vals[i] = a1 * x**3 + b1 * x**2 + c1 * x + A
            P2_index = i # tracks the switching index to P2
        elif (P2 < x_vals[i] <= Pf): # third section
            x = x_vals[i] - P2
            y_vals[i] = a2 * x**3 + b2 * x**2 + c2 * x + y_vals[P2_index]
        else:
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

    return kappa


def plot_field(x, B, kappa):
    fig, ax = plt.subplots(2, 1, figsize=(12, 12))

    ax[0].plot(x, B * 1e6, label="Bz", color="blue")
    ax[0].legend()
    ax[0].grid(True)
    ax[0].set_ylabel("Magnetic Field (μT)", fontsize=20)

    ax[1].plot(x, kappa, label="Adiabaticity", color="purple")
    ax[1].set_yscale("log")
    ax[1].set_xlabel("Axial Position along the Path (m)", fontsize=20)
    ax[1].set_ylabel("Adiabaticity κ", fontsize=20)
    ax[1].legend()
    ax[1].grid(True)

    plt.show()



x, B, dB_dx = polynomial_field(A, B0, 0, 2, 3, 5, 59*a, -89*a, 0, -2*a, 12.75*a, -27*a, 0.001)
kappa = calc_kappa(gamma, v, B, dB_dx)
plot_field(x, B, kappa)





