"""
Code to maximze the lowest adiabaticity of a magnetic field
tapering function using gradient descent optimization algorithms

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
L0 = - 3.75 / 2
L1 = - 3.5 / 2
L2 = - 3 / 2
L3 = - 2.6 / 2
L4 = - 2.55 / 2
L5 = - 2.4 / 2
L6 = - 2.25 / 2
Lf = - 2 / 2

def adiabaticity_of_field(x):
    """sets up the exponential polynomial function for the given parameters

    Args:
        x (ndarray[float]) : input values representing the field values

    Returns:
        float : 1 / (minimum adiabaticity)

    """

    x_vals = np.linspace(L0, Lf, x.shape[0])
    y_vals = x
    dy_dx = np.gradient(y_vals, x_vals)

    kappa = calc_kappa(gamma, v, y_vals, dy_dx)

    return 1 / np.min(kappa)


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
    B = 10**(-6) * B
    dB_dx = 10**(-6) * dB_dx
    kappa = np.zeros(len(B))
    for i in range(len(B)):
        if (np.abs(dB_dx[i]) == 0 or B[i] == A):
            kappa[i] = 10**7
        else:
            kappa[i] = (gamma * B[i]**2) / (v * np.abs(dB_dx[i]))
            if kappa[i] > 10**7:
                kappa[i] = 10**7

    return kappa


def plot_field(x, B, kappa, lowest_kappa):
    fig, ax = plt.subplots(2, 1, figsize=(12, 12))

    ax[0].plot(x, B, label="Bz", color="blue")
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


x0 = np.loadtxt("starting_field.txt", delimiter=",")

lower_bnds = np.zeros(x0.shape[0])
lower_bnds[0] = A
lower_bnds[-1] = B0

upper_bnds = A * np.ones(x0.shape[0])
upper_bnds[0] = A
upper_bnds[-1] = B0

bnds = scipy.optimize.Bounds(lower_bnds, upper_bnds)

def callbackF(xk):
    print(f"Current kappa: {1 / adiabaticity_of_field(xk)}")
res = scipy.optimize.minimize(adiabaticity_of_field, x0, 
                            method='L-BFGS-B',
                            bounds=bnds,
                            options={'maxiter': 50, 'disp': True},
                            callback=callbackF
                            )


print(adiabaticity_of_field(res.x))

x_vals = np.linspace(L0, Lf, res.x.shape[0])
dy_dx = np.gradient(res.x, x_vals)
kappa = calc_kappa(gamma, v, res.x, dy_dx)
plot_field(x_vals, res.x, kappa, np.min(kappa))

