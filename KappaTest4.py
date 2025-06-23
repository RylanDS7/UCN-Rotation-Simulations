"""
Simulates rotation of the spin vector of ultracold neutrons by an external guiding field

Code by Libertad Barron Palos

"""


import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import argparse

# Constants
v = 7  # Speed of neutrons in m/s
gamma = 1.832e8  # Gyromagnetic ratio for neutrons in rad/s/T
Bo = 1e-6  # Constant magnetic field in T in the z direction

# Parse command line arguments
parser = argparse.ArgumentParser(description="Simulate Spin Evolution and Adiabaticity Parameter")
parser.add_argument('--A', type=float, required=True, help="Magnetic field amplitude in T")
parser.add_argument('--L', type=float, required=True, help="Length of the field region in m")
parser.add_argument('--yo', type=float, required=True, help="y-offset for the field region in m")
args = parser.parse_args()

# Unpack the arguments
A = args.A  # Magnetic field amplitude in T
L = args.L  # Length of the field region in m
yo = args.yo  # y-offset in m

# Define magnetic field components and their derivatives
def B_field_and_derivative(y):
    r = -(yo + L) / 2
    Ao = A - Bo
    # Default magnetic field values in case y does not fall into any condition
    Bx, By, Bz = 0, 0, 0
    dBx_dy, dBy_dy, dBz_dy = 0, 0, 0

    if y < yo:  # Condition for y < yo
        By = A
        Bx = 0
        Bz = 0
        dBy_dy = 0
        dBx_dy = 0
        dBz_dy = 0
    elif yo <= y <= yo + L:  # Condition for yo <= y <= yo + L
        By = A * np.cos(np.pi * (y - yo) / (2 * L))
        Bz = A * np.sin(np.pi * (y - yo) / (2 * L))
        dBy_dy = -A * (np.pi / (2 * L)) * np.sin(np.pi * (y - yo) / (2 * L))
        dBz_dy = A * (np.pi / (2 * L)) * np.cos(np.pi * (y - yo) / (2 * L))
    elif y > yo + L:  # Condition for y > yo + L
        # For y > yo + L, you should define Bx, By, Bz
        # Assuming no field in the x-direction and using the provided formula for Bz
        By = 0
        Bz = ((Ao / (np.pi * r)) * (r * np.arccos(1 + (y / r)) - np.sqrt(-y * (2 * r + y))) + Bo)
        Bx = 0  # No field in the x-direction for this region
        # Numerical derivative for Bz approximation
        delta_y = 1e-6
        Bz_forward = ((Ao / (np.pi * r)) * (r * np.arccos(1 + ((y + delta_y) / r)) - np.sqrt(-(y + delta_y) * (2 * r + (y + delta_y)))) + Bo)
        dBy_dy = 0
        dBz_dy = (Bz_forward - Bz) / delta_y

    return np.array([Bx, By, Bz]), np.array([dBx_dy, dBy_dy, dBz_dy])

# Bloch equation
def bloch_eq(y, S):
    B, _ = B_field_and_derivative(y)
    dS = gamma * np.cross(S, B)
    return dS / v  # Normalize by velocity since dy = v * dt

# Initial spin vector
S0 = np.array([0, 1, 0])

# Define the range for y
y_start = yo - 0.05  # Start 5 cm before yo
y_end = 0  # End at y = 0
y_vals = np.linspace(y_start, y_end, 500)

# Solve the Bloch equations
solution = solve_ivp(bloch_eq, [y_start, y_end], S0, t_eval=y_vals, method='RK45')

# Extract spin components
Sx, Sy, Sz = solution.y

# Calculate the adiabaticity parameter kappa
kappa_vals = []
Bx_vals, By_vals, Bz_vals = [], [], []

for y in y_vals:
    B, dB_dy = B_field_and_derivative(y)
    B_magnitude = np.linalg.norm(B)
    dB_dy_magnitude = np.linalg.norm(dB_dy)
    Bx_vals.append(B[0])
    By_vals.append(B[1])
    Bz_vals.append(B[2])
    if B_magnitude != 0:
        kappa = abs(gamma * B_magnitude) / (v*dB_dy_magnitude / B_magnitude)
    else:
        kappa = 0
    kappa_vals.append(kappa)

kappa_vals = np.array(kappa_vals)

# Compute the minimum value of kappa, ignoring NaN values
kappa_min = np.nanmin(kappa_vals)  # This will ignore NaN values


# Compute the ratio of final to initial spin
idx_initial = np.argmin(np.abs(y_vals - yo))
idx_final = np.argmin(np.abs(y_vals - (yo + L)))
idx_yo_L = np.argmin(np.abs(y_vals - (yo + L)))
idx_0 = np.argmin(np.abs(y_vals - 0))

Sy_initial = Sy[idx_initial]
Sz_final = Sz[idx_final]
Sz_at_yo_L = Sz[idx_yo_L]
Sz_at_0 = Sz[idx_0]

# Ratios
ratio_rot = Sz_final / Sy_initial if Sy_initial != 0 else 0
ratio_tap = Sz_at_0 / Sz_at_yo_L if Sz_at_yo_L != 0 else 0

# Ratio text with LaTeX-style formatting
ratio_text_rot = f"$\\left(P_n^f / P_n^i\\right)^{{rotation}} = {ratio_rot:.3f}$"
ratio_text_tap = f"$\\left(P_n^f / P_n^i\\right)^{{tapper}} = {ratio_tap:.3f}$"

# Plot the results
fig, axs = plt.subplots(3, 1, figsize=(12, 12), sharex=True)

# First plot for magnetic field components
axs[0].plot(y_vals, np.array(Bx_vals) * 1e6, label="Bx", color="blue")  # Convert to µT
axs[0].plot(y_vals, np.array(By_vals) * 1e6, label="By", color="orange")  # Convert to µT
axs[0].plot(y_vals, np.array(Bz_vals) * 1e6, label="Bz", color="green")  # Convert to µT
axs[0].axvspan(yo, yo + L, color='gray', alpha=0.3, label="Rotation Region")
axs[0].set_ylabel("Magnetic Field (μT)")
axs[0].legend()
axs[0].grid(True, which='both', axis='both')

# Second plot for spin components
axs[1].plot(y_vals, Sx, label="Sx", color="red")
axs[1].plot(y_vals, Sy, label="Sy", color="blue")
axs[1].plot(y_vals, Sz, label="Sz", color="green")
axs[1].axvspan(yo, yo + L, color='gray', alpha=0.3, label="Rotation Region")
axs[1].set_ylabel("Spin Components")
axs[1].text(0.5, 0.8, ratio_text_rot, horizontalalignment='center', verticalalignment='center', transform=axs[1].transAxes, fontsize=12)
axs[1].text(0.5, 0.7, ratio_text_tap, horizontalalignment='center', verticalalignment='center', transform=axs[1].transAxes, fontsize=12)
axs[1].legend()
axs[1].grid(True, which='both', axis='both')

# Third plot for the adiabaticity parameter (kappa)
axs[2].plot(y_vals, kappa_vals, label="Adiabaticity Parameter κ", color="purple")
axs[2].axvspan(yo, yo + L, color='gray', alpha=0.3, label="Rotation Region")
axs[2].set_xlabel("y (m)")
axs[2].set_ylabel("κ (Adiabaticity Parameter)")
axs[2].legend()
axs[2].grid(True, which='both', axis='both')

# Add text annotation for kappa_min
axs[2].text(0.95, 0.95, f"$\\kappa_{{min}} = {kappa_min:.3f}$", horizontalalignment='right', verticalalignment='top', transform=axs[2].transAxes, fontsize=12, color='black')

# Adjust layout and display plots
plt.tight_layout()

# Save the plot as a PDF
plt.savefig("plot_output.pdf", dpi=300)  # Save as PDF with 300 DPI resolution

# Show the plot
plt.show()