"""
Simulates rotation of the spin vector of ultracold neutrons 
travelling through a 3D external guiding field 

Code by Rylan Stutters
Adapted from code by Libertad Barron Palos

"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import argparse

class UCNspinRotSim:
    """Class for running UCN spin polarization simulations
    
    Attributes:
        
    """

    def __init__(self, v, gamma, Bfield, D, yo):
        """Initialize simulation

        Args:
            v (float): speed of UCNs
            gamma (float): gyromagnetic ratio
            Bfield (np.array[float]): B field everywhere in space, of the form 
                                [[pos_x, pos_y, pos_z], [B_x, B_y, B_z]]
            D (float): diameter of the simulation region
            yo (float): length of the simulation region

        """

        # init class attributes
        self.v = v
        self.gamma = gamma
        self.Bfield = Bfield
        self.D = D
        self.yo = yo

    
    def getField(self, pos):
        """Retrieves mag field at specified position

        Args:
            pos (np.array[float]): position to get field, [pos_x, pos_y, pos_z]

        Returns:
            B (np.array[float]): B field components at point pos, [B_x, B_y, B_z]
        """

        # find closest index for corresponding position
        idx = np.argmin(np.linalg.norm(self.Bfield[0] - pos, axis=1))

        return self.Bfield[1][idx]
    

    def plotField(self):
        """3D plots the vector field self.Bfield

        """

        # init plot
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111, projection='3d')

        ax.quiver(self.Bfield[0][:,0], self.Bfield[0][:,1], self.Bfield[0][:,2], 
                  self.Bfield[1][:,0], self.Bfield[1][:,1], self.Bfield[1][:,2],
                  length=0.01, normalize=True)

        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        plt.show()





# Constants
v = 7  # Speed of neutrons in m/s
gamma = 1.832e8  # Gyromagnetic ratio for neutrons in rad/s/T
Bo = 1e-6  # Constant magnetic field in T in the z direction
A = 0.5e-4
L = 0.05
yo = -0.4

# Example field
def B_field(y):
    r = -(yo + L) / 2
    Ao = A - Bo
    # Default magnetic field values in case y does not fall into any condition
    Bx, By, Bz = 0, 0, 0

    if y < yo:  # Condition for y < yo
        By = A
        Bx = 0
        Bz = 0
    elif yo <= y <= yo + L:  # Condition for yo <= y <= yo + L
        By = A * np.cos(np.pi * (y - yo) / (2 * L))
        Bz = A * np.sin(np.pi * (y - yo) / (2 * L))
    elif y > yo + L:  # Condition for y > yo + L
        # For y > yo + L, you should define Bx, By, Bz
        # Assuming no field in the x-direction and using the provided formula for Bz
        By = 0
        Bz = ((Ao / (np.pi * r)) * (r * np.arccos(1 + (y / r)) - np.sqrt(-y * (2 * r + y))) + Bo)
        Bx = 0  # No field in the x-direction for this region

    return np.array([Bx, By, Bz])

# Generate field
pos = []
B = []
x_vals = np.linspace(0, 0.25, 10)
y_vals = np.linspace(-0.5, 0.0, 50)
z_vals = np.linspace(0, 0.25, 10)

for x in x_vals:
    for y in y_vals:
        for z in z_vals:
            pos.append([x, y, z])
            B.append(B_field(y))

sim = UCNspinRotSim(v, gamma, [np.array(pos), np.array(B)], 0.25, yo)

sim.plotField()

