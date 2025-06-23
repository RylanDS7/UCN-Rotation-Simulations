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
            Bfield (np.array[np.array[float]], np.array[np.array[float]]): B field everywhere in space, of the form 
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


    def generate_neutron(self):
        """function to init neutron position and velocity

        Returns:
            [np.array[float], np.array[float] : position and velocities of generated neutron
                                                [[pos_x, pos_y, pos_z], [vel_x, vel_y, vel_z]]

        """

        # Randomly generate a position on the circular source
        r = (self.D / 2) * np.sqrt(np.random.uniform())
        theta = np.random.uniform(0, 2 * np.pi)
        x = r * np.cos(theta)
        z = r * np.sin(theta)
        y = self.yo  # Start at one end of the tube
        
        # Randomly generate a velocity direction
        phi = np.random.uniform(0, 2 * np.pi)
        cos_theta = np.random.uniform(0.6, 1)  # Ensure longitudinal component is >60%
        sin_theta = np.sqrt(1 - cos_theta**2)
        vx = self.v * sin_theta * np.cos(phi)
        vy = self.v * cos_theta
        vz = self.v * sin_theta * np.sin(phi)
        
        if vy > 0:  # Ensure positive y-direction
            return np.array([x, y, z]), np.array([vx, vy, vz])
        else:
            return self.generate_neutron()
        
    
    def reflect(self, position, velocity):
        """Function to calculate specular reflection

        Args:
            position (np.array[float]): current position of particle, [pos_x, pos_y, pos_z]
            velocity (np.array[float]): current velocity of particle, [vel_x, vel_y, vel_z]

        Returns:
            np.array[float] : new velocity of reflected particle [vel_x, vel_y, vel_z]

        """
        x, y, z = position
        vx, vy, vz = velocity
        # Normal to the cylindrical wall
        normal = np.array([x, 0, z]) / np.sqrt(x**2 + z**2)
        # Decompose velocity
        v_parallel = velocity - np.dot(velocity, normal) * normal
        v_perpendicular = np.dot(velocity, normal) * normal
        # Invert the perpendicular component for specular reflection
        new_velocity = v_parallel - v_perpendicular
        return new_velocity
    

    def simulate_path(self):
        """function to simulate the path of one neutron

        Returns:
            np.array[np.array[float]] : path of simulated neutron [path_x, path_y, path_z]

        """
        position, velocity = self.generate_neutron()
        path_x, path_y, path_z = [position[0]], [position[1]], [position[2]]
        dt = 0.001


        while self.yo <= position[1] <= 0:  # Keep particle within tube bounds
            # Check if the particle is still within the tube before updating
            if position[1] + velocity[1] * dt > 0:
                # If particle would exceed tube length, stop the simulation
                break
            
            # Update position
            position += velocity * dt

            # Check for wall collision
            distance_to_axis = np.sqrt(position[0]**2 + position[2]**2)
            if distance_to_axis > (self.D / 2):
                # Reflect the velocity
                velocity = self.reflect(position, velocity)
                # Correct position to ensure it stays inside the tube
                correction_factor = (self.D / 2) / distance_to_axis
                position[0] *= correction_factor
                position[2] *= correction_factor

            # Record the position
            path_x.append(position[0])
            path_y.append(position[1])
            path_z.append(position[2])

        return np.array([path_x, path_y, path_z])
    

    def plot_path(self, path):
        """3D plots a neutron path

        Args:
            path (np.array[np.array[float]]): path of simulated neutron [path_x, path_y, path_z]

        """

        path_x = path[0]
        path_y = path[1]
        path_z = path[2]

        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111, projection='3d')
        ax.plot(path_x, path_y, path_z, alpha=0.7)

        # Cylinder visualization
        theta = np.linspace(0, 2 * np.pi, 100)
        z = np.linspace(yo, 0, 100)
        theta, z = np.meshgrid(theta, z)
        x_cylinder = (self.D / 2) * np.cos(theta)
        y_cylinder = z
        z_cylinder = (self.D / 2) * np.sin(theta)

        ax.plot_surface(x_cylinder, y_cylinder, z_cylinder, color='cyan', alpha=0.1)

        # Labels and show plot
        ax.set_xlabel("X (m)")
        ax.set_ylabel("Y (m)")
        ax.set_zlabel("Z (m)")
        ax.set_title("Neutron Paths in a Cylindrical Tube")
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
x_vals = np.linspace(-0.05, 0.05, 10)
y_vals = np.linspace(-1, 0.0, 50)
z_vals = np.linspace(-0.05, 0.05, 10)

for x in x_vals:
    for y in y_vals:
        for z in z_vals:
            pos.append([x, y, z])
            B.append(B_field(y))

sim = UCNspinRotSim(v, gamma, [np.array(pos), np.array(B)], 0.095, yo)

sim.plot_path(sim.simulate_path())

