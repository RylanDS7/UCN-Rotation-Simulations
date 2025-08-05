"""
Simulates ultracold neutron trajectories in a cylindrical tube

Code by Libertad Barron Palos:

"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import argparse

# Constants
v_neutrons = 8  # Velocity in m/s
dt = 0.001  # Smaller time step in seconds

# Parse command line arguments
parser = argparse.ArgumentParser(description="Simulate Neutron Trajectories Inside the Guide")
parser.add_argument('--yo', type=float, required=True, help="y-offset for the field region in m")
parser.add_argument('--di', type=float, required=True, help="internal neutron guide diameter")
parser.add_argument('--nn', type=int, required=True, help="number of neutrons")
args = parser.parse_args()

# Unpack the arguments
tube_length = args.yo  # Length in m
tube_radius = args.di / 2  # Radius in m
n_particles = args.nn  # Number of particles

# Function to generate a random particle with velocity constraint
def generate_particle():
    rejection_count = 0
    while True:
        rejection_count += 1
        # Randomly generate a position on the circular source
        r = tube_radius * np.sqrt(np.random.uniform())
        theta = np.random.uniform(0, 2 * np.pi)
        x = r * np.cos(theta)
        z = r * np.sin(theta)
        y = 0  # Start at one end of the tube
        
        # Randomly generate a velocity direction
        phi = np.random.uniform(0, 2 * np.pi)
        cos_theta = np.random.uniform(0.6, 1)  # Ensure longitudinal component is >60%
        sin_theta = np.sqrt(1 - cos_theta**2)
        vx = v_neutrons * sin_theta * np.cos(phi)
        vy = v_neutrons * cos_theta
        vz = v_neutrons * sin_theta * np.sin(phi)
        
        if vy > 0:  # Ensure positive y-direction
            return np.array([x, y, z]), np.array([vx, vy, vz])
        
        # Print how many particles are rejected (for debugging)
        if rejection_count % 1000 == 0:  # Print every 1000 rejected particles
            print(f"Rejected {rejection_count} particles")

# Function to calculate specular reflection
def reflect(position, velocity):
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

# Simulate particle paths
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')

valid_particles = 0  # Keep track of the number of valid particles

while valid_particles < n_particles:
    position, velocity = generate_particle()  # Generate valid particle
    path_x, path_y, path_z = [position[0]], [position[1]], [position[2]]
    
    while 0 <= position[1] <= tube_length:  # Keep particle within tube bounds
        # Check if the particle is still within the tube before updating
        if position[1] + velocity[1] * dt > tube_length:
            # If particle would exceed tube length, stop the simulation
            break
        
        # Update position
        position += velocity * dt

        # Check for wall collision
        distance_to_axis = np.sqrt(position[0]**2 + position[2]**2)
        if distance_to_axis > tube_radius:
            # Reflect the velocity
            velocity = reflect(position, velocity)
            # Correct position to ensure it stays inside the tube
            correction_factor = tube_radius / distance_to_axis
            position[0] *= correction_factor
            position[2] *= correction_factor

        # Record the position
        path_x.append(position[0])
        path_y.append(position[1])
        path_z.append(position[2])

    # Plot the path if it's a valid particle
    ax.plot(path_x, path_y, path_z, alpha=0.7)
    valid_particles += 1

# Cylinder visualization
theta = np.linspace(0, 2 * np.pi, 100)
z = np.linspace(0, tube_length, 100)
theta, z = np.meshgrid(theta, z)
x_cylinder = tube_radius * np.cos(theta)
y_cylinder = z
z_cylinder = tube_radius * np.sin(theta)

ax.plot_surface(x_cylinder, y_cylinder, z_cylinder, color='cyan', alpha=0.1)

# Labels and show plot
ax.set_xlabel("X (m)")
ax.set_ylabel("Y (m)")
ax.set_zlabel("Z (m)")
ax.set_title("Neutron Paths in a Cylindrical Tube")
plt.show()