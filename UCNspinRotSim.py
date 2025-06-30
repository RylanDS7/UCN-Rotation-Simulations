"""
Simulates rotation of the spin vector of ultracold neutrons 
travelling through a 3D external guiding field 

Code by Rylan Stutters
Adapted from code by Libertad Barron Palos

"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from tqdm import tqdm

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
    

    def plot_Field(self):
        """3D plots the vector field self.Bfield

        """

        # init plot
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111, projection='3d')

        # Position and vector components
        X, Y, Z = self.Bfield[0][:, 0], self.Bfield[0][:, 1], self.Bfield[0][:, 2]
        U, V, W = self.Bfield[1][:, 0], self.Bfield[1][:, 1], self.Bfield[1][:, 2]

        # scalar field for coloring (e.g., magnitude of the vector)
        C = np.linalg.norm(self.Bfield[1], axis=1)

        # Normalize vectors for plotting
        B_norm = np.stack((U, V, W), axis=1)
        B_norm = B_norm / np.linalg.norm(B_norm, axis=1)[:, np.newaxis]

        ax.quiver(X, Y, Z,
                  B_norm[:,0], B_norm[:,1], B_norm[:,2],
                  length=1E-3, normalize=True, colors=plt.cm.viridis(C / C.max()))

        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        ax.legend()
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
        y = self.yo # Start at one end of the tube
        
        # Randomly generate a velocity direction
        phi = np.random.uniform(0, 2 * np.pi)
        cos_theta = np.random.uniform(0.75, 1)  # Ensure longitudinal component is >75%
        sin_theta = np.sqrt(1 - cos_theta**2)
        vx = self.v * sin_theta * np.cos(phi)
        vy = self.v * cos_theta
        vz = self.v * sin_theta * np.sin(phi)

        return np.array([x, y, z]), np.array([vx, vy, vz])
        
    
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
        dt = 0.0001


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
        """3D plots a neutron path and the spin changes over the path

        Args:
            path (np.array[np.array[float]]): path of simulated neutron [path_x, path_y, path_z]

        """

        path_x = path[0]
        path_y = path[1]
        path_z = path[2]

        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111, projection='3d')
        ax.plot(path_x, path_y, path_z, alpha=0.7)
        
        spin = self.solve_spins(np.array([0, 1, 0]), path)
        ax.quiver(path_x, path_y, path_z, spin[0], spin[1], spin[2], length=0.01, normalize = True)

        # Cylinder visualization
        theta = np.linspace(0, 2 * np.pi, 100)
        z = np.linspace(self.yo, 0, 100)
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

        # Plot the field and spin graphs
        _, axs = plt.subplots(2, 1, figsize=(12, 12), sharex=True)

        path_field = np.array([self.getField(path[:, i]) for i in range(path.shape[1])]).T

        # First plot for magnetic field components
        axs[0].plot(path[1], path_field[0] * 1e6, label="Bx", color="blue")  # Convert to µT
        axs[0].plot(path[1], path_field[1] * 1e6, label="By", color="orange")  # Convert to µT
        axs[0].plot(path[1], path_field[2] * 1e6, label="Bz", color="green")  # Convert to µT
        axs[0].set_ylabel("Magnetic Field (μT)")
        axs[0].legend()
        axs[0].grid(True, which='both', axis='both')

        # Second plot for spin components
        axs[1].plot(path[1], spin[0], label="Sx", color="red")
        axs[1].plot(path[1], spin[1], label="Sy", color="blue")
        axs[1].plot(path[1], spin[2], label="Sz", color="green")
        axs[1].set_ylabel("Spin Components")
        axs[1].legend()
        axs[1].grid(True, which='both', axis='both')

        plt.show()


    def bloch_eq(self, pos, S):
        """The bloch equation

        Args:
            pos (float): current position
            S (np.array[float]): current spin vector [S_x, S_y, S_z]

        Returns:
            float : the result of the bloch equation as ds_dr

        """
        B = self.getField(pos)
        dS = self.gamma * np.cross(S, B)
        return dS / self.v  # Normalize by velocity since dy = v * dt
    
    def solve_spins(self, S0, path):
        """Solves the spin evolution for a given neutron path

        Args:
            S0 (np.array[float]): initial spin vector
            path (np.array[np.array[float]]): path of simulated neutron [path_x, path_y, path_z]

        Returns:
            np.array[np.array[float]] : path evolution of spin vector [spin_x, spin_y, spin_z]

        """
        # Interpolation function for position along arc length
        def interpolate_position(s):
            return np.array([
                np.interp(s, arc_lengths, path[0]),
                np.interp(s, arc_lengths, path[1]),
                np.interp(s, arc_lengths, path[2])
            ])
        
        # ODE: dS/ds = bloch_eq(r(s), S)
        def dS_ds(s, S):
            r = interpolate_position(s)
            return self.bloch_eq(r, S)

        # Parameterize path by arc length
        arc_lengths = np.zeros(path.shape[1])
        for i in range(1, path.shape[1]):
            arc_lengths[i] = arc_lengths[i-1] + np.linalg.norm(path[:, i] - path[:, i-1])

        # progress bar for ivp solution
        pbar = tqdm(total=len(arc_lengths), desc="Solving IVP")
        current_index = [0]

        # wrapper of dS_ds with updating progress
        def ivp(t, y):
            # only update when we pass next arc_length step
            while (current_index[0] < len(arc_lengths)) and (t >= arc_lengths[current_index[0]]):
                pbar.update(1)
                current_index[0] += 1
            return dS_ds(t, y)
    
        # Solve ivp
        solution = solve_ivp(ivp, [arc_lengths[0], arc_lengths[-1]], S0, t_eval=arc_lengths, method='RK45')

        return np.array(solution.y)
