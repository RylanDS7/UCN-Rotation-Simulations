"""
Simulates rotation of the spin vector of ultracold neutrons 
travelling through a 3D external guiding field 

Code by Rylan Stutters
Adapted from code by Libertad Barron Palos

"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d, RegularGridInterpolator
from tqdm import tqdm

class UCNspinRotSim:
    """Class for running UCN spin polarization simulations
    
    Attributes:
        v (float): UCN average speed
        gamma (float): gyromagnetic ratio
        Bfield (np.array[np.array[float]], np.array[np.array[float]]): B field everywhere in space, of the form 
            [[pos_x, pos_y, pos_z], [B_x, B_y, B_z]]
        D (float): diameter of the simulation region
        yo (float): starting point of the simulation region
        yf (float): ending point of the simulation region
        
    """

    def __init__(self, v, gamma, Bfield, D, yo, yf):
        """Initialize simulation

        Args:
            v (float): speed of UCNs
            gamma (float): gyromagnetic ratio
            Bfield (np.array[np.array[float]], np.array[np.array[float]]): B field everywhere in space, of the form 
                                [[pos_x, pos_y, pos_z], [B_x, B_y, B_z]]
            D (float): diameter of the simulation region
            yo (float): length of the simulation region
            yf (float): ending point of the simulation region

        """

        # init class attributes
        self.v = v
        self.gamma = gamma
        self.Bfield = Bfield
        self.D = D
        self.yo = yo
        self.yf = yf
        self.setup_interpolator()


    def setup_interpolator(self):
        """Initialize mag field interpolator

        """
        positions = self.Bfield[0]
        Bvectors = self.Bfield[1]

        x_unique = np.unique(positions[:, 0])
        y_unique = np.unique(positions[:, 1])
        z_unique = np.unique(positions[:, 2])

        expected_size = len(x_unique) * len(y_unique) * len(z_unique)
        assert positions.shape[0] == expected_size, "Data points do not form a complete grid"

        dtype = [('x', float), ('y', float), ('z', float)]
        structured_positions = np.array([tuple(pos) for pos in positions], dtype=dtype)
        sorted_indices = np.argsort(structured_positions, order=['x', 'y', 'z'])

        positions_sorted = positions[sorted_indices]
        Bvectors_sorted = Bvectors[sorted_indices]

        Bx = Bvectors_sorted[:, 0].reshape(len(x_unique), len(y_unique), len(z_unique))
        By = Bvectors_sorted[:, 1].reshape(len(x_unique), len(y_unique), len(z_unique))
        Bz = Bvectors_sorted[:, 2].reshape(len(x_unique), len(y_unique), len(z_unique))

        from scipy.interpolate import RegularGridInterpolator
        self.B_interp_x = RegularGridInterpolator((x_unique, y_unique, z_unique), Bx)
        self.B_interp_y = RegularGridInterpolator((x_unique, y_unique, z_unique), By)
        self.B_interp_z = RegularGridInterpolator((x_unique, y_unique, z_unique), Bz)


    def getField(self, pos):
        """Retrieve the field at a position using the grid interpolator

        Args:
            pos (np.array[float]): current position of particle, [pos_x, pos_y, pos_z]

        Returns:
            np.array[float] : interpolated magnetic field at pos [Bx, By, Bz]

        """
        Bx_val = self.B_interp_x(pos)
        By_val = self.B_interp_y(pos)
        Bz_val = self.B_interp_z(pos)

        return np.array([Bx_val, By_val, Bz_val]).flatten()
    

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
        plt.show()


    def generate_neutron(self):
        """function to initialize neutron position and velocity

        Returns:
            [np.array[float], np.array[float]] : position and velocities of generated neutron
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
        cos_theta = np.random.uniform(0.6, 1)  # Ensure longitudinal component is >75%
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
        dt = 0.005 / velocity[1]


        while self.yo <= position[1] <= self.yf:  # Keep particle within tube bounds
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
    

    def plot_path(self, path, spin):
        """3D plots a neutron path and the spin changes over the path

        Args:
            path (np.array[np.array[float]]): path of simulated neutron [path_x, path_y, path_z]
            spin (np.array[np.array[float]]): path evolution of spin vector [spin_x, spin_y, spin_z]

        """

        path_x = path[0]
        path_y = path[1]
        path_z = path[2]

        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111, projection='3d')
        ax.plot(path_x, path_y, path_z, alpha=0.7)
        
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

        plt.show()


    def plot_spins(self, path, spin):
        """Plots a graph of magnetic field experienced along the path
            and spin evolution as a function of path y component

        Args:
            path (np.array[np.array[float]]): path of simulated neutron [path_x, path_y, path_z]
            spin (np.array[np.array[float]]): path evolution of spin vector [spin_x, spin_y, spin_z]

        """

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
            pos (np.array[float]): current position [x, y, z]
            S (np.array[float]): current spin vector [S_x, S_y, S_z]

        Returns:
            float : the result of the bloch equation as ds_dr

        """
        B = self.getField(pos)
        dS_dt = self.gamma * np.cross(S, B)
        return dS_dt / self.v # Normalize by velocity since dy = v * dt
    
    
    def upsample_path(self, path, upsample_factor):
        """Given a path (3xN), upsample by linear interpolation by a factor of upsample_factor.
        
        Args:
            path: 3xN np.array of positions
            upsample_factor: int, number of times to increase resolution
        
        Returns:
            path_upsampled: 3xM np.array, with M = (N-1)*upsample_factor + 1
        """
        N = path.shape[1]
        # Original arc length parameterization
        arc_lengths = np.zeros(N)
        for i in range(1, N):
            arc_lengths[i] = arc_lengths[i-1] + np.linalg.norm(path[:, i] - path[:, i-1])
        
        # New finer arc length parameterization
        total_length = arc_lengths[-1]
        M = (N - 1) * upsample_factor + 1
        arc_fine = np.linspace(0, total_length, M)
        
        # Interpolate each coordinate independently
        interp_funcs = [interp1d(arc_lengths, path[i], kind='linear') for i in range(3)]
        
        path_fine = np.vstack([f(arc_fine) for f in interp_funcs])
        
        return path_fine
        

    def solve_spins(self, S0, path, upsample_factor):        
        """Solves the spin evolution for a given neutron path and updates a progress bar

        Args:
            S0 (np.array[float]): initial spin vector
            path (np.array[np.array[float]]): path of simulated neutron [path_x, path_y, path_z]
            upsample_factor (int): number of times to increase resolution of the path

        Returns:
            np.array[np.array[float]] : path of simulated neutron with upsampling [path_x, path_y, path_z]
            np.array[np.array[float]] : path evolution of spin vector [spin_x, spin_y, spin_z]

        """
        # upsample path for smaller step size
        path_fine = self.upsample_path(path, upsample_factor)

        # get array of step positions along the path
        arc_lengths = np.zeros(path_fine.shape[1])
        for i in range(1, path_fine.shape[1]):
            arc_lengths[i] = arc_lengths[i - 1] + np.linalg.norm(path_fine[:, i] - path_fine[:, i - 1])

        # interpolate the bloch equation along the path
        interp_funcs = [interp1d(arc_lengths, path_fine[i], kind='linear', fill_value="extrapolate") for i in range(3)]
        def get_pos(s):
            return np.array([f(s) for f in interp_funcs])
        def bloch_at(s, S):  # Bloch equation evaluated on arc-length
            return self.bloch_eq(get_pos(s), S)

        # setup and solve spins iteratively
        S = np.zeros((3, len(arc_lengths)))
        S[:, 0] = S0

        # progress bar for solution
        pbar = tqdm(total=len(arc_lengths), desc="Solving")

        for i in range(1, len(arc_lengths)):
            ds = arc_lengths[i] - arc_lengths[i - 1]
            s_prev = arc_lengths[i - 1]
            S_prev = S[:, i - 1]

            # RK4 step
            k1 = bloch_at(s_prev, S_prev)
            k2 = bloch_at(s_prev + ds/2, S_prev + ds/2 * k1)
            k3 = bloch_at(s_prev + ds/2, S_prev + ds/2 * k2)
            k4 = bloch_at(s_prev + ds, S_prev + ds * k3)

            pbar.update(1)

            # RK4 evaluation by averaging slopes
            S_next = S_prev + ds * (k1 + 2*k2 + 2*k3 + k4) / 6
            S[:, i] = S_next / np.linalg.norm(S_next)

        return path_fine, S

