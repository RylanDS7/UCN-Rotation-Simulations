"""
Simulates the path of ultracold neutrons 
and associated values along that path

Code by Rylan Stutters
Adapted from code by Libertad Barron Palos

"""

import numpy as np
from scipy.interpolate import interp1d, RegularGridInterpolator

class UCNpath:
    """Class for simulating UCN paths
    
    Attributes:
        theta (float): the angle in degrees that the simulated nuetron makes with the y axis
        path (np.array[np.array[float]]): the path of the UCN through space
        arc_lengths (np.array[float]): the UCN path parameterized into arclengths along the path
        collisions (np.array[np.array[float]]): the positions along the path that collisions occured
        Bfield_on_path (np.array[np.array[float]]): B field along the UCN path - only non-null if save_Bfield is called
        spins (np.array[np.array[float]]): path evolution of spin vector [spin_x, spin_y, spin_z] - only non-null if save_spins is called
        adiabaticities (np.array[float]) : adiabaticity along path - only non-null if save_adiabaticity is called
    """

    def __init__(self, v, D, yo, yf, upsample_factor):
        """Initialize object

        Args:
            v (float): speed of UCNs
            D (float): diameter of the simulation region
            yo (float): length of the simulation region
            yf (float): ending point of the simulation region
            upsample_factor (int): the factor the path is upsampled beyond 0.005m

        """

        self.simulate_path(v, D, yo, yf, upsample_factor)
        self.parameterize_path()


    def generate_neutron(self, v, D, yo):
        """function to initialize neutron position and velocity
            writes init velocity angle to self.theta
        Args:
            v (float): speed of UCNs
            D (float): diameter of the simulation region
            yo (float): length of the simulation region

        Returns:
            [np.array[float], np.array[float]] : position and velocities of generated neutron
                                                [[pos_x, pos_y, pos_z], [vel_x, vel_y, vel_z]]

        """

        # Randomly generate a position on the circular source
        r = (D / 2) * np.sqrt(np.random.uniform())
        theta = np.random.uniform(0, 2 * np.pi)
        x = r * np.cos(theta)
        z = r * np.sin(theta)
        y = yo # Start at one end of the tube
        
        # Randomly generate a velocity direction
        phi = np.random.uniform(0, 2 * np.pi)
        cos_theta = np.random.uniform(0.6, 1)  # Ensure longitudinal component is >60%
        sin_theta = np.sqrt(1 - cos_theta**2)
        vx = v * sin_theta * np.cos(phi)
        vy = v * cos_theta
        vz = v * sin_theta * np.sin(phi)

        # save theta angle
        self.theta = np.arccos(cos_theta) * 180 / np.pi

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
    

    def simulate_path(self, v, D, yo, yf, upsample_factor):
        """function to simulate the path of one neutron
            writes the path to self.path
            writes the collisions to self.collisions

        Args:
            v (float): speed of UCNs
            gamma (float): gyromagnetic ratio
            D (float): diameter of the simulation region
            yo (float): length of the simulation region
            yf (float): ending point of the simulation region
            upsample_factor (int): the factor the path is upsampled beyond 0.005m

        """
        position, velocity = self.generate_neutron(v, D, yo)
        path_x, path_y, path_z = [position[0]], [position[1]], [position[2]]
        dt = 0.005 / (velocity[1] * upsample_factor)

        collisions = []


        while yo <= position[1] <= yf:  # Keep particle within tube bounds
            # Check if the particle is still within the tube before updating
            if position[1] + velocity[1] * dt > 0:
                # If particle would exceed tube length, stop the simulation
                break
            
            # Update position
            position += velocity * dt

            # Check for wall collision
            distance_to_axis = np.sqrt(position[0]**2 + position[2]**2)
            if distance_to_axis > (D / 2):
                # Reflect the velocity
                velocity = self.reflect(position, velocity)
                # Correct position to ensure it stays inside the tube
                correction_factor = (D / 2) / distance_to_axis
                position[0] *= correction_factor
                position[2] *= correction_factor

                collisions.append(np.copy(position))

            # Record the position
            path_x.append(position[0])
            path_y.append(position[1])
            path_z.append(position[2])

        self.path = np.array([path_x, path_y, path_z])
        self.collisions = np.array(collisions)
    

    def parameterize_path(self):
        """function to parameterize the path into discrete steps
            and save the result in self.arc_lengths

        """
        arc_lengths = np.zeros(self.path.shape[1])
        for i in range(1, self.path.shape[1]):
            arc_lengths[i] = arc_lengths[i - 1] + np.linalg.norm(self.path[:, i] - self.path[:, i - 1])

        self.arc_lengths = arc_lengths


    def save_Bfield(self, Bx_interp, By_interp, Bz_interp):
        """saves the Bfield to self.Bfield_on_path give interpolators for the magnetic field

        Args:
            Bx_interp (RegularGridInterpolator): Interpolator for Bx
            By_interp (RegularGridInterpolator): Interpolator for By
            Bz_interp (RegularGridInterpolator): Interpolator for Bz

        """

        def getField(pos):
            Bx_val = Bx_interp(pos)
            By_val = By_interp(pos)
            Bz_val = Bz_interp(pos)

            return np.array([Bx_val, By_val, Bz_val]).flatten()
    
        Bfield = []

        for i in range(self.path.shape[1]):
            Bfield.append(getField(self.path[:,i]))

        self.Bfield_on_path = np.array(Bfield)

    def save_spins(self, spins):
        """saves the spin evolution to self.spins

        Args:
            spins (np.array[np.array[float]]): path evolution of spin vector [spin_x, spin_y, spin_z]

        """
        self.spins = spins

    def calc_adiabaticity(self, gamma, v):
        """saves the adiabaticities to self.adiabaticities

        Args:
            gamma (float): gyromagnetic ratio
            v (float): speed of the UCNs

        """
        adiabaticities = [0]

        for i in range(1, self.path.shape[1] - 1):
            B_prev = self.Bfield_on_path[i - 1]
            B_current = self.Bfield_on_path[i]
            B_next = self.Bfield_on_path[i + 1]

            ds_prev = self.arc_lengths[i] - self.arc_lengths[i - 1]
            ds_next = self.arc_lengths[i + 1] - self.arc_lengths[i]

            dB_ds = np.linalg.norm(B_next - B_prev) / (ds_prev + ds_next)
            B = np.linalg.norm(B_current)
            kappa = (gamma * B) / (v * dB_ds / B)

            adiabaticities.append(kappa)

        adiabaticities[0] = adiabaticities[1]
        adiabaticities.append(adiabaticities[-1])

        self.adiabaticities = np.array(adiabaticities)
