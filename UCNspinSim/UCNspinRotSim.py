"""
Simulates rotation of the spin vector of ultracold neutrons 
travelling through a 3D external guiding field 

Code by Rylan Stutters

"""

import UCNpath as path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from scipy.interpolate import RegularGridInterpolator
from tqdm import tqdm

class UCNspinRotSim:
    """Class for running UCN spin polarization simulations
    
    Attributes:
        gamma (float): gyromagnetic ratio
        v (float): speed of the UCNs
        Bfield (np.array[np.array[float]], np.array[np.array[float]]): B field everywhere in space, of the form 
            [[pos_x, pos_y, pos_z], [B_x, B_y, B_z]]
        B_interp_x (RegularGridInterpolator): Interpolator for Bx
        B_interp_y (RegularGridInterpolator): Interpolator for By
        B_interp_z (RegularGridInterpolator): Interpolator for Bz
        UCNpaths (np.array[UCNpath]): The array of path objects for the UCN paths
        
    """

    def __init__(self, gamma, Bfield, num_paths, v, D, yo, yf, upsample_factor):
        """Initialize simulation

        Args:
            gamma (float): gyromagnetic ratio
            Bfield (np.array[np.array[float]], np.array[np.array[float]]): B field everywhere in space, of the form 
                                [[pos_x, pos_y, pos_z], [B_x, B_y, B_z]]
            num_paths: The number of UCN paths to generate
            v (float): speed of UCNs
            D (float): diameter of the simulation region
            yo (float): length of the simulation region
            yf (float): ending point of the simulation region
            upsample_factor (int): the factor the path is upsampled beyond 0.005m

        """

        # init class attributes
        self.gamma = gamma
        self.v = v
        self.Bfield = Bfield

        self.setup_interpolator()
        self.init_paths(num_paths, v, D, yo, yf, upsample_factor)


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

        self.B_interp_x = RegularGridInterpolator((x_unique, y_unique, z_unique), Bx)
        self.B_interp_y = RegularGridInterpolator((x_unique, y_unique, z_unique), By)
        self.B_interp_z = RegularGridInterpolator((x_unique, y_unique, z_unique), Bz)

    def init_paths(self, num_paths, v, D, yo, yf, upsample_factor):
        """Initialize UCN paths

        Args:
            num_paths: The number of UCN paths to generate
            v (float): speed of UCNs
            D (float): diameter of the simulation region
            yo (float): length of the simulation region
            yf (float): ending point of the simulation region
            upsample_factor (int): the factor the path is upsampled beyond 0.005m

        """
        UCNpaths = []
        for i in range(num_paths):
            newPath = path.UCNpath(v, D, yo, yf, upsample_factor)
            newPath.save_Bfield(self.B_interp_x, self.B_interp_y, self.B_interp_z)
            newPath.calc_adiabaticity(self.gamma, v)
            UCNpaths.append(newPath)

        self.UCNpaths = np.array(UCNpaths)



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


    def bloch_eq(self, B, S):
        """The bloch equation

        Args:
            B (np.array[float]): current Bfield vector [Bx, By, Bz]
            S (np.array[float]): current spin vector [S_x, S_y, S_z]

        Returns:
            float : the result of the bloch equation as ds_dr

        """
        dS_dt = self.gamma * np.cross(S, B)
        return dS_dt / self.v # Normalize by velocity since dy = v * dt
        

    def solve_spins(self, S0):        
        """Solves the spin evolution for a all neutron paths and update a progress bar
        saves the spin evolution and adiabaticities within the UCNpath instance

        Args:
            S0 (np.array[float]): initial spin vector

        """
        for path in self.UCNpaths:
            arc_lengths = path.arc_lengths
            # setup and solve spins iteratively
            S = np.zeros((3, len(arc_lengths)))
            S[:, 0] = S0

            # progress bar for solution
            pbar = tqdm(total=len(arc_lengths), desc="Solving")

            for i in range(1, len(arc_lengths)):
                ds = arc_lengths[i] - arc_lengths[i - 1]
                s_prev = arc_lengths[i - 1]
                S_prev = S[:, i - 1]

                # get necessary Bfields
                B_prev = path.Bfield_on_path[i - 1]
                B_current = path.Bfield_on_path[i]
                B_inter = (B_prev + B_current) / 2

                # RK4 step
                k1 = self.bloch_eq(B_prev, S_prev)
                k2 = self.bloch_eq(B_inter, S_prev + ds/2 * k1)
                k3 = self.bloch_eq(B_inter, S_prev + ds/2 * k2)
                k4 = self.bloch_eq(B_current, S_prev + ds * k3)

                pbar.update(1)

                # RK4 evaluation by averaging slopes
                S_next = S_prev + ds * (k1 + 2*k2 + 2*k3 + k4) / 6
                S[:, i] = S_next / np.linalg.norm(S_next)

            path.save_spins(np.array(S))
    
    
    def plot_spin_set(self, pdf_name="spin_plots.pdf"):
        """Plots a graph of magnetic field experienced along the path
            and spin evolution as a function of path y component
            for a collection of paths and compiles them into a pdf

        Args:
            path (np.array[np.array[np.array[float]]]): paths of simulated neutron [path_x, path_y, path_z]
            spin (np.array[np.array[np.array[float]]]): path evolutions of spin vectors [spin_x, spin_y, spin_z]
            collisions_set (np.array[np.array[np.array[float]]]) : collision locations along the paths
            thetas (np.array[float]): thetas corresponding to the path

        """

        with PdfPages(pdf_name) as pdf:
            i = 0
            for UCNpath in self.UCNpaths:
                i += 1
                fig, ax = plt.subplots(3, 1, figsize=(12, 12))

                path = UCNpath.path
                spin = UCNpath.spins
                collisions = UCNpath.collisions
                path_field = UCNpath.Bfield_on_path.T
                adiabaticities = UCNpath.adiabaticities

                ax[0].plot(path[1], path_field[0] * 1e6, label="Bx", color="blue")
                ax[0].plot(path[1], path_field[1] * 1e6, label="By", color="orange")
                ax[0].plot(path[1], path_field[2] * 1e6, label="Bz", color="green")
                ax[0].vlines(collisions[:,1], 0, 400, label="Collisions", colors='yellow', linestyles='dotted')
                ax[0].legend()
                ax[0].grid(True)
                ax[0].set_ylabel("Magnetic Field (μT)", fontsize=20)
                ax[0].set_title(f"Path index: {i} / {len(self.UCNpaths)} | θ = {UCNpath.theta:.2f} degrees", fontsize=22)

                ax[1].plot(path[1], spin[0], label="Sx", color="red")
                ax[1].plot(path[1], spin[1], label="Sy", color="blue")
                ax[1].plot(path[1], spin[2], label="Sz", color="green")
                ax[1].vlines(collisions[:,1], -1, 1, label="Collisions", colors='yellow', linestyles='dotted')
                ax[1].legend()
                ax[1].grid(True)
                ax[1].set_ylabel("Spin Components", fontsize=20)

                ax[2].plot(path[1], adiabaticities, label="Adiabaticity", color="purple")
                ax[2].set_yscale("log")
                ax[2].set_xlabel("Axial Position of the Path: y (m)", fontsize=20)
                ax[2].set_ylabel("Adiabaticity κ", fontsize=20)
                ax[2].legend()
                ax[2].grid(True)

                fig.suptitle(f"UCN Spin Evolution (Path {i})", fontsize=16)
                pdf.savefig(fig)
                plt.close(fig)
                