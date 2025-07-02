# UCN-Rotation-Simulations
Simulations for the spin vector evolution of random ultracold neutron (UCN) paths under a given magnetic field's influence. The class UCNspinRotSim is used for the main simulation.

## UCNspinRotSim
This python class generates neutron paths and solves the spin evolution along that path for a specified magnetic vector field and UCN guide geometry. An example of how to use the class is demonstrated in testFieldFullSim.py

The Python libraries that must be installed for this class to work are
- NumPy
- SciPy
- MatplotLib
- tqdm

Things to note about how the simulation functions:
- The discrete vector field passed to the class upon initialization is linearly interpolated as a grid to produce a continuous field for the simulation. This means a magnetic field data set that does not resemble a grid shape cannot be used with this code directly
- Calling simulate_path generates a new UCN and calculates its path at the same time
- Calling solve_spins solves the Bloch Equation along the inputted path to determine the spin evolution of the UCN. The IVP is solved using the RK4 method and the upsample factor determines how many samples the path will be given that lie between two axial layers of the magnetic field data set.

