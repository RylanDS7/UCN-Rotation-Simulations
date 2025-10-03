# UCN Rotation Simulations
Simulations for the spin vector evolution of random ultracold neutron (UCN) paths under a given magnetic field's influence. The class UCNspinRotSim is used for the main simulation which is located in the UCNspinSim folder.

## UCN Spin Simulation
This set of scipts includes the UCNpath class which generates neutron paths and the UCNspinRotSim class which solves the spin evolution for a given UCN path and a specified magnetic vector field. Initializing an instance of UCNspinRotSim will automatically construct a list of UCNpath instances and save them as a class attribute. An example of how to use the UCNspinRotSim class is demonstrated in testFieldFullSim.py

The Python libraries that must be installed for the UCNpath and UCNspinRotSim classes to work are
- NumPy
- SciPy
- MatplotLib
- tqdm

Things to note about how the simulation functions:
- The discrete vector field passed to the class upon initialization is linearly interpolated as a grid to produce a continuous field for the simulation. This means a magnetic field data set that does not resemble a grid shape cannot be used with this code directly
- Calling UCNpath.simulate_path generates a new UCN and calculates its path at the same time
- Calling solve_spins solves the Bloch Equation along the inputted path to determine the spin evolution of the UCN. The IVP is solved using the RK4 method and the upsample factor determines how many samples the path will be given that lie between two axial layers of the magnetic field data set.

## Tapering Field Tuning

The TaperingFieldTuning folder contains scipts that were used to optimize the function which is used to generate the spin tapering field. These scripts optimize the function under different assumptions either analytically or using gradient descent to maximize the lowest adiabaticity value reached by the function.

## Coil Pathing

The CoilPathing folder contains scripts to clean and generate contours which can be used as the wire paths for magnetic coils that can produce some field.

- contourCleaner.py converts a contour exported from COMSOL into a list of ordered points

- TXTtoSketch.bas can be run as a SolidWorks macro to take this list of ordered points and turn it into a 3D sketch of the contour

