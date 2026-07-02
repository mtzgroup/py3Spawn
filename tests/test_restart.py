import numpy as np
import pyspawn

# Restart test for the hermetic cone model. Run AFTER a start script (e.g.
# test_prop_seeded.py) has left sim.json + sim.hdf5 in the cwd. The backend
# methods must be re-imported before restarting (they are not serialized), and
# restart uses restart_from_file(json, h5) -- not the old read_from_file(json).
pyspawn.import_methods.into_simulation(pyspawn.qm_integrator.rk2)
pyspawn.import_methods.into_simulation(pyspawn.qm_hamiltonian.adiabatic)
pyspawn.import_methods.into_traj(pyspawn.potential.test_cone)
pyspawn.import_methods.into_traj(pyspawn.classical_integrator.vv)

tfinal = 6.0

sim = pyspawn.simulation()

sim.restart_from_file("sim.json", "sim.hdf5")

sim.set_maxtime_all(tfinal)

sim.propagate()







