# PR0 hermetic oracle: deterministic propagation on the built-in analytic
# conical-intersection model (pyspawn.potential.test_cone). No external QM code.
#
# Differs from the legacy test_prop.py in two ways, both required to make it a
# usable golden-trajectory oracle under Python 3:
#   1. Fixed RNG seed  -> initial pos/mom are reproducible run-to-run.
#   2. Modern traj API -> pyspawn.traj(numdims, numstates) (the ctor now takes
#      dims/states; the old no-arg form was pre-Py3-migration).
#
# Run from a scratch dir (it writes sim.hdf5 / sim.json / working.hdf5):
#   PYTHONPATH=<repo> python test_prop_seeded.py
import numpy as np
import pyspawn

# --- deterministic initial conditions -----------------------------------
SEED = 20260702
np.random.seed(SEED)

# --- wire the hermetic (no-QM) backend + integrators --------------------
pyspawn.import_methods.into_simulation(pyspawn.qm_integrator.rk2)
pyspawn.import_methods.into_simulation(pyspawn.qm_hamiltonian.adiabatic)
pyspawn.import_methods.into_traj(pyspawn.potential.test_cone)
pyspawn.import_methods.into_traj(pyspawn.classical_integrator.vv)

# --- simulation parameters ----------------------------------------------
t0 = 0.0
timestep = 0.02
tfinal = 4.0
ndims = 2
nstates = 2
istate = 1

pos = np.random.normal(0.0, 1.0, ndims)
mom = np.random.normal(0.0, 0.1, ndims)
wid = np.ones(ndims)
m = np.ones(ndims)

# --- build trajectory + simulation (modern API) -------------------------
traj1 = pyspawn.traj(ndims, nstates)
traj1.init_traj(t0, ndims, pos, mom, wid, m, nstates, istate, "00")
traj1.set_spawnthresh(1.0)

sim = pyspawn.simulation()
sim.add_traj(traj1)
sim.set_timestep_all(timestep)
sim.set_mintime_all(t0)
sim.set_maxtime_all(tfinal)
sim.init_amplitudes_one()

sim.propagate()
