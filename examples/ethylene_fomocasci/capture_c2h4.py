# Capture a hermetic QM record/replay fixture from a real TeraChem run.
#
# This is start_c2h4.py with three additions (marked ### CAPTURE ###):
#   1. tfinal shortened to keep the tape small (tune CAPTURE_TFINAL below);
#   2. the wired ES backend wrapped in RecordingBackend BEFORE propagate();
#   3. the tape + golden trajectory saved AFTER propagate().
#
# Run this on the TeraChem machine (TCPB server up on `port`). Then copy
# qm_fixture.tape + qm_golden.hdf5 back to any machine; ReplayBackend reproduces
# the trajectory with no TeraChem. See design/qm_fixture_capture.md.
#
# Determinism note: initial_wigner(seed) does np.random.seed(seed) right before
# its draws, and this is plain AIMS (the SSAIMS bare-RNG in simulation.py never
# fires), so this start is reproducible -- a valid golden reference.
import shutil

import numpy as np
import pyspawn
import pyspawn.general
from pyspawn.es_record_replay import RecordingBackend, save_tape
from pyspawn.traj import traj as traj_cls

# terachemserver port
port = 54322

# random number seed
seed = 87061

# Velocity Verlet classical propagator
clas_prop = "vv"

# adapative 2nd-order Runge-Kutta quantum propagator
qm_prop = "rk2"

# adiabtic NPI quantum Hamiltonian
qm_ham = "adiabatic"

# use TeraChem CASSCF or CASCI to compute potentials
potential = "terachem_cas"

# initial time
t0 = 0.0

# time step
ts = 10.0

### CAPTURE ### short final time so the tape stays small. 500 au = ~50 steps
# is enough to validate the pipeline; raise it (e.g. to a value that lets the
# physics spawn once) for a fixture that also covers centroids/back-prop.
CAPTURE_TFINAL = 500.0
tfinal = CAPTURE_TFINAL

# number of dimensions
numdims = 18

# number of electronic states
numstates = 2

# TeraChem job options
tc_options = {
    "method":       'hf',
    "basis":        '6-31g',
    "atoms":        ["C", "C", "H", "H", "H", "H"],
    "charge":       0,
    "spinmult":     1,
    "closed_shell": True,
    "restricted":   True,

    "precision":    "double",
    "threall":      1.0e-20,

    "casci":        "yes",
    "fon":          "yes",
    "closed":       7,
    "active":       2,
    "cassinglets":  2,
    "castargetmult": 1,
    "cas_energy_states": [0, 1],
    "cas_energy_mults": [1, 1],
    }

# trajectory parameters
traj_params = {
    "tc_port": port,
    "time": t0,
    "timestep": ts,
    "maxtime": tfinal,
    "spawnthresh": (0.5 * np.pi) / ts / 20.0,
    "istate": 1,
    "widths": np.asarray([30.0, 30.0, 30.0,
                        30.0, 30.0, 30.0,
                        6.0, 6.0, 6.0,
                        6.0, 6.0, 6.0,
                        6.0, 6.0, 6.0,
                        6.0, 6.0, 6.0]),
    "atoms": tc_options["atoms"],
    "masses": np.asarray([21864.0, 21864.0, 21864.0,
                    21864.0, 21864.0, 21864.0,
                    1822.0, 1822.0, 1822.0,
                    1822.0, 1822.0, 1822.0,
                    1822.0, 1822.0, 1822.0,
                    1822.0, 1822.0, 1822.0]),
    "tc_options": tc_options
    }

sim_params = {
    "quantum_time": traj_params["time"],
    "timestep": traj_params["timestep"],
    "max_quantum_time": traj_params["maxtime"],
    "qm_amplitudes": np.ones(1, dtype=np.complex128),
    "qm_energy_shift": 77.6,
}

# import routines needed for propagation
exec("pyspawn.import_methods.into_simulation(pyspawn.qm_integrator." + qm_prop + ")")
exec("pyspawn.import_methods.into_simulation(pyspawn.qm_hamiltonian." + qm_ham + ")")
exec("pyspawn.import_methods.into_traj(pyspawn.potential." + potential + ")")
exec("pyspawn.import_methods.into_traj(pyspawn.classical_integrator." + clas_prop + ")")

### CAPTURE ### wrap the wired backend (a class attribute set by into_traj) so
# every ES call is recorded. Transparent: the recorded run is bit-identical to
# an un-wrapped run (proven on the cone by tests/test_es_replay.py).
tape = {}
traj_cls._es_backend = RecordingBackend(traj_cls._es_backend, tape)

# check for the existence of files from a past run
pyspawn.general.check_files()

# set up first trajectory
traj1 = pyspawn.traj(numdims, numstates)
traj1.set_numstates(numstates)
traj1.set_numdims(numdims)
traj1.set_parameters(traj_params)

# sample initial position and momentum from Wigner distribution (requires hessian.hdf5)
traj1.initial_wigner(seed)

# set up simulation
sim = pyspawn.simulation()
sim.add_traj(traj1)
sim.set_parameters(sim_params)

# begin propagation
sim.propagate()

### CAPTURE ### persist the fixture (tape) and the golden trajectory.
save_tape(tape, "qm_fixture.tape")
shutil.copy2("sim.hdf5", "qm_golden.hdf5")
print("### CAPTURE ### wrote qm_fixture.tape (%d entries) + qm_golden.hdf5"
      % len(tape))
