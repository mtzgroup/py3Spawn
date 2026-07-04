#!/usr/bin/env python
"""QM hermetic replay oracle for py3Spawn.

The analytic-cone oracle (tests/pr0_oracle.py) is structurally blind to the QM
path -- S_elec / DGAS / CI-vector + orbital continuation state that only the real
QM backends produce. This oracle closes that gap WITHOUT needing TeraChem: it
replays a captured fixture (one real ethylene SA2-CASSCF AIMS run) through
ReplayBackend and asserts the whole trajectory -- including the spawn, centroid,
and back-propagation channels -- reproduces the golden HDF5 at max|diff|==0.

Hermetic: needs only numpy + h5py + the committed fixture. terachem_cas imports
tcpb under try/except, and ReplayBackend never touches the TeraChem code path, so
neither tcpb nor a TeraChem server is required to run this.

The fixture was captured by examples/ethylene_fomocasci/capture_c2h4.py on a
TeraChem machine; see design/qm_fixture_capture.md. To refresh it, re-run that
capture and copy qm_fixture.tape / qm_golden.hdf5 / hessian.hdf5 into
tests/fixtures/ under the names below.

Usage:
    PYTHONPATH=<repo-root> python tests/qm_replay_oracle.py
Exits 0 on PASS, 1 on FAIL. Runs in a temp dir it cleans up.
"""
import os
import sys
import shutil
import tempfile
import contextlib

import numpy as np
import h5py

import pyspawn
import pyspawn.general
from pyspawn.es_record_replay import ReplayBackend, load_tape, max_hdf5_diff
from pyspawn.traj import traj as traj_cls

_FIX = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fixtures")
TAPE = os.path.join(_FIX, "qm_ethylene_fomocasci.tape")
GOLDEN = os.path.join(_FIX, "qm_ethylene_fomocasci_golden.hdf5")
HESSIAN = os.path.join(_FIX, "qm_ethylene_fomocasci_hessian.hdf5")

# These MUST match examples/ethylene_fomocasci/capture_c2h4.py (the run that
# produced the fixture); any drift changes the geometry sequence and misses the
# tape keys.
SEED = 87061
TS = 10.0
TFINAL = 500.0
NUMDIMS = 18
NUMSTATES = 2

TC_OPTIONS = {
    "method": 'hf', "basis": '6-31g',
    "atoms": ["C", "C", "H", "H", "H", "H"],
    "charge": 0, "spinmult": 1, "closed_shell": True, "restricted": True,
    "precision": "double", "threall": 1.0e-20,
    "casci": "yes", "fon": "yes", "closed": 7, "active": 2,
    "cassinglets": 2, "castargetmult": 1,
    "cas_energy_states": [0, 1], "cas_energy_mults": [1, 1],
}


@contextlib.contextmanager
def _silence():
    """pyspawn's task-eval driver prints hundreds of lines per step; mute it
    unless QM_ORACLE_VERBOSE=1."""
    if os.environ.get("QM_ORACLE_VERBOSE"):
        yield
        return
    with open(os.devnull, "w") as devnull:
        old = sys.stdout
        sys.stdout = devnull
        try:
            yield
        finally:
            sys.stdout = old


def _build_sim():
    pyspawn.import_methods.into_simulation(pyspawn.qm_integrator.rk2)
    pyspawn.import_methods.into_simulation(pyspawn.qm_hamiltonian.adiabatic)
    pyspawn.import_methods.into_traj(pyspawn.potential.terachem_cas)
    pyspawn.import_methods.into_traj(pyspawn.classical_integrator.vv)
    # Drive ES from the tape -- no TeraChem contacted.
    traj_cls._es_backend = ReplayBackend(load_tape(TAPE))

    traj_params = {
        "tc_port": 0,
        "time": 0.0, "timestep": TS, "maxtime": TFINAL,
        "spawnthresh": (0.5 * np.pi) / TS / 20.0,
        "istate": 1,
        "widths": np.asarray([30.0, 30.0, 30.0, 30.0, 30.0, 30.0,
                              6.0, 6.0, 6.0, 6.0, 6.0, 6.0,
                              6.0, 6.0, 6.0, 6.0, 6.0, 6.0]),
        "atoms": TC_OPTIONS["atoms"],
        "masses": np.asarray([21864.0, 21864.0, 21864.0, 21864.0, 21864.0, 21864.0,
                              1822.0, 1822.0, 1822.0, 1822.0, 1822.0, 1822.0,
                              1822.0, 1822.0, 1822.0, 1822.0, 1822.0, 1822.0]),
        "tc_options": TC_OPTIONS,
    }
    sim_params = {
        "quantum_time": 0.0, "timestep": TS, "max_quantum_time": TFINAL,
        "qm_amplitudes": np.ones(1, dtype=np.complex128),
        "qm_energy_shift": 77.6,
    }

    pyspawn.general.check_files()
    tr = pyspawn.traj(NUMDIMS, NUMSTATES)
    tr.set_numstates(NUMSTATES)
    tr.set_numdims(NUMDIMS)
    tr.set_parameters(traj_params)
    tr.initial_wigner(SEED)          # reads hessian.hdf5 in CWD

    sim = pyspawn.simulation()
    sim.add_traj(tr)
    sim.set_parameters(sim_params)
    return sim


def main():
    for f in (TAPE, GOLDEN, HESSIAN):
        if not os.path.exists(f):
            print("QM replay oracle: MISSING FIXTURE %s" % f)
            print("  (capture it on a TeraChem machine; see "
                  "design/qm_fixture_capture.md)")
            return 1

    tmp = tempfile.mkdtemp(prefix="qm_oracle_")
    cwd = os.getcwd()
    try:
        shutil.copy2(HESSIAN, os.path.join(tmp, "hessian.hdf5"))
        os.chdir(tmp)
        with _silence():
            _build_sim().propagate()
        n, d, worst, mism = max_hdf5_diff(GOLDEN, os.path.join(tmp, "sim.hdf5"))
    finally:
        os.chdir(cwd)
        shutil.rmtree(tmp, ignore_errors=True)

    ok = (d == 0 and mism == 0)
    print("QM hermetic replay oracle (ethylene SA2-CASSCF, replayed fixture)")
    print("  replay reproduces golden: %d datasets, max|diff|=%g, "
          "shape-mismatch=%d, worst=%s -> %s"
          % (n, d, mism, worst, "PASS" if ok else "FAIL"))
    print("OVERALL: %s" % ("PASS" if ok else "FAIL"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
