#!/usr/bin/env python
"""Hermetic self-validation of the ES record/replay harness, on the cone.

Proves the machinery in pyspawn/es_record_replay.py works BEFORE it is used on a
real (un-runnable-here) QM trajectory:

  1. RECORDING IS TRANSPARENT -- a run wrapped in RecordingBackend reproduces the
     un-wrapped cone trajectory bit-for-bit (max|diff|=0).
  2. REPLAY REPRODUCES        -- a run driven by ReplayBackend (no cone math at
     all, just the recorded ES outputs) reproduces the same trajectory (max|diff|=0).

If both hold, then when a real TeraChem tape is captured
(design/qm_fixture_capture.md), the same ReplayBackend gives a hermetic QM oracle.

Run: PYTHONPATH=<repo> python tests/test_es_replay.py   (exit 0 = PASS)
"""
import os
import sys
import shutil
import tempfile
import contextlib

import numpy as np
import pyspawn
from pyspawn.traj import traj
from pyspawn.potential.test_cone import TestConeBackend
from pyspawn.es_record_replay import (RecordingBackend, ReplayBackend,
                                      max_hdf5_diff)

SEED = 20260702
TIMESTEP = 0.02
TFINAL = 4.0
NDIMS = 2
NSTATES = 2
ISTATE = 1


@contextlib.contextmanager
def _silence():
    if os.environ.get("ES_REPLAY_VERBOSE"):
        yield
        return
    with open(os.devnull, "w") as devnull:
        old = sys.stdout
        sys.stdout = devnull
        try:
            yield
        finally:
            sys.stdout = old


def _wire():
    pyspawn.import_methods.into_simulation(pyspawn.qm_integrator.rk2)
    pyspawn.import_methods.into_simulation(pyspawn.qm_hamiltonian.adiabatic)
    pyspawn.import_methods.into_traj(pyspawn.potential.test_cone)
    pyspawn.import_methods.into_traj(pyspawn.classical_integrator.vv)


def _run(rundir, backend=None):
    """Run the cone trajectory; if `backend` is given, override the wired ES
    backend with it (the record/replay wrappers)."""
    os.makedirs(rundir, exist_ok=True)
    cwd = os.getcwd()
    os.chdir(rundir)
    try:
        _wire()
        if backend is not None:
            traj._es_backend = backend      # class attr; shared by all instances
        np.random.seed(SEED)
        pos = np.random.normal(0.0, 1.0, NDIMS)
        mom = np.random.normal(0.0, 0.1, NDIMS)
        tr = pyspawn.traj(NDIMS, NSTATES)
        tr.init_traj(0.0, NDIMS, pos, mom, np.ones(NDIMS), np.ones(NDIMS),
                     NSTATES, ISTATE, "00")
        tr.set_spawnthresh(1.0)
        sim = pyspawn.simulation()
        sim.add_traj(tr)
        sim.set_timestep_all(TIMESTEP)
        sim.set_mintime_all(0.0)
        sim.set_maxtime_all(TFINAL)
        sim.init_amplitudes_one()
        with _silence():
            sim.propagate()
    finally:
        os.chdir(cwd)
    return os.path.join(rundir, "sim.hdf5")


def main():
    tmp = tempfile.mkdtemp(prefix="es_replay_")
    try:
        golden = _run(os.path.join(tmp, "golden"))                 # un-wrapped
        tape = {}
        rec = _run(os.path.join(tmp, "rec"),
                   RecordingBackend(TestConeBackend(), tape))       # record
        rep = _run(os.path.join(tmp, "rep"), ReplayBackend(tape))   # replay

        n1, d1, _, m1 = max_hdf5_diff(golden, rec)
        n2, d2, _, m2 = max_hdf5_diff(golden, rep)
        tol = 1e-12
        ok1 = (d1 <= tol and m1 == 0)
        ok2 = (d2 <= tol and m2 == 0)

        print("ES record/replay self-validation (cone)")
        print("  [1] recording transparent: %d datasets, max|diff|=%g, "
              "mism=%d -> %s" % (n1, d1, m1, "PASS" if ok1 else "FAIL"))
        print("  [2] replay reproduces:      %d datasets, max|diff|=%g, "
              "mism=%d -> %s" % (n2, d2, m2, "PASS" if ok2 else "FAIL"))
        print("  tape entries: %d" % len(tape))
        if not (ok1 and ok2):
            print("OVERALL: FAIL")
            return 1
        print("OVERALL: PASS")
        return 0
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    sys.exit(main())
