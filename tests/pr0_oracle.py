#!/usr/bin/env python
"""PR0 hermetic regression oracle for py3Spawn.

Runs entirely on the built-in analytic conical-intersection model
(pyspawn.potential.test_cone) -- NO TeraChem/Molcas required -- and asserts the
two properties every later refactor PR must preserve:

  1. PROPAGATION DETERMINISM: two independent seeded runs to t=tfinal produce
     bit-identical trajectories.
  2. RESTART EQUIVALENCE: a run interrupted at t=thalf and resumed to t=tfinal
     is bit-identical to an uninterrupted run to t=tfinal. This is the
     durability property the whole architecture rewrite must not regress.

Usage:
    PYTHONPATH=<repo-root> python tests/pr0_oracle.py
Exits 0 on PASS, 1 on FAIL. Writes scratch runs under a temp dir it cleans up.
"""
import os
import sys
import shutil
import tempfile
import contextlib
import numpy as np
import h5py
import pyspawn


@contextlib.contextmanager
def _silence():
    """pyspawn's task-eval driver prints hundreds of lines per step. Mute
    stdout during propagation so the oracle emits only its verdict.
    (Set PR0_ORACLE_VERBOSE=1 to see the full pyspawn log.)"""
    if os.environ.get("PR0_ORACLE_VERBOSE"):
        yield
        return
    with open(os.devnull, "w") as devnull:
        old = sys.stdout
        sys.stdout = devnull
        try:
            yield
        finally:
            sys.stdout = old

SEED = 20260702
TIMESTEP = 0.02
THALF = 2.0
TFINAL = 4.0
NDIMS = 2
NSTATES = 2
ISTATE = 1


def _wire_backend():
    """(Re)attach the hermetic cone backend + integrators to the classes."""
    pyspawn.import_methods.into_simulation(pyspawn.qm_integrator.rk2)
    pyspawn.import_methods.into_simulation(pyspawn.qm_hamiltonian.adiabatic)
    pyspawn.import_methods.into_traj(pyspawn.potential.test_cone)
    pyspawn.import_methods.into_traj(pyspawn.classical_integrator.vv)


def _initial_conditions():
    np.random.seed(SEED)
    pos = np.random.normal(0.0, 1.0, NDIMS)
    mom = np.random.normal(0.0, 0.1, NDIMS)
    return pos, mom


def _fresh_sim(maxtime):
    _wire_backend()
    pos, mom = _initial_conditions()
    wid = np.ones(NDIMS)
    m = np.ones(NDIMS)
    tr = pyspawn.traj(NDIMS, NSTATES)
    tr.init_traj(0.0, NDIMS, pos, mom, wid, m, NSTATES, ISTATE, "00")
    tr.set_spawnthresh(1.0)
    sim = pyspawn.simulation()
    sim.add_traj(tr)
    sim.set_timestep_all(TIMESTEP)
    sim.set_mintime_all(0.0)
    sim.set_maxtime_all(maxtime)
    sim.init_amplitudes_one()
    return sim


def _run_full(rundir, maxtime=TFINAL):
    os.makedirs(rundir, exist_ok=True)
    cwd = os.getcwd()
    os.chdir(rundir)
    try:
        with _silence():
            _fresh_sim(maxtime).propagate()
    finally:
        os.chdir(cwd)
    return os.path.join(rundir, "sim.hdf5")


def _run_with_restart(rundir):
    os.makedirs(rundir, exist_ok=True)
    cwd = os.getcwd()
    os.chdir(rundir)
    try:
        with _silence():
            _fresh_sim(THALF).propagate()       # phase 1: to t=THALF
            _wire_backend()                     # backend not serialized
            sim = pyspawn.simulation()
            sim.restart_from_file("sim.json", "sim.hdf5")
            sim.set_maxtime_all(TFINAL)         # phase 2: resume to t=TFINAL
            sim.propagate()
    finally:
        os.chdir(cwd)
    return os.path.join(rundir, "sim.hdf5")


def _load(path):
    d = {}
    with h5py.File(path, "r") as f:
        f.visititems(
            lambda n, o: d.__setitem__(n, o[()])
            if isinstance(o, h5py.Dataset) else None)
    return d


def _max_diff(p1, p2):
    a, b = _load(p1), _load(p2)
    keys = sorted(set(a) & set(b))
    maxd, worst, mism = 0.0, None, 0
    for k in keys:
        x, y = np.asarray(a[k]), np.asarray(b[k])
        if x.shape != y.shape:
            mism += 1
            continue
        if np.issubdtype(x.dtype, np.number) and x.size:
            d = np.nanmax(np.abs(x - y))
            if d > maxd:
                maxd, worst = d, k
    return len(keys), maxd, worst, mism


def main():
    tmp = tempfile.mkdtemp(prefix="pr0_oracle_")
    try:
        ref = _run_full(os.path.join(tmp, "ref"))
        rep = _run_full(os.path.join(tmp, "rep"))
        rst = _run_with_restart(os.path.join(tmp, "restart"))

        n1, d1, w1, m1 = _max_diff(ref, rep)
        n2, d2, w2, m2 = _max_diff(ref, rst)

        tol = 1e-12
        ok1 = (d1 <= tol and m1 == 0)
        ok2 = (d2 <= tol and m2 == 0)

        print("PR0 hermetic oracle (pyspawn.potential.test_cone)")
        print("  [1] propagation determinism: %d datasets, max|diff|=%g, "
              "mismatches=%d -> %s" % (n1, d1, m1, "PASS" if ok1 else "FAIL"))
        print("  [2] restart equivalence:     %d datasets, max|diff|=%g, "
              "mismatches=%d -> %s" % (n2, d2, m2, "PASS" if ok2 else "FAIL"))
        if not (ok1 and ok2):
            print("OVERALL: FAIL")
            return 1
        print("OVERALL: PASS")
        return 0
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    sys.exit(main())
