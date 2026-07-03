# Capturing a QM record/replay fixture

**Purpose.** Turn ONE real TeraChem/Molcas AIMS run into a permanent, hermetic
regression fixture. Only the *capture* needs the QM program; the resulting tape
replays anywhere (see `pyspawn/es_record_replay.py`). This closes the validation
gap the cone oracle cannot reach (`current_architecture.md` §4e: no `S_elec` /
DGAS / QM continuation state).

**Key idea — capture once, replay forever.** The capture produces two small files
(`qm_fixture.tape`, `qm_golden.hdf5`). Copy them back to any machine (no TeraChem
needed) and the `ReplayBackend` reproduces the full trajectory hermetically, so
every QM-touching refactor gets a `max|diff|=0` gate. You do **not** need to move
your whole workflow to the TeraChem box — just run the capture there.

---

## Prerequisites on the TeraChem machine

- Python 3 with `numpy`, `h5py` (and `matplotlib` for analysis).
- `tcpb` (TeraChem Protocol Buffers client) — `pyspawn/potential/terachem_cas.py`
  imports `TCProtobufClient`.
- A running TeraChem TCPB server on the port the run uses (`tc_port`).

## Step 1 — get the code (git, not tar)

If the machine can reach GitHub:

```bash
git clone git@github.com:mtzgroup/py3Spawn.git       # or: cd existing && git fetch
cd py3Spawn && git checkout python3 && git pull
```

If the cluster is air-gapped (no outbound git), make a single-file bundle here and
`scp` that (preserves full history, excludes `__pycache__`/scratch — cleaner and
verifiable, unlike a raw tar):

```bash
# on this machine:
git bundle create py3spawn.bundle python3
# scp py3spawn.bundle to the cluster, then there:
git clone -b python3 py3spawn.bundle py3Spawn
```

## Step 2 — confirm the port is healthy (hermetic, no TeraChem)

Before touching QM, make sure the Py3 package itself runs on the new machine:

```bash
cd py3Spawn && PYTHONPATH=$PWD python tests/pr0_oracle.py     # expect OVERALL: PASS
```

This is the analytic cone — no TeraChem — so a PASS confirms numpy/h5py and the
port are intact on that machine.

## Step 3 — capture the fixture

Take a **short** existing QM example (e.g. `examples/ethylene_fomocasci/`) so the
tape is small. After the usual wiring (`into_traj(pyspawn.potential.terachem_cas)`
+ `into_traj(...vv)`), wrap the already-wired backend and run as normal:

```python
from pyspawn.es_record_replay import RecordingBackend, save_tape
from pyspawn.traj import traj

tape = {}
# wrap whatever into_traj wired (LegacyMutatingBackend around terachem_cas):
traj._es_backend = RecordingBackend(traj._es_backend, tape)

# ... build the simulation and run exactly as the example does ...
sim.propagate()

save_tape(tape, "qm_fixture.tape")     # the recorded ES outputs
# and keep the trajectory as golden:
import shutil; shutil.copy2("sim.hdf5", "qm_golden.hdf5")
```

`RecordingBackend` is transparent — the run is bit-identical to an un-wrapped run
(proven on the cone by `tests/test_es_replay.py`), so `qm_golden.hdf5` is a real
reference trajectory. Keep the run to a few a.u. and, if the physics spawns,
that's fine — centroids and back-propagation are recorded too.

## Step 4 — replay anywhere (the QM oracle)

Copy `qm_fixture.tape` + `qm_golden.hdf5` back to any machine. Re-run the **same**
simulation setup, but drive ES from the tape instead of TeraChem, and diff:

```python
from pyspawn.es_record_replay import ReplayBackend, load_tape, max_hdf5_diff
from pyspawn.traj import traj

traj._es_backend = ReplayBackend(load_tape("qm_fixture.tape"))
# ... identical simulation setup + sim.propagate(), writing sim.hdf5 ...

n, d, worst, mism = max_hdf5_diff("qm_golden.hdf5", "sim.hdf5")
assert d == 0 and mism == 0, (d, worst, mism)
print("QM replay oracle: %d datasets, max|diff|=%g -> PASS" % (n, d))
```

The replay setup must reproduce the recorded run's initial conditions (geometry,
states, timestep, tfinal, seed). Easiest: reuse the capture's start script with
only the backend line swapped. Wrap this as `tests/qm_replay_oracle.py` once the
fixture exists, and it becomes a standing regression gate for the QM path.

---

## Caveats

- **Determinism.** Replay reproduces a run only if the run was deterministic.
  Capture a fixed-geometry start (no runtime Wigner sampling; no SSAIMS stochastic
  selection — or a fixed `ss_seed`). Runtime randomness is centralized in PR2;
  until then, pick a deterministic QM example.
- **Geometry key.** The tape is keyed by `(label, direction, round(geometry,10))`
  (`es_record_replay.py`). Valid unless a `(label, direction)` chain revisits the
  exact rounded geometry with a *different* ES output — not expected for a
  continuous trajectory (see `es_seam_contract.md` §4.1).
- **What's recorded.** Only the ES *outputs* the dynamics consumes (`energies`,
  `forces`, `timederivcoups`, `S_elec_flat`, `wf`). The QM continuation state
  (orbs/civecs/phases) is *not* recorded because replay never recomputes ES.
- **Bring back the small files, not the machine.** The tape + golden are small;
  the hermetic replay/iteration happens off the TeraChem box.
