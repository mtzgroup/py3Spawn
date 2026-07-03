# CLAUDE.md — PySpawn refactor project

## What this project is

Restructuring **PySpawn** (ab initio multiple spawning / AIMS, plus SSAIMS and
surface-hopping variants) so that it is maintainable by people other than its
original author. The physics is correct and validated; the problem is purely
architectural.

**This file lives at the root of the `py3Spawn` repo** (the fork of PySpawn).
Remotes: **`origin` = `git@github.com:mtzgroup/py3Spawn.git`** (our fork, where
all work is pushed) and **`upstream` = `git@github.com:blevine37/pySpawn17.git`**
(Ben Levine's canonical repo — pull from it, open PRs back to it, never push
directly). Ben is currently upstream's only maintainer. **Work happens on the
`python3` branch**, which is `master` + 2 commits and already includes SSAIMS —
it is the superset branch and the correct base. The `pyspawn/` package is the
**source to refactor**.

- `design/nextfms_skeleton/` — a small, runnable **target-state skeleton** (toy
  physics) showing the architecture to refactor *toward*. Read its `README.md`
  first; it is the blueprint. Do not confuse it with production code — its
  physics is a stub. (This is project scaffolding, not part of PySpawn proper —
  exclude it from any PR opened back to `upstream`.)

**The goal, in one sentence:** make the electronic-structure/driver control flow
read *serially* from the top, with all the concurrency-and-resilience machinery
sealed at the bottom behind one narrow seam — so a student can edit the physics
without being able to break the resilience.

## The core diagnosis (read before touching anything)

PySpawn achieves crash-resilience by **inversion of control**: it builds a queue
of task *strings*, sorts them by time, and `eval()`s them one per loop cycle,
reconstructing the dynamics from externalized state. This is why it is hard to
reason about — the linear narrative of an AIMS step is shredded across a
scheduler.

Concretely, in `pyspawn/simulation.py`:

- `propagate()` (**line ~297**) is the main loop.
- `update_queue()` (**line ~619**) builds task strings like
  `self.traj["<key>"].propagate_step()` and
  `self.centroids["<key>"].compute_centroid()`.
- The loop pops one and runs it via **`eval(current)` (line ~336)**. This is the
  inversion, in its most literal form.

The resilience this buys is **deterministic-replay-style durability**: state is
serialized to `sim.json` + `sim.hdf5` every cycle (`restart_output()`,
`restart_from_file()`), so a killed run resumes. That durability is the feature
worth keeping. The `eval`-string task queue is the cost worth removing.

**The target architecture keeps the durability and removes the inversion** by
sealing all scheduling/concurrency behind a single `executor.evaluate(requests)`
call that *looks synchronous* to the physics. See `design/nextfms_skeleton/` for the
working demonstration (including a crash+restart that reproduces an uninterrupted
trajectory byte-for-byte).

## Hard constraints — do not violate

1. **The `python3` branch is already ported to Python 3.** Verified: every file
   in `pyspawn/` parses under Python 3, and there are **no live Python-2
   statements** anywhere (the only `print "..."` / `iteritems` matches are inside
   commented-out debug lines). `master` is still Python 2 (~158 live `print`
   statements) — do NOT base work on it. The port is done at the language level;
   what remains is *validating* it against the test oracle (see PR0). Note the
   `eval` task-string driver in `simulation.py` survived the port intact — the
   Py3 migration did not touch the architecture, which is exactly the seam this
   project removes.
2. **Every change is validated against a trajectory oracle.** The existing tests
   are the oracle — especially `tests/test_restart.py`,
   `test_prop.py`, and `test_wigner.py`. A refactor step is only complete when it
   reproduces a known trajectory within tolerance. If a step changes numerical
   output, it is wrong until proven otherwise.
3. **No behavior change inside a refactor step.** Extraction/porting steps are
   pure refactors. Behavior changes (new async driver, new executor) come *after*
   the seam exists and *only* behind that seam.
4. **Determinism disciplines are load-bearing and invisible.** The whole
   resilience model depends on: (a) all randomness drawn from a single serialized
   PRNG carried in the durable state — never a bare `np.random` / global seed;
   (b) no electronic-structure point recomputed within a run (the cache/store is
   authoritative, so NAC phases can't silently flip); (c) no side effects (file
   writes, logging) in the physics except through the framework. These are
   exactly the things a "cleanup" refactor tends to remove. Do not remove them.
   If you must touch them, call it out explicitly in the PR description.
5. **Do not rewrite the validated physics.** The FMS matrix elements
   (`build_S`, `build_Sdot`, `build_H`, `build_V`, `build_tau`, `build_T`,
   `build_Heff` in `simulation.py`, ~lines 475–617), the spawning logic
   (`spawn_as_necessary`, ~line 782), `complexgaussian.py`, and the `traj`
   propagators are assets to **preserve and move**, not re-derive.

## Where the electronic-structure seam is (the key question)

The ES backend is **already modularized** — this is the single most important
finding, and it means the seam mostly *exists* and PR1 is smaller than feared:

- Each QM backend is its own module in `pyspawn/potential/` — `terachem_cas.py`,
  `molcas_cas.py`, `terachem_dft.py`, `test_cone.py` — and each defines a
  function **`compute_elec_struct(self, zbackprop)`** with a consistent signature
  (see e.g. `potential/terachem_cas.py` ~line 22).
- These functions are attached to the `traj` class by **monkey-patching via
  `exec`**: `import_methods.into_traj(x)` (`pyspawn/import_methods.py`) loops over
  a backend module's attributes and does `exec("traj." + method + " = x." +
  method)`. The example scripts call
  `pyspawn.import_methods.into_traj(pyspawn.potential.terachem_cas)` to select a
  backend. (The same `exec`-injection wires the classical integrator, QM
  Hamiltonian, and QM integrator — see `into_simulation`.)
- At runtime the ES call is `self.compute_elec_struct(zbackprop)`. **Correction
  (verified while doing PR1): it is NOT called from `propagate_step`.** Its live
  callers are the *injected* velocity-Verlet integrator
  `classical_integrator/vv.py` (×3: `prop_first_step` at `x_t` and `x_tpdt`,
  `prop_not_first_step` once), `traj.compute_centroid` (`traj.py` ~line 818), and
  `hessian.py` (×3). So the classical+ES fusion lives in the injected `vv.py`
  integrator, not in `traj.py`. Each caller does `compute_elec_struct` then reads
  `forces_i`/`energies` straight back via getters — the "mutate `self`, re-read
  via getter" pattern the `ESResult` seam replaces. (`traj.py` ~782 used to hold
  a commented-out `eval`-string dispatcher; PR1 replaced it with a native
  `compute_elec_struct` shim over the backend registry.)

**Seam verdict:** PR1 is *"formalize a seam that already exists."* The backends
are already the right granularity — the work is to replace the `exec`-monkeypatch
injection with a clean interface (a registry or an `ElectronicStructureBackend`
base class the backends subclass), and to change the contract so
`compute_elec_struct` **returns an `ESResult`** rather than mutating `self` in
place. Two things to untangle in the process: (a) it is currently fused with
classical propagation inside the injected `vv.py` integrator (not
`propagate_step` — see the corrected call-site note above); separate them.
(b) backend selection is by injection at import time; make it explicit. Map the target
contract to `design/nextfms_skeleton/nextfms/es_provider.py` (`ESRequest` / `ESResult` /
`compute_one`, plus the `supports`/coupling-ladder pattern). Note the injection
also covers the classical integrator and QM Hamiltonian/integrator — the same
"replace `exec`-injection with a real interface" move applies to all of them, but
ES is the one that matters for the driver refactor.

## PR0 status — DONE (hermetic oracle established)

PR0 is complete on the `python3` branch. What was found and done:

- **Two stale Py2→3 breakages fixed** (parser-clean but broken at runtime):
  - `potential/test_cone.py` used `exec("x = ...")` to write function locals —
    a no-op in Py3. Replaced with explicit `getattr` dispatch. (Only this file
    had the pattern; backends were clean.)
  - `tests/test_prop.py` and `tests/test_restart.py` called the pre-migration
    API (`pyspawn.traj()` with no args; `sim.read_from_file`). Updated to the
    current API (`pyspawn.traj(ndims, nstates)`; `restart_from_file(json, h5)`),
    matching the SSAIMS example.
- **Hermetic oracle added: `tests/pr0_oracle.py`** — runs only on the analytic
  cone model (no TeraChem/Molcas) and asserts, with `max|diff| == 0` across 100
  HDF5 datasets:
  1. propagation determinism (two seeded runs are bit-identical), and
  2. **restart equivalence** (interrupt at t=2.0, resume to t=4.0 == uninterrupted
     to t=4.0) — the durability property PR3 must not regress.
  Run it: `PYTHONPATH=<repo> python tests/pr0_oracle.py` (exit 0 = PASS;
  `PR0_ORACLE_VERBOSE=1` to see pyspawn's log). This is the regression gate for
  every later PR.
- **`tests/test_prop_seeded.py`** — seeded, deterministic start script (the old
  `test_prop.py` drew unseeded initial conditions, so it could not be a golden
  reference).
- **Environment:** runs under pyenv **`TJM-3.13`** (Python 3.13, has
  numpy/h5py/matplotlib). No conda. `.python-version` at the repo-parent pins it.

**Observations for later PRs (NOT fixed in PR0):**
- The `eval`-task-string driver prints hundreds of lines of `### ...` log per
  step to stdout (~77k lines for a 4 a.u. run). The verbosity is a symptom of
  the driver design; PR3 replaces it. Oracle mutes stdout to cope.
- `traj.py:1052` emits `RuntimeWarning: invalid value encountered in sqrt`
  (`Wlj = np.sqrt(1 - ...)` going slightly negative) in the cone NPI-coupling
  path. Pre-existing; does not affect reproducibility. Worth a guard eventually.

## Staged plan (each step independently validated; do them in order)

Do **not** attempt a big-bang rewrite. Each step below has its own regression
oracle (reproduce a known trajectory) so a divergence localizes to one step.

- **PR0 — Validate the existing Py3 port + establish the test-oracle baseline.**
  The language port is already done on the `python3` branch (see constraint #1),
  so this is NOT a porting task — it is making the ported code *run* and
  capturing golden trajectories. Steps: (a) create a Python-3 environment with
  the third-party deps (`h5py`, `numpy`, etc. — `pyspawn` imports but the env
  here lacks `h5py`); (b) get the test suite executing under Py3 (`tests/`
  targets external QM codes, so identify which tests run without TeraChem/Molcas
  — `test_cone` uses the built-in `potential/test_cone.py` model and is the
  likely candidate for a hermetic oracle); (c) run the runnable tests and **save
  their trajectory output as golden reference files** — every later PR is
  validated against these. (d) Audit the two things a language port can silently
  break but a parser won't catch: **integer vs. true division** (`/` changed
  meaning Py2→3; check force/normalization math) and **dict iteration order /
  `.keys()` returning views not lists** (the task-queue and traj-dict loops).
  Oracle: the ported code reproduces physically sensible trajectories that become
  the baseline. No architecture change.
- **PR1 — Extract the ES seam.** Lift `traj.compute_elec_struct` into a pure
  `ElectronicStructureBackend.compute_one(request) -> ESResult`; replace the
  string-name backend dispatch with a registry. Zero behavior change; validate
  against `test_prop.py` / `test_restart.py`. This PR sizes the whole job.
- **PR2 — Centralize the PRNG into serialized state.** Route all randomness
  (SSAIMS selection `ss_seed`, Wigner sampling, any SH draws) through one PRNG
  carried in the restart state. Oracle: `test_wigner.py` + a seeded SSAIMS run
  reproduces exactly. Precondition for trusting replay.
- **PR3 — Replace the task-queue driver.** Swap the `update_queue`/`eval` loop
  for a serial-reading `advance(state)` over the PR1 seam, with an executor that
  caches ES results keyed by request and blocks until a step's evaluations are
  done. Start with a threaded executor; the physics loop reads forward. Oracle:
  end-to-end trajectory + restart identical to pre-refactor.
- **PR4 — Rent the resilient tier (optional/next-gen).** Back the executor with
  Parsl (HPC scheduling + durable app-cache) so runs distribute across a cluster.
  Physics tier unchanged. See the `ParslExecutor` sketch in
  `design/nextfms_skeleton/nextfms/executor.py`.

## Build / run / test

Work on the **`python3` branch** (`git checkout python3`; `origin` = the
`mtzgroup/py3Spawn` fork). The package is Python 3 but has no
`setup.py`/`pyproject` — it runs by putting `pyspawn/` on `PYTHONPATH` and
importing, and it needs third-party deps installed (`numpy`, **`h5py`**, and for
real runs the external QM codes TeraChem / OpenMolcas). Import currently fails
only for a missing `h5py`, not for any syntax reason.

```bash
git checkout python3            # NOT master (master is still Python 2)
# Use pyenv TJM-3.13 (pinned in .python-version) — it ALREADY has
# numpy/h5py/matplotlib. Do NOT use conda. Only install if a dep is missing.
cd tests
# Many tests (test_tc_*) require TeraChem/Molcas. Start with ones that use the
# built-in analytic model potential (pyspawn/potential/test_cone.py):
python test_prop.py
python test_restart.py          # the restart/resume oracle — leaned on hardest by PR3
python test_wigner.py           # the RNG oracle — leaned on by PR2
```

**First real action for PR0:** determine which tests run *without* an external QM
code, get those green under Py3, and freeze their output as golden trajectories.
The dig below already did that triage.

### Test oracle map (from inspecting `tests/` + `potential/`)

Backend wiring splits the suite cleanly:

| Test | Backend | External QM? | Role |
|---|---|---|---|
| `test_prop.py` | `potential.test_cone` | **No — pure numpy** | full propagation; **primary hermetic oracle** |
| `test_timer.py` | `potential.test_cone` | **No** | longer cone run (tfinal=40); walltime path |
| `test_restart.py` | *(none — restored from `sim.json`)* | No, but needs a prior run's `sim.json` | **PR3 restart oracle** |
| `test_analysis.py` | *(none)* | No, but needs `sim.hdf5` + matplotlib | analysis/plotting only |
| `test_wigner.py` | `terachem_cas` | **Yes** | PR2 RNG oracle — but needs TeraChem |
| `test_hessian.py`, `test_tc_*` | `terachem_cas` | **Yes** | require TeraChem/Molcas — not hermetic |

**`test_cone` is a pure analytic conical-intersection model** (`potential/test_cone.py`,
math+numpy only, no subprocess/socket) — so `test_prop`/`test_timer` are the
self-contained oracle path. Two blockers to fix in PR0 before they run:

1. **`exec`-into-locals Py3 bug (BLOCKER, localized to `test_cone.py`).**
   `compute_elec_struct` does `exec("x = self.get_...positions()[0]")` then uses
   `x`. In Py2 `exec` wrote into function locals; in **Py3 `exec` is a function
   and the assignment does NOT leak into the enclosing scope** → `NameError`.
   Confirmed by direct test. Fix: replace the 2 assignment-`exec` lines with
   normal assignments using `getattr`/`setattr` (e.g.
   `x = getattr(self, "get_"+cbackprop+"positions")()[0]`). Good news: this
   pattern is **NOT systemic** — only `test_cone.py` has it; the real backends
   (`terachem_cas`, etc.) don't. But note this is the *kind* of silent Py2→3
   breakage to grep for everywhere: `grep -rn 'exec ?("[a-z_]* ?='`.
2. **Unseeded initial conditions → not reproducible.** `test_prop` draws
   `pos`/`mom` from `np.random.normal` with **no seed**, so its trajectory
   differs every run and can't be a golden file as-is. For PR0, add a fixed seed
   (or hardcode initial `pos`/`mom`) to freeze a deterministic reference. This is
   the same disease PR2 cures globally — PR0 just needs one seeded run to capture
   a baseline.

**Recommended PR0 oracle build:** fix the `test_cone` `exec` bug → make a seeded
variant of `test_prop` → run it → save the resulting `sim.hdf5`/`sim.json` as the
golden reference → then `test_restart` (which restores backend + state from that
`sim.json`) becomes a live restart oracle chained off it. That gives you a
hermetic propagation oracle **and** the restart oracle PR3 leans on, with zero
dependency on TeraChem/Molcas.

The target-state skeleton is Python 3, self-contained, and runnable now:

```bash
cd design/nextfms_skeleton
PYTHONPATH=. python demo_replay.py     # crash+restart reproduces trajectory
PYTHONPATH=. python demo_methods.py    # same driver runs AIMS and FSSH
```

## Conventions for this repo

- When editing the repo, prefer minimal diffs; this is someone else's
  validated research code. Explain any numerical-path change in the commit.
- Reference the skeleton by file when proposing a change: "make
  `traj.compute_elec_struct` return an `ESResult` like
  `design/nextfms_skeleton/nextfms/es_provider.py`."
- Never introduce a bare `np.random`, a module-level seed, or a file write
  inside physics code — see hard constraint #4.
- Keep the two-tier split visible: physics files read top-to-bottom with no
  async/executor references; all scheduling lives in one place.
