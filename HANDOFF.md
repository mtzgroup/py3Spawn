# HANDOFF

## Status

PySpawn refactor, branch `python3`, all work committed and **pushed to
`origin/python3`** (fork `mtzgroup/py3Spawn`), working tree clean. **PR1 and PR1b
are DONE**: the electronic-structure seam is now a real interface — a name
registry + `compute_one(ESRequest) -> ESResult` with an opaque `ESState`
continuation handle — instead of the old monkeypatch injection, and every ES call
(trajectory, centroid, Hessian) routes through it. A NPI physics bug was fixed
(flagged, gated) and a record/replay fixture harness was added. The hermetic cone
oracle (`tests/pr0_oracle.py`) held at **OVERALL: PASS, max|diff|=0** through every
step. The immediate next task needs a **TeraChem machine**: capture one QM
trajectory as a replay fixture (`design/qm_fixture_capture.md`) so the QM path
gets a hermetic regression gate.

Working disciplines (keep these): every change is gated by `tests/pr0_oracle.py`
at `max|diff|=0`; PRs/steps are small and isolated; commits are authored as
**Todd J. Martinez <toddjmartinez@gmail.com>** with a `Co-Authored-By: Claude`
trailer; **Todd runs `git push`** (Claude commits locally — though pushing has
worked directly from Todd's terminal with his SSH keys). Env: Python 3.13; deps in
`requirements.txt` (numpy/h5py/matplotlib); `tcpb` only for TeraChem.

## This session

- **PR1 (had just landed) → PR1b (this session).** Narrowed the seam from
  `compute_one(self, traj, zbackprop)` to `compute_one(ESRequest)`. The cone
  backend now computes from request values alone and never touches a `traj`.
  Introduced `ESState` (opaque continuation; cone = `wf`). Sealed the Hessian
  bypass. QM backends still ride `LegacyMutatingBackend` via a transitional
  `request.traj`/`request.zbackprop` escape hatch.
- **Key finding — `prev_wf` was write-only dead state.** Traced every reader:
  after PR1 nothing reads `get_prev_wf`; the real continuation state is `wf`.
  Removed `prev_wf`. (Todd asked to verify closely before I acted — good call; the
  restart oracle can't prove it because it restores `prev_wf` on both legs.)
- **Physics fix.** `compute_tdc`'s clamp block tested the off-diagonal `W[0,1]`/
  `W[1,0]` but wrote the diagonal, so `arcsin(|W|>1)` could NaN. Fixed. Verified
  via instrumentation that the cone **never** exercises it (peak |W_offdiag| ~=
  0.9998), so `pr0_oracle.py` is unchanged — which also demonstrated that the
  oracle certifies the *covered trajectory*, not the *code*. Added
  `tests/test_compute_tdc.py` (red on bug, green on fix). A separate latent
  fragility (`Wlk`/`Etmp` unguarded `sqrt`) was noted, NOT fixed (out of scope).
- **Design docs written and pressure-tested.** `current_architecture.md`,
  `es_seam_contract.md`, `redesign_proposal.md`. Pressure tests (logged in
  redesign §8): retiring the `*_qm` HDF5 "bus" into an executor cache HOLDS but
  the cache key must be provenance `(label, time)` NOT geometry, and the driver is
  two cursors with a spawn-causality barrier; Layer-2 god-object decomposition
  must be emergent (seam-driven), not a standalone refactor.
- **Biggest architectural insight:** `working.hdf5` is not just the restart store
  — it is the *synchronization bus* (per-TBF ES outputs are written, then re-read
  into `*_qm` fields at a common quantum time for the coupled Hamiltonian).
- **Fixture harness.** `es_record_replay.py` (RecordingBackend/ReplayBackend),
  validated on the cone at `max|diff|=0`. Turns one TeraChem run into a permanent
  hermetic QM oracle.
- **Repo hygiene.** Untracked `.python-version` (pyenv pin `TJM-3.13`, not
  portable) + gitignored it; added `requirements.txt`; updated README.

## Open questions

- **QM path is unvalidated here.** The cone oracle is structurally blind to
  `S_elec` / DGAS / QM continuation state. The QM backends also have not actually
  *run* under Py3 — there may be latent port bugs (bytes/str at the tcpb boundary,
  `np.fromfile`, integer division). The record/replay fixture is the plan to close
  this; a first QM run may surface port bugs to fix.
- **`ESState` still lands on the `traj`** (serialization unchanged). Relocating it
  into the executor's durable store is deferred to PR3 (and is what shrinks the
  god-object checkpoint).
- **Boundary-row rule** for the record/replay + `*_qm` retirement: whether the
  forward and backprop first steps both emit a row exactly at `firsttime` is not
  fully pinned (redesign §8.2). Nail before PR3.
- **`.tape` naming / committing the fixture:** decide whether the captured QM tape
  + golden hdf5 get committed (reproducible CI oracle) or kept out (size).

## Next steps

1. **Capture the QM fixture (needs TeraChem).** On the TeraChem machine: clone,
   `pip install -r requirements.txt` (+ `tcpb`), confirm the port with
   `python tests/pr0_oracle.py`, then run a short *deterministic* QM example
   wrapping the backend in `RecordingBackend` and save `qm_fixture.tape` +
   `qm_golden.hdf5`. Full steps in `design/qm_fixture_capture.md`. Watch for latent
   Py3 breakage in `terachem_cas.py` while you're there.
2. **Build the QM replay oracle** (`tests/qm_replay_oracle.py`) from the captured
   fixture using `ReplayBackend` + `max_hdf5_diff` — hermetic, runnable anywhere.
   This is the gate for step 3.
3. **Migrate the QM backends off `LegacyMutatingBackend`** to native
   `compute_one(ESRequest)` (terachem_cas/terachem_dft/molcas_cas), gated by the
   step-2 oracle. Then drop the transitional `request.traj`/`zbackprop` escape
   hatch from `ESRequest`.
4. **PR2 — centralize the PRNG** into serialized state (precondition for trusting
   replay), then **PR3** — replace the `eval` task-queue driver with the serial
   `advance()` over an executor, retiring the `*_qm` bus into a provenance-keyed
   cache (see `design/redesign_proposal.md`).

## Files touched

- `pyspawn/npi_coupling.py` — NEW: shared traj-free NPI kernel `compute_npi_tdc(W, dt)`.
- `pyspawn/potential/es_backend.py` — `ESState`/`ESRequest` added; `compute_one`
  narrowed to `(request)`; `ESResult` carries `state`; `prev_wf` write dropped.
- `pyspawn/potential/test_cone.py` — `TestConeBackend.compute_one(req)` computes
  from values, uses the shared NPI kernel, returns `ESState(wf=...)`.
- `pyspawn/traj.py` — `compute_tdc` delegates to the kernel (clamp bug fixed);
  `compute_elec_struct` builds an `ESRequest`; `prev_wf` no longer written.
- `pyspawn/import_methods.py` — `into_hessian` wires `_es_backend` (Hessian bypass
  sealed).
- `pyspawn/es_record_replay.py` — NEW: Recording/Replay backends + tape + hdf5 diff.
- `tests/test_compute_tdc.py` — NEW: gate for the NPI clamp fix.
- `tests/test_es_replay.py` — NEW: hermetic self-validation of the fixture harness.
- `design/{current_architecture,es_seam_contract,redesign_proposal,qm_fixture_capture}.md`
  — NEW: architecture map, seam contract, target architecture + pressure-test log,
  QM fixture capture instructions.
- `CLAUDE.md` — added the "PR1 / PR1b status — DONE" section (auto-loaded);
  corrected the ES call-site description.
- `README.md`, `requirements.txt`, `.gitignore`, `.python-version` — env
  portability (untrack the pyenv pin, add requirements, reinforce Python 3).
