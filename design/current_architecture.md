# PySpawn — current architecture (as-is map)

**Purpose.** A navigable map of how the code is laid out *today*, so the physics,
the data structures, and the electronic-structure ↔ dynamics interface can be
evaluated on their own terms — including the question "is this worth salvaging,
and which parts?" Every claim is anchored to `file:line` (branch `python3`, as of
the PR1 seam work). Line numbers drift with edits; treat them as pointers.

Companion document: [`es_seam_contract.md`](es_seam_contract.md) — the proposed
tight/opaque ES seam. This document is descriptive (what exists); that one is
prescriptive (where we're going).

---

## 1. The big picture: three tiers and one bus

PySpawn is an ab-initio multiple-spawning (AIMS) code. Structurally it is three
tiers plus a shared file that couples them:

1. **Classical + ES tier (per trajectory).** Each trajectory basis function
   (TBF) is propagated on *its own clock* by a velocity-Verlet integrator that
   calls electronic structure at each step. `traj.propagate_step` →
   `vv.prop_*_step` → `traj.compute_elec_struct` → ES backend.
   Files: `traj.py`, `classical_integrator/vv.py`, `potential/*`.

2. **Quantum tier (coupled amplitudes).** The TBFs form a non-orthogonal basis;
   their complex amplitudes `C` evolve under an effective Hamiltonian built from
   *all* TBFs and centroids evaluated at one common time. `simulation` builds the
   FMS matrices `S, Sdot, H, Heff`; `qm_integrator/rk2.py` propagates `C`.
   Files: `simulation.py`, `qm_hamiltonian/{adiabatic,dgas}.py`,
   `qm_integrator/rk2.py`, `complexgaussian.py`.

3. **Scheduler tier (the inversion of control).** The main loop does not call the
   physics directly. It builds a queue of task *strings*, sorts them by
   simulation time, pops the earliest, and `eval()`s it — reconstructing dynamics
   from externalized state so a killed run can resume.
   File: `simulation.py` `propagate()` / `update_queue()`.

4. **The bus: `working.hdf5`.** This is not only the restart history. It is the
   *synchronization channel* between tier 1 and tier 2. Tier 1 writes each TBF's
   state to HDF5 as it propagates on its own clock; tier 2 reads it *back* at a
   shared "quantum time" into `*_qm` snapshot fields, so the coupled Hamiltonian
   sees every TBF/centroid at one consistent instant (§4c).

The determinism/durability model (constraint #4 in `CLAUDE.md`) is what makes the
`eval` scheduler survivable: all state is serialized every cycle, so replay
reconstructs an identical trajectory.

---

## 2. Core data structures

### 2.1 `fmsobj` — the serialization base (`fmsobj.py`)

Every persistent object subclasses `fmsobj`. It provides the JSON (de)serializer
used for restart:

- `to_dict()` (`fmsobj.py:14`) copies `self.__dict__`, converts numpy arrays to
  nested lists, encodes complex numbers as `"^complex(re,im)"` strings
  (`:24–38`), and *recurses into nested `fmsobj` instances* — tagging each with a
  `fmsobjlabel` so it can be reconstructed (`:37–51`). Dicts of `fmsobj`s (the
  `traj` and `centroids` dicts) are recursed element by element.
- `from_dict()` (`:54`) inverts it; `write_to_file`/`read_from_file` (`:93`,
  `:100`) are the JSON I/O.

Consequence: **anything you store as an attribute on a `traj`/`simulation` is
automatically serialized.** This is the mechanism that makes ES continuation
state durable (§4f) — and the reason that state is smeared across `traj` rather
than encapsulated.

### 2.2 `traj` — the central object (`traj.py`)

`traj(fmsobj)` (`:8`), constructed `traj(numdims, numstates)`, `__init__` at
`:16–79`. It is effectively a *god object*: nuclear dynamics, electronic outputs,
electronic continuation state, time-history, synchronization snapshots, and spawn
bookkeeping all live here. One class-level attribute precedes `__init__`:
`_es_backend = None` (`:14`, the PR1 ES seam; not serialized).

Attribute inventory (categorized — this partition is the spine of §4):

- **(a) Nuclear dynamics state.** `positions, momenta, widths, masses` (`:23–26`),
  `time, time_half_step, timestep, firsttime, maxtime, mintime` (`:17–32`).
  Time-history copies for the multistep integrator:
  `positions_{t,tpdt,tmdt}` (`:59–61`), `momenta_{t,tpdt,tmdt}` (`:62–64`),
  `energies_{t,tpdt,tmdt}` (`:65–67`).
- **(b) Electronic-structure OUTPUT quantities.** `wf` (`:38`), `energies`
  (`:40`), `forces` (`:41`; per-state accessor `get_forces_i()` returns
  `forces[istate,:]`, `:607`), `timederivcoups` (`:42`), `S_elec_flat` (`:43`,
  flattened electronic overlap matrix).
- **(c) ES CONTINUATION state.** In `__init__`: `prev_wf` (`:39`). **Added lazily
  by the QM backends, not in `__init__`:** `orbs, civecs, electronic_phases,
  prev_wf_positions, norbs, ncivecs` (and MOLCAS `wfn, inporbs`). Confirmed absent
  from the constructor; created at first `compute_elec_struct`, copied on
  spawn/centroid only behind `hasattr` guards (`init_spawn_traj` `:485`,
  `init_centroid` `:544`).
- **(d) `*_qm` synchronization snapshots.** `positions_qm, momenta_qm,
  energies_qm, forces_i_qm, timederivcoups_qm` (`:72–76`). Not live state — a
  time-synchronized snapshot reloaded from HDF5 (§4c).
- **(e) Amplitude / spawn / control.** `istate, numstates, numdims, length_wf`
  (`:22–37`), `label` (`:28`), `spawntimes, spawnthresh, spawnlastcoup` (`:56–58`),
  `z_spawn_now, z_dont_spawn` (`:68–69`), `h5_datasets, h5_datasets_half_step`
  (`:29–30`), `numchildren` (`:70`), `tc_port` (`:79`). **Note:** the complex
  amplitude `C` is *not* here — it lives on `simulation` (§2.5b).
- **(f) `backprop_` mirror.** AIMS back-propagates each TBF to fill its past;
  twins exist for time, energies, forces, positions, momenta, wf, prev_wf,
  timederivcoups, S_elec_flat (`:45–54`) plus lazy ES-continuation twins.
  `set_backprop_timederivcoups` negates the sign (`:709`) — derivative w.r.t. `-t`.

Accessor surface: ~70+ `get_`/`set_` methods. The nuclear/control ones are
defined here; **most ES-specific accessors (`get_orbs`, `set_civecs`,
`get_electronic_phases`, `get_wf0/1`, …) are *not* — they are attached by the
backend layer** (§6). `traj.py` only touches them behind `hasattr` guards. This is
the ES boundary leaking into the physics object.

### 2.3 centroid — a `traj` carrying a state pair

A centroid is an *ordinary* `traj` instance built by
`traj.init_centroid(existing, child, label)` (`traj.py:520`), distinguished only
by (i) carrying both `istate` and `jstate` (`:527`) and (ii) a label with an
`_a_` separator (`"<keyi>_a_<keyj>"`). It represents the width-weighted phase-space
midpoint of two TBFs (positions/momenta averaged in `update_centroids`,
`simulation.py:721,766`) and holds the coupling ES data (energies,
timederivcoups, S_elec_flat) that fills *off-diagonal* FMS matrix elements. There
is no separate centroid class.

### 2.4 `hessian` — a `traj` subclass (`hessian.py:9`)

`hessian(traj)` reuses the whole nuclear/accessor surface for Hessian/Wigner
setup. **It is a second ES entry point that bypasses the PR1 seam** — see §6.

### 2.5 `simulation` — the orchestrator (`simulation.py`)

`simulation(fmsobj)` (`:25`), `__init__` `:29–88`. Note the FMS matrices are
*not* created in `__init__`; they are built lazily inside the `build_*` methods.

- **(a) Trajectory containers.** `traj = dict()` (`:31`, keyed by label),
  `centroids = dict()` (`:35`, keyed `"<i>_a_<j>"`), `traj_map` (`:58`,
  label→matrix index, maintained in `add_traj` `:157`), `istates_dict` (`:61`,
  analysis), `num_traj_qm` (set via `compute_num_traj_qm` `:449`).
- **(b) Quantum amplitude state.** `qm_amplitudes` — the complex `C` vector
  (`:64`, grown in `compute_num_traj_qm` `:451`, init `init_amplitudes_one`
  `:436`). `qm_energy_shift` (`:67`). Populations are computed transiently
  (`c* S c`), not stored.
- **(c) FMS Hamiltonian matrices (lazy).** `S` (`build_S` `:483`), `Sinv`
  (`invert_S` `:520`), `Sdot` (`:502`), `H` (`:535`), `Heff` (`:616`), `V`
  (`:548`), `tau` (`:573`), `T` (`:596`). `dgas_coeffs`/`dgas_coeffs_next_time`
  exist *only* under the DGAS mixin (`dgas.py:84`).
- **(d) Time / control.** `quantum_time` (`:47`), `quantum_time_half_step`
  (`:49`), `timestep` (`:51`), `max_quantum_time` (`:74`, loop terminator `:325`),
  `max_walltime` (`:77`), `olapmax` (`:44`, spawn overlap ceiling).
- **(e) Task queue.** `queue = ["END"]` (`:38`, always sentinel-terminated),
  `tasktimes` (`:40`). Manipulated by `add_task` (`:171`), `pop_task` (`:621`),
  `insert_task` (`:663`), `update_queue` (`:623`).
- **(f) SSAIMS state.** `ssa_*` controls (`:80–88`, plus `enable_ssaims`
  `:1131`), including `ssa_seed` and `ssa_rand_calls` — the stochastic-selection
  RNG bookkeeping (relevant to PR2).

---

## 3. Control flow: one `propagate()` cycle

`propagate()` (`simulation.py:308`) is a `while True` loop; each iteration
(`:312–362`):

1. `update_centroids()` (`:316`) — recompute centroid positions/momenta, mark
   which are computable this step (nuclear overlap `> 0.001`, else zero the
   coupling data).
2. `update_queue()` (`:320`) — rebuild the task list (below).
3. **Terminate** (`:325`) if `quantum_time + 1e-6 > max_quantum_time` (delete
   working files and `return`). **Walltime** guard (`:335`).
4. If `queue[0] != "END"` (`:341`): `current = pop_task()` (`:345`);
   **`eval(current)` (`:347`)** — run exactly one task string this cycle. This is
   the inversion of control, in its most literal form.
5. `spawn_as_necessary()` (`:354`).
6. `propagate_quantum_as_necessary()` (`:358`).
7. `restart_output()` (`:362`) — must be last (durability barrier).

**`update_queue()` (`:623`)** builds four task categories, each inserted in
ascending time order by `insert_task` (`:663`), so the earliest-in-sim-time task
runs first:

| Category | Task string | Time key | Guard |
|---|---|---|---|
| Forward traj | `self.traj["<k>"].propagate_step()` | `get_time()` | `maxtime > time` |
| Backward traj | `self.traj["<k>"].propagate_step(zbackprop=True)` | `get_backprop_time()` | `mintime < bp_time` |
| Forward centroid | `self.centroids["<k>"].compute_centroid()` | `get_time()` | `get_z_compute_me()` |
| Backward centroid | `self.centroids["<k>"].compute_centroid(zbackprop=True)` | `get_backprop_time()` | `get_z_compute_me_backprop()` |

**The amplitude propagation is NOT in the queue.** It is driven separately by
`propagate_quantum_as_necessary()` (`:364–430`), gated on `max_info_time` — the
earliest time to which *all* TBFs and centroids have data. While
`max_info_time > quantum_time`, it calls `qm_propagate_step()` (rk2, `:421/424`),
`ssaims_step()` (`:427`), `h5_output()` (`:430`). So tier 1 (per-TBF ES+classical)
runs ahead task-by-task; tier 2 (amplitudes) advances only once every TBF has
reached the next quantum time. `max_info_time` is the coupling point between the
two clocks.

---

## 4. The ES ↔ dynamics interface (the detailed shape)

This is the interface the refactor most needs to get right. It has an unusual
two-path shape.

### 4a. Where the seam is (post-PR1)

`traj.compute_elec_struct(zbackprop)` (`traj.py:787`) is a native shim →
`self._es_backend.compute_one(self, zbackprop)` then `apply_to_traj(...)`.
Backend selection is a registry wired by `import_methods.into_traj` (§6). Live
callers:

- `vv.py` velocity-Verlet ×3 (`:21` at `x_t`, `:38` at `x_tpdt` in
  `prop_first_step`; `:85` in `prop_not_first_step`) — the classical+ES fusion.
- `traj.compute_centroid` (`traj.py:832`).
- `hessian.build_hessian_hdf5_semianalytical` ×3 (`hessian.py:14,58,65`) —
  **bypasses the seam** (§6).

### 4b. What crosses the seam — three categories

Auditing every ES quantity by who consumes it:

| Category | Quantities | Consumed by | Must stay… |
|---|---|---|---|
| **1. Consumed by dynamics** | `energies`, `forces` (`forces_i`), `timederivcoups`, `S_elec_flat` | integrator (vv), spawning, **FMS Hamiltonian builders** | visible/typed |
| **2. ES continuation state** | `orbs`, `civecs`, `electronic_phases`, `prev_wf_positions`, `norbs/ncivecs`, `wf`/`prev_wf`, MOLCAS `wfn/inporbs` | *only the ES backend itself*, next step | opaque handle |
| **3. Recorded, not consumed** | `wf0`/`wf1` (cone) and any logged-only datasets | HDF5 log / analysis only | recorded |

The categories overlap: `wf` is both (2) — the backend reads `prev_wf` to phase —
and (3) — `wf0/wf1` are HDF5 datasets. `S_elec_flat` is (1) *and* (3). The point
of [`es_seam_contract.md`](es_seam_contract.md) is to separate these cleanly: the
dynamics should see only (1), never (2).

### 4c. The two-hop path from ES output to the Hamiltonian (the `*_qm` bus)

**Dynamics does not read ES outputs directly from the trajectory that produced
them — it reads a synchronized snapshot reloaded from HDF5.** Two distinct
consumers, two mechanisms:

- **Integrator (same TBF, same step): direct.** `vv.py` calls
  `compute_elec_struct` then immediately reads `get_forces_i()`/`get_energies()`
  off the *same* `traj` (`:22,24,39,40,86,87`). Direct, in-memory, same clock.
- **Coupled Hamiltonian (all TBFs, one shared time): via HDF5.** Each TBF writes
  `energies/forces/timederivcoups/S_elec_flat` to `working.hdf5` during
  propagation (`h5_output`). Then `get_all_qm_data_at_time_from_h5`
  (`traj.py:987`, `_half_step` `:1014`) reads those datasets back at the quantum
  time and does `setattr(self, dset + "_qm", data)` (`:1009,1036`). The matrix
  builders read only the `*_qm` fields. `simulation.get_qm_data_from_h5` (`:453`)
  and `_half_step` (`:466`) refresh TBFs and centroids before each `Heff` build.

So the ES→quantum coupling is *mediated by the HDF5 bus*: to reach the coupled
Hamiltonian, an ES output must (i) be declared in `h5_datasets` and (ii) round-trip
through `working.hdf5` into its `*_qm` field. That is why HDF5 is load-bearing for
correctness, not just restart — and why the `*_qm` snapshot exists at all
(different TBFs live on different clocks; the snapshot re-synchronizes them).

### 4d. ES → Heff dependency (adiabatic path — the hermetic/cone oracle)

`qm_hamiltonian/adiabatic.py` defines only the two orchestrators
`build_Heff_{first,second}_half` (`:6,26`), which set the half-step time and run
`build_S → invert_S → build_Sdot → build_H → build_Heff`. The `build_*` bodies are
the **defaults in `simulation.py`** (`:479–616`):

| Builder (`simulation.py`) | ES input (getter) | Produces |
|---|---|---|
| `build_S` (`:483`) | none directly; nuclear `positions_qm,momenta_qm` via `cg.overlap_nuc_elec`; electronic part is a **δ on `istate`** | `S` |
| `build_Sdot` (`:502`) | **`forces_i_qm`** (+ positions/momenta) via `cg.Sdot_nuc_elec` | `Sdot` |
| `build_V` (`:548`) | **`energies_qm[istate]`** (diagonal; centroid off-diag when `istate==jstate`) | `V` |
| `build_tau` (`:573`) | **`timederivcoups_qm[jstate]`** at centroids (`istate≠jstate`) | `tau` |
| `build_T` (`:596`) | none ES; nuclear `positions_qm,momenta_qm,widths,masses` | `T` |
| `build_H` (`:535`) | — `T + V + tau` | `H` |
| `build_Heff` (`:616`) | — `Sinv·(H − i·Sdot)` | `Heff` |

**ES outputs that reach `Heff` (adiabatic):** energies → `V`; time-derivative
couplings (at centroids) → `tau`; forces → `Sdot`; positions/momenta/widths/masses
→ `S`,`T`, all overlaps. The electronic overlap is an implicit `istate`-δ inside
`complexgaussian`'s `*_elec` wrappers (`:17–21,63–67,117–121`) — no electronic
overlap matrix is materialized. `S_elec_flat` is *not read* on this path.

### 4e. DGAS / SSAIMS path (`qm_hamiltonian/dgas.py`) — what differs

DGAS overrides the whole pipeline. The architectural differences:

1. **Explicit electronic overlap.** `dgas_coeffs[i,j,:]` (unit vector on the
   occupied adiabatic state, `build_DGAS_coeffs` `:67`) builds an explicit
   `S_elec[i,j]` matrix (`build_S_elec_DGAS` `:118`) that multiplies *every*
   nuclear matrix: `S = S_nuc * S_elec` (`:145`), `Sdot_nuc * S_elec` (`:157`),
   `T = kinetic_nuc * S_elec` (`:369`). Contrast the adiabatic δ.
2. **`S_elec_flat` is consumed here.** `build_Sdot_elec_DGAS` reads
   `centroid.get_S_elec_flat().reshape((nstat,nstat))` (`:170`) — the raw
   ES electronic-overlap matrix — and derives NPI derivative couplings
   analytically. **This is the sole consumer of `S_elec_flat`** and the reason it
   is category (1), not continuation state.
3. **`tau` is dropped** (`:300,306` commented); nonadiabatic coupling moves into
   `Sdot_elec`.
4. **`V` off-diagonals** couple different-state centroids via a state-summed
   energy expectation (a diabatic gap over `E[ist]`, `:331–335`); adiabatic zeros
   them.

**Oracle blind spot:** the hermetic gate (`pr0_oracle.py`) runs cone + adiabatic,
which never writes or reads `S_elec_flat`. The most entangled ES field — a visible
Hamiltonian input flowing from a *centroid* — is invisible to our regression gate.
Any change touching it needs QM-side validation (recorded fixture or hardware).

### 4f. ES continuation state (the stateful, path-dependent part)

ES is *path-dependent*: each call depends on the previous call's output, not just
the geometry. For `terachem_cas` (`potential/terachem_cas.py`):

- **`orbs`** (MO coefficients) seed the next SCF/CASSCF (`:84` → `c0.old` →
  `options["guess"]`/`casguess`). Written back at `:121,130`.
- **`electronic_phases`** track adiabatic-state signs, read *and* updated by
  comparing overlap signs against the previous step (`:163–170`). Determinism-
  critical (constraint #4b — NAC phases must not flip).
- **`prev_wf_positions`** (previous geometry) is needed to compute the overlap
  `S` (`:145,194`). `civecs`, `S_elec_flat`, `norbs/ncivecs` round out the state.

Mechanism today: this state lives as **named attributes on `traj`**, read back by
the same backend at the next step, and made durable by `fmsobj.to_dict`
serializing `self.__dict__` wholesale into `sim.json`. Scratch files (`c0.old`,
`CIvecs.Singlet.old`) are regenerated each step from the traj-stored arrays —
ephemeral transport to TeraChem, not the source of truth. The cone analogue is
`wf`/`prev_wf`, and `pr0_oracle.py` proves it round-trips at `max|diff|=0`.

This is preservation achieved by the *widest possible* boundary: ES state is
named, mutable, and co-serialized with the physics. Tightening it is the subject
of the companion contract doc.

---

## 5. Durability & determinism

**Two-file scheme** (`restart_output` `:895`, `restart_from_file` `:888`):

- **`sim.json` = the current object graph.** `fmsobj.write_to_file` → `to_dict`
  recurses the whole `simulation` (all `traj`s, all centroids, amplitudes,
  `traj_map`, times, SSAIMS state). Up to 3 rotating backups (`:903–919`).
- **`sim.hdf5` = append-only time-series history *and* the `*_qm` sync bus.**
  Synced from `working.hdf5` by `shutil.copy2` (`:941`). Per-step matrices
  (`S/Sdot/Sinv/H/Heff`), amplitudes, `quantum_time`, per-TBF geometries/energies.
- Restart: `read_from_file` rebuilds the graph (re-instantiating each `traj` via
  `(numdims, numstates)`, `:129–140`); `working.hdf5` is restored from the h5 so
  propagation resumes reading past geometries. HDF5 is consumed lazily on demand,
  not loaded to memory.

**Determinism disciplines** (constraint #4, load-bearing and easy to break):
(a) all randomness from one serialized PRNG (SSAIMS `ssa_seed`/`ssa_rand_calls`,
Wigner sampling — PR2 centralizes this); (b) no ES point recomputed within a run
(the store is authoritative so NAC phases can't flip — §4f); (c) no side effects
in physics except through the framework. The `eval`-replay model only works
because these hold.

---

## 6. The injection / monkeypatch layer (`import_methods.py`)

Backends and integrators are selected by copying a module's functions onto a class
at import time:

- `into_traj(x)` (`:10`) — copies every non-underscore attribute of module `x`
  onto the `traj` class. Post-PR1 it **skips `compute_elec_struct`** and instead
  wires the ES backend via the registry (`BACKEND` → real backend;
  else `LegacyMutatingBackend` around a legacy in-place `compute_elec_struct`,
  `:19–25`). Selects: the ES backend (`potential/*`) *and* the classical
  integrator (`classical_integrator/vv`).
- `into_simulation(x)` (`:27`) — wires the QM Hamiltonian
  (`qm_hamiltonian/{adiabatic,dgas}`) and QM integrator (`qm_integrator/rk2`).
- `into_hessian(x)` (`:5`) — **unchanged; still monkeypatches the raw
  `compute_elec_struct` onto `hessian`**, shadowing the inherited seam shim. This
  is the residual ES entry point that bypasses the registry (§4a).

Example `start.py` scripts choose backends by building `into_traj(...)` strings
and `exec`ing them (`examples/*/start*.py`) — a second `eval`/`exec` site,
orthogonal to the task-queue one. Also note `complexgaussian` uses `eval` on its
`positions_i="positions_qm"` string args to select which traj attribute to read
(`cg.*` functions) — this is how one code path serves both classical-time and
`*_qm` data.

---

## 7. Salvageability — assets vs. liabilities

**Assets (preserve and move, do not re-derive):**

- The **validated FMS physics**: matrix-element builders (`simulation.py:479–616`;
  `dgas.py`), `complexgaussian.py` nuclear math, spawning logic
  (`spawn_as_necessary` `:784`), the velocity-Verlet integrator, the two
  Hamiltonian variants. These are correct and hard-won.
- The **durability model**: serialize-every-cycle + HDF5 history gives
  deterministic replay/restart. `pr0_oracle.py` proves it (cone). Keep it.
- **Backends are already modular** at the right granularity
  (`potential/*` each with `compute_elec_struct`); PR1 formalized their selection.

**Liabilities (the architecture to refactor toward the skeleton):**

- **`eval` task-string scheduler** (`propagate`/`update_queue`) — the inversion of
  control that shreds the linear AIMS narrative. PR3 target.
- **`traj` as god object** — nuclear + electronic outputs + ES continuation state
  + `*_qm` snapshots + spawn state on one object, all wholesale-serialized. The ES
  boundary leaks into the physics object (§2.2, §4f).
- **HDF5 as synchronization bus** (§4c) — correctness depends on a file round-trip
  and `*_qm` shadow fields. Powerful (it *is* how multi-clock TBFs re-sync) but
  implicit and easy to break.
- **Wide ES boundary** — ES state is ~7–10 named accessors per QM backend,
  injected onto `traj`; the dynamics *can* read/modify it. Contract doc addresses
  this.
- **Injection by monkeypatch + `exec`/`eval`** in three places (task queue,
  `start.py` backend selection, `cg` attribute selectors). PR1 removed it for ES
  selection; the pattern persists elsewhere.

**Bottom line:** the physics and the durability model are worth salvaging; the
scheduler and the god-object/wide-boundary data model are the liabilities. The
staged plan (`CLAUDE.md`) keeps the former while replacing the latter behind one
seam at a time, each gated by `pr0_oracle.py`.
