# PySpawn — target architecture (redesign proposal)

**Purpose.** A concrete target to refactor *toward*: the physics reads serially
top-to-bottom, and all concurrency/resilience/durability is sealed behind one
seam. This is the "where we're going" companion to
[`current_architecture.md`](current_architecture.md) (what exists) and
[`es_seam_contract.md`](es_seam_contract.md) (the ES seam in detail). It refines
the staged plan in `CLAUDE.md` with what the code-mapping revealed — chiefly that
`working.hdf5` is a synchronization bus, not just a restart store.

Prototype to anchor against: `design/nextfms_skeleton/` (`executor.py`,
`es_provider.py`, `demo_replay.py`).

---

## 1. Thesis: invert the inversion

The current design inverts control — it serializes a god-object and `eval`s task
strings — to buy crash-durability via deterministic replay
(`current_architecture.md` §1, §3). The durability is the asset worth keeping; the
inversion is the cost worth removing. This redesign **keeps the durability and
removes the inversion** by making durability a property of *one sealed component*
(the executor) rather than the shape of the whole program. The AIMS step goes back
to being a linear narrative a student can read and edit without being able to
break the resilience.

---

## 2. Four layers, one seam

### Layer 1 — Pure physics kernels (values in, values out)

The validated assets — ES-at-a-geometry, the velocity-Verlet step, the FMS matrix
elements (`build_S/Sdot/V/tau/T`), the amplitude RK2, `complexgaussian` — become
**pure functions**: no `self`-mutation, no I/O, no scheduler awareness.
`build_V(states_at_t) -> V`, not `build_V(self)` reading `*_qm` off the object.

This is a *move, not a rewrite* (constraint #5): the arithmetic is preserved
verbatim; only how inputs arrive changes — from "read shadow fields off `self`" to
"receive a value." The two Hamiltonian variants (`adiabatic`, `dgas`) become
registry-selected **coupling-model strategies**, each implementing
`build_heff(tbf_states, centroid_states) -> Heff` — the same plug-in pattern as
the ES backend, so SSAIMS/surface-hopping variants attach without touching the
driver.

### Layer 2 — Typed state, decomposed from the god-object

`traj` today fuses nuclear state + ES outputs + ES continuation state + `*_qm`
snapshots + spawn flags, all wholesale-serialized (`current_architecture.md` §2.2).
Decompose into small explicit records:

- `NuclearState` — positions, momenta, widths, masses, clock.
- `ESResult` + opaque `ESState` handle — the [contract](es_seam_contract.md).
- `QuantumState` — amplitude vector `C`, basis membership, `traj_map`.

A "trajectory" becomes a thin record plus a reference to its ES chain — not ~70
accessors — and the ES boundary stops leaking into the physics object.

### Layer 3 — The driver: one AIMS step, read serially

The logic currently shredded across `update_queue`/`eval` becomes a function you
can read:

```python
def advance(state: SimState, dt) -> SimState:
    t = state.quantum_time + dt
    # (a) advance every live TBF to t — classical + ES
    es   = executor.evaluate([es_request(tbf, t) for tbf in state.live_tbfs(t)])
    tbfs = [vv_step(tbf, es[tbf.id], dt) for tbf in state.live_tbfs(t)]
    # (b) spawn where coupling peaked
    tbfs += spawn_where_needed(tbfs)
    # (c) build the coupled Hamiltonian from TBF states AT t (no file round-trip)
    cents = centroids_for(tbfs)
    ces   = executor.evaluate([es_request(c, t) for c in cents])
    Heff  = coupling_model.build_heff(tbfs, cents, es, ces)   # adiabatic | dgas
    # (d) propagate amplitudes over the completed interval
    C = rk2(state.C, Heff, dt)
    return SimState(tbfs, C, t)
```

No `eval`, no task strings, no `backprop_` prefix soup — back-propagation is just a
second ES chain the executor fills, and `live_tbfs(t)` asks for what it needs.

*Pressure-test refinement (§8.1): the pseudocode above is simplified. The real
driver runs **two cursors** — an ES-frontier cursor ahead of a trailing quantum
cursor that releases only when no pending spawn threatens the interval. This
spawn-causality barrier is what `max_info_time` encodes today; it is essential
logic, not decoration.*

### Layer 4 — The executor seam (all the hard stuff, sealed)

```python
results = executor.evaluate(requests)   # looks synchronous; blocks; keyed by request
```

Underneath, and **nowhere else in the codebase**, live concurrency, retries,
backpressure, HPC scheduling (Parsl — the skeleton's `ParslExecutor`), and
durability. Physics files contain zero `async`/thread/executor references — the
two-tier split `CLAUDE.md` asks for, made structural.

---

## 3. The linchpin: durability as a memoized cache — which also retires the `*_qm` bus

Two mechanisms in the current code are secretly the same operation — "recover a
computed ES result by identity":

- **Durability** = serialize the god-object + replay `eval` task strings.
- **Synchronization** = TBFs write ES to `working.hdf5`; the Hamiltonian reads it
  *back* into `*_qm` fields at a common quantum time (`current_architecture.md`
  §4c).

Collapse both into **one durable, content-addressed result cache** behind the
executor:

- **Durability becomes free replay.** `advance` is a pure `SimState -> SimState`.
  On restart, re-run it from `t₀`; every `es_request` is idempotent and hits the
  cache, so the narrative replays identically to the crash point, then continues
  live. You stop serializing a call stack; you memoize a pure function. (This is
  what `demo_replay.py` demonstrates.)
- **The `*_qm` bus disappears.** "Gather all TBFs at quantum time `t`" becomes
  "read the cached ES results for those requests" — keyed by **`(basis-function
  label, time)`** (pressure test §8.1), which is exactly how `working.hdf5`
  addresses ES data today, minus the file round-trip and the `setattr` shadow
  fields. `working.hdf5` reverts to output/analysis history (and, optionally, the
  cache's durable backing) and stops being an in-band coupling mechanism.

One mechanism replaces two implicit ones. **This is the single biggest structural
win, and the load-bearing claim of the whole redesign — pressure-tested in §6.**

---

## 4. Determinism requirements (make the disciplines unrepresentable)

Replay-as-durability is only valid if the physics is a deterministic function of
durable state (constraint #4). The redesign should make the disciplines
*structural*, not merely encouraged:

- **PRNG** reachable only through serialized state — a physics kernel *cannot*
  call a global `np.random` because it holds no such reference (PR2).
- **No ES recompute** — the cache is authoritative; `ESKey` identity guarantees a
  point is computed once, so NAC phases can't flip (`es_seam_contract.md` §4).
- **No side effects in physics** — kernels are pure; only the framework writes
  files/logs. The category-3 `record` tap (contract §1) is how logged-but-unused
  quantities (e.g. `wf0/wf1`) still reach HDF5 without physics doing I/O.

---

## 5. Migration (staged, each gated by `pr0_oracle.py` at max|diff|=0)

The staged plan lands here; the mapping adds one sub-step:

- **PR1 (done)** — ES selection via registry; the seam exists.
- **PR1b** — narrow `compute_one(ESRequest)`, opaque `ESState`; seal the Hessian
  bypass (`current_architecture.md` §6). Layer 2 for ES.
- **PR2** — one serialized PRNG. Precondition for trusting replay.
- **PR3** — replace `eval`/`update_queue` with serial `advance` over a threaded
  `executor.evaluate`. **New sub-step the mapping revealed:** make the matrix
  builders take explicit TBF states instead of `*_qm`, retiring the HDF5 sync bus
  in the same move (§3). This is what collapses durability and synchronization
  into the cache.
- **PR4** — back the executor with Parsl (HPC + durable app-cache).

---

## 6. Risks & open questions (to be pressure-tested)

- **Retiring the `*_qm` bus into the cache** (§3) — the load-bearing claim.
  **Pressure-tested: it holds** (see §8). The cache must be keyed by provenance
  `(basis-function label, time)` — matching how `get_all_qm_data_at_time_from_h5`
  actually addresses ES data — **not** by geometry; and the driver must run the ES
  frontier ahead of a trailing, spawn-gated quantum cursor. Details and the two
  residual open items are in §8.
- **Replay demands strict determinism** — any leaked `np.random`/side effect
  silently diverges replay. Mitigation: make it unrepresentable (§4).
- **QM backends can't be oracle-validated here** (`current_architecture.md` §4e).
  The DGAS/`S_elec` path needs a recorded-fixture or hardware oracle before
  migration.
- **Molcas file-state** (contract §5.2) resists the clean opaque-array model; the
  `dump_state`/`load_state` codecs handle it but it is the ugliest corner.

---

## 7. What "done" looks like

- **One seam.** Every ES code reaches the dynamics only through
  `compute_one(ESRequest) -> ESResult`; the driver reaches all expensive work only
  through `executor.evaluate`.
- **Readable physics.** `advance` is a serial narrative; no `eval`, no `*_qm`
  shadow fields, no async in physics files.
- **Durability for free.** Crash/restart replays a pure function against a durable
  cache — no hand-rolled scheduler, no god-object serialization.
- **Executor-ready.** Value-in/value-out kernels + a content-addressed cache mean
  the same code runs single-threaded, threaded, or on a Parsl cluster with zero
  change to physics or backends.

---

## 8. Pressure-test log

### 8.1 "Retire the `*_qm` bus into the executor cache" (§3) — HOLDS, with three corrections

Tested against the real gather (`traj.py:987`), the `max_info_time` barrier
(`simulation.py:364–413`), and spawn seeding (`traj.py:474–504`):

1. **Cache key = `(label, time)`, not geometry.** The gather is exact-match row
   lookup in each TBF's HDF5 group by label+time (`:991–1002`); geometry is never
   the address. Geometry-keying would collide the forward/backprop chains that
   compute ES at the same spawn geometry with separate state. The faithful cache
   *is* the HDF5 group layout in memory — minus the file, minus the silent
   `ipoint = -1 → last row` fallback (`:999–1008`). (Refines
   [`es_seam_contract.md`](es_seam_contract.md) §4.1.)
2. **The driver is two cursors, not a lockstep loop.** `max_info_time` holds the
   quantum tier 1–2 steps behind the ES frontier and freezes it before a pending
   spawn (`:377–379`) or a backprop `mintime` (`:385–387`). The serial driver must
   carry this spawn-causality barrier explicitly (§Layer 3).
3. **Basis membership + retroactive back-fill are relocated, not removed.**
   `live_tbfs(t)` reproduces active-TBF membership; the executor must accept
   requests at *past* times — a spawn emits ES for the new TBF's back-propagation
   and its new centroids at τ ≤ t_spawn, landing as past-`(label, time)` cache
   entries, exactly as backprop rows land in HDF5 today.

**Bonus:** the redesign removes the silent last-row fallback — a cache miss is
explicit, where the HDF5 gather returns a wrong row.

### 8.2 Residual open items

- **Boundary-row rule.** Whether the forward and backprop first steps both emit a
  row at exactly `firsttime` (a possible duplicate) is not fully pinned; the
  `if not zbackprop` output guards (`vv.py:31`) suggest not. The cache needs a
  defined boundary rule that the HDF5 layout currently encodes implicitly. Nail
  before PR3.
- **QM validation gap.** The cone oracle cannot see `S_elec`/DGAS/QM continuation
  state (§6); QM-path correctness needs a record/replay fixture captured from one
  real TeraChem run.
