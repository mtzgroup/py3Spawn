# The electronic-structure seam contract (opaque-state design)

**Purpose.** Specify a *tight and localized* interface between electronic
structure and dynamics: one narrow seam every ES code goes through, across which
ES information round-trips as an **opaque handle** the dynamics never interprets.
This is prescriptive — where PR1's seam should evolve. For what exists today, read
[`current_architecture.md`](current_architecture.md) first; this document assumes
its §4 (the three-category partition and the `*_qm` HDF5 bus).

Maps to the target skeleton `design/nextfms_skeleton/nextfms/es_provider.py`
(`ESRequest`/`ESResult`/`compute_one`), extended for the two things the toy
skeleton omits: **path-dependent continuation state** and **restart durability**.

---

## 1. The core idea: three categories, one opaque handle

ES output splits three ways (audited in `current_architecture.md` §4b):

1. **Consumed by the dynamics** — `energies`, `forces`, `timederivcoups`,
   `S_elec` (electronic overlap). Read by the integrator, spawning, and the FMS
   Hamiltonian builders. Must be **visible and typed**.
2. **ES continuation state** — `orbs`, `civecs`, `electronic_phases`,
   `prev_wf_positions`, `prev_wf`, … Read only by the ES backend itself, on the
   next step. Must be an **opaque handle** the dynamics cannot read.
3. **Recorded but not consumed** — `wf0`/`wf1` and any logged-only datasets. The
   framework records them to HDF5; the dynamics ignores them.

The contract makes these three separate channels, so that "the dynamics never
interprets or modifies ES state" is **structural, not a promise** — category 2
never passes through physics code at all.

---

## 2. Types (extending `es_provider.py`)

```python
# --- (2) opaque continuation state ---------------------------------------
class ESState:
    """Backend-private state threaded across a trajectory's ES calls. OPAQUE to
    the physics: the dynamics tier stores it and hands it back, never reads a
    field. Each backend defines its own contents:
        test_cone    -> prev_wf
        terachem_cas -> orbs, civecs, electronic_phases, prev_wf_positions,
                        norbs, ncivecs
        molcas_cas   -> + jobiph, wfn, inporbs  (FILES, not arrays -- see 5.2)
    """

# --- the cacheable identity of one ES point (frozen -> hashable key) ------
@dataclass(frozen=True)
class ESKey:
    geom_key: tuple            # rounded geometry -> exact hashable key
    method: str
    states: tuple[int, ...]
    want_coupling: Coupling    # NACV | OVERLAP | ENERGY_GAP (the coupling ladder)
    want_gradients: bool = True
    # prior_state is deliberately NOT a key field -- see §4 (determinism).
    # NOTE (pressure test, §4.1): the EXECUTOR cache is keyed by provenance
    # (basis-function label, time), NOT by this geometry key. geom_key is an
    # optional content-dedup layer, safe only where proven phase-independent.

# --- a request = identity + the history needed to compute it --------------
@dataclass
class ESRequest:
    key: ESKey
    prior_state: ESState | None = None   # this chain's previous output; None at step 0
    direction: str = "forward"           # "forward" | "backprop": two independent chains

# --- a result = the three categories, explicitly separated ----------------
@dataclass
class ESResult:
    # (1) consumed by the dynamics -- typed, first-class
    energies: np.ndarray
    forces: np.ndarray                          # (nstates, ndims); traj "forces_i"
    timederivcoups: np.ndarray | None = None
    overlaps: np.ndarray | None = None          # S_elec, flattened -- DGAS reads it
    # (2) opaque continuation state -- carried, never inspected by physics
    state: ESState | None = None
    # (3) recorded-but-not-consumed -- framework logs to HDF5, dynamics ignores
    record: dict = field(default_factory=dict)  # e.g. {"wf0": ..., "wf1": ...}
```

The backend interface gains two serialization hooks so the framework stays
**agnostic to what is inside the blob**:

```python
class ElectronicStructureBackend:
    supports: tuple[Coupling, ...] = (Coupling.ENERGY_GAP,)   # coupling ladder

    def compute_one(self, req: ESRequest) -> ESResult: ...
    def dump_state(self, s: ESState) -> object: ...   # -> json-able (arrays->lists / file bytes)
    def load_state(self, j: object) -> ESState: ...   # inverse, on restart
```

Contrast with the current `compute_one(self, traj, zbackprop)` (PR1), which hands
the backend the *whole* `traj`. Here the backend sees only an `ESRequest` — no
`traj`, no reach-back into physics state. That is what makes the seam narrow.

---

## 3. How state is fed back (the threading)

The **framework** — not the physics, not the backend — owns a store keyed by
`(trajectory, direction)`:

```
prior = es_store.get((traj_id, direction))            # None on the first step
res   = backend.compute_one(ESRequest(key, prior, direction))

apply_visible(res, traj, direction)     # energies/forces/tdc/overlaps -> dynamics (cat 1)
record_to_hdf5(res.record, traj, direction)           # wf0/wf1 -> log      (cat 3)
es_store[(traj_id, direction)] = res.state            # opaque handoff      (cat 2)
```

What this buys over today's "mutate `self`, re-read via getter":

- **Structural opacity.** `res.state` never passes through dynamics code. Category
  2 is unreadable by the physics because it is never handed to it.
- **`direction` is first-class.** The forward and back-propagation ES chains
  (`current_architecture.md` §2.2f) become two keys in one store, replacing the
  `backprop_` string prefix smeared across every accessor.
- **Centroids included.** Centroids are `traj`s that also produce ES (they supply
  `overlaps`/`timederivcoups` to off-diagonal Hamiltonian elements). They get
  their own `(centroid_id, direction)` chain in the same store.

Interaction with the `*_qm` bus (`current_architecture.md` §4c): `apply_visible`
writes the category-1 outputs where they are today (so `h5_output` → `*_qm`
reload → Hamiltonian is unchanged). The contract changes *what the backend
touches*, not how the coupled Hamiltonian is fed. That keeps the change local.

---

## 4. Determinism — why `prior_state` is not in the cache key

The identity of an ES point is `ESKey` (geometry + method + states + want).
Idempotency-by-geometry means each point is computed **exactly once per run**
(constraint #4b: no ES point recomputed). The phase alignment baked in at
first-compute — which depends on `prior_state` — is therefore frozen in the cached
result; on replay/restart you read the cached result and **the phase cannot flip**,
precisely because you never recompute with a different `prior_state`.

So `prior_state` is a *computation input* but *not* part of the hashable identity:

- If it were in the key, every step's geometry would look novel → the cache is
  defeated and the "compute once" guarantee (and phase stability) is lost.
- Keeping the key geometry-only makes replay both cheap and phase-stable.

This is the one place the skeleton's "frozen `ESRequest` is the cache key" needs
splitting: `ESKey` (frozen, hashable, the cache identity) vs `ESRequest`
(`ESKey` + `prior_state` + `direction`, the full call). The executor caches on
`ESKey`.

Path-dependence is real and unavoidable: `electronic_phases` is resolved by
comparing against the *immediately previous* step
(`terachem_cas.py:166–172`). The opaque handle is not merely an optimization
(orbital guess) — it carries correctness-critical, history-dependent state that
**cannot be reconstructed from a geometry**. That is why it must round-trip across
the seam rather than be recomputed behind it.

### 4.1 Cache key = provenance `(label, time)`, not geometry (pressure-test result)

Pressure-testing the executor cache against the *actual* gather
(`get_all_qm_data_at_time_from_h5`, `traj.py:987`) showed the code addresses ES
results by **(basis-function label, time)** — an exact-match row lookup in the
TBF's own HDF5 group (`:991–1002`) — and *never* by geometry. Consequences for the
key:

- **Do not key the executor cache by geometry.** `geom_key` (above, and the
  skeleton's `from_tbf`) is unsafe as the cache identity: at a spawn the new TBF's
  forward and back-propagation chains compute ES at the *same* spawn geometry with
  *separate* continuation state (`init_spawn_traj` seeds `backprop_*` state
  independently, `traj.py:474–504`). Geometry-keying would collide them.
- **Key by `(label, time)`.** The executor cache then *is* the `working.hdf5`
  group layout held in memory — minus the file round-trip and minus a latent bug
  (the gather silently returns the *last* row on a missed time, `:999–1008`;
  a cache miss is instead explicit). Direction is subsumed: forward and backprop
  write disjoint time ranges of one `(label, time)` series.

`geom_key` survives only as an *optional* later content-dedup layer, valid where
one can prove no two `(label, direction)` chains need different results at a shared
geometry — which the path-dependent phase generally forbids. So `ESKey` is demoted
from "the cache key" to "an optional dedup hint."

---

## 5. Serialization & durability

The `es_store` is the only new durable object. The framework serializes each live
`(traj, direction)` entry by calling `backend.dump_state` → one opaque nested blob
per chain → into `sim.json`/`sim.hdf5`; rehydrates via `load_state`. This
**replaces** today's mechanism (N named `orbs`/`civecs`/`phases` attributes on
`traj`, serialized wholesale by `fmsobj.to_dict`) with one opaque field the physics
object exposes no accessors for.

### 5.1 `S_elec` is visible output, not opaque state (resolved)

Audit result (`current_architecture.md` §4e): `S_elec_flat` is **written only** by
the backend and **read by `dgas.build_Sdot_elec_DGAS`** (`dgas.py:170`) to build
the DGAS derivative coupling → `Heff`. The backend never reads its own prior
`S_elec_flat`. Therefore it is category (1) — it belongs in `ESResult.overlaps`,
**not** in `ESState`. (It is also HDF5-logged, category 3, but that is orthogonal.)

### 5.2 MOLCAS state is files, not arrays (design wrinkle)

`molcas_cas` continuation state includes `jobiph`, `wfn`, `inporbs` — Molcas
*files*, not in-memory arrays. The opaque-blob abstraction handles this gracefully
(the framework does not care what is inside), but `dump_state` for MOLCAS must
capture file **contents** (or a content-addressed copy), not scratch-dir **paths**,
or restart-after-move loses the state. This is the argument for the
`dump_state`/`load_state` codec hooks over generic numpy-JSON: only the backend
knows its state is file-shaped.

### 5.3 Oracle constraint (do not regress)

`pr0_oracle.py` compares HDF5 datasets at `max|diff|=0`. Two consequences:

- The category-3 `record` tap **must** keep `wf0`/`wf1` (and any QM-logged
  datasets) byte-identical — "opaque to the dynamics" must not mean "dropped from
  the log."
- The gate is **blind to `S_elec_flat`** (cone + adiabatic never touch it;
  `current_architecture.md` §4e). Migrating the QM backends to this contract
  therefore needs its own validation — a recorded-fixture oracle or a live
  TeraChem/Molcas run — because the cone oracle cannot see the most entangled
  field.

---

## 6. Migration mapping (per backend, later PRs)

| Today (named, on `traj`) | Under the contract |
|---|---|
| `get/set_[bp_]orbs`, `…civecs`, `…electronic_phases`, `…prev_wf_positions`, `…norbs/ncivecs` | fields inside `ESState`; gone from `traj` |
| `get_[bp_]wf` read as previous; `set_[bp_]wf` | `prev_wf` in `ESState`; `wf0/wf1` emitted in `res.record` |
| `set_[bp_]S_elec_flat` | `ESResult.overlaps` (visible) |
| `compute_elec_struct` reads `self.get_orbs()` … | `compute_one` reads `req.prior_state` |
| `compute_elec_struct` writes `self.set_*` | `compute_one` returns `ESResult(state=…, overlaps=…, …)` |
| `into_traj` injects ~7–10 ES-state accessors onto `traj` | none injected — state lives in the blob |

Sequencing (each gated by `pr0_oracle.py` where it can see the change):

1. **Introduce `ESRequest`/`ESState`, narrow `compute_one(req)`** — migrate the
   cone backend (`TestConeBackend`) off `compute_one(self, traj, zbackprop)` to
   `compute_one(req)` returning `state=ESState(prev_wf=…)` and
   `record={"wf0":…,"wf1":…}`. Fully oracle-gated. This proves the opaque handle
   end-to-end on the hermetic path.
2. **Seal the Hessian bypass** — teach `into_hessian` to wire `_es_backend` like
   `into_traj` (hessian *is* a `traj`), so all ES flows through the one seam.
3. **Migrate the QM backends** off `LegacyMutatingBackend` to native
   `compute_one` + `dump_state`/`load_state`. Not cone-oracle-gated (§5.3) —
   needs a QM validation strategy first.

---

## 7. What "tight and localized" looks like when done

- **One seam.** Every ES code — cone, TeraChem, Molcas, a future ML surrogate —
  reaches the dynamics only through `compute_one(ESRequest) -> ESResult`.
- **Narrow data.** The backend sees an `ESRequest` (geometry, states, want, prior
  handle), never the `traj`. The dynamics sees `ESResult` category-1 fields, never
  the opaque `state`.
- **State preserved, structurally.** Continuation state round-trips as an opaque
  handle the framework threads and serializes; the physics cannot read or corrupt
  it. Path-dependence (phase tracking) is honored by threading, not recompute.
- **Executor-ready.** Because `compute_one` takes a value (`ESRequest`) and
  returns a value (`ESResult`), and `ESKey` is a cache key, the call can later be
  fanned out, cached, retried, and made crash-durable behind
  `executor.evaluate(requests)` (PR3/PR4) with **zero change to the physics or the
  backends** — the whole point of the seam.
