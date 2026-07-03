# nextfms — target-state skeleton

A runnable template for the "restore FMS v2's factoring, rent the resilience"
architecture. It is **not** a physics engine — the physics is stubbed with a toy
avoided-crossing model. It exists to show the *shape* PySpawn should be
refactored **toward**, and to hand a student a concrete picture of the
pure-physics tier.

## The one idea

Two tiers, one narrow seam. The inversion of control that makes the code hard to
reason about is **sealed at the floor**; everything above it reads serially.

```
        ┌─────────────────────────────────────────────┐
        │  driver.run() / advance()   ← reads FORWARD   │   PHYSICS TIER
        │  physics.py  (integrator, spawn/clone/hop)    │   (students edit;
        │  toy_backend.py  (ES adapter: compute_one)    │    pure functions;
        │  state.py   (durable SimState + PRNG)         │    no async in sight)
        └───────────────┬─────────────────────────────┘
                        │  the seam: two calls only
                        │    down: backend.compute_one(request)
                        │    up:   executor.evaluate(requests) -> {req: result}
        ┌───────────────┴─────────────────────────────┐
        │  executor.py  ThreadedExecutor / ParslExecutor │  HARD CORE
        │  concurrency · persistent cache · backpressure │  (Ben edits; sealed;
        │  priority · crash-recovery via replay          │   all inversion here)
        └───────────────────────────────────────────────┘
```

## What the two demos prove

**`demo_replay.py` — resilience via deterministic replay (Pattern A).**
Runs AIMS, hard-stops at step 25 (simulated crash, in-memory state discarded),
then restarts in a fresh executor from the checkpoint. The restarted trajectory
fingerprint is **identical** to an uninterrupted reference run. Steps 1–25 are
served from the persistent ES cache; only unfinished work recomputes. No
coroutine stack is ever serialized — only `SimState`, only at step boundaries.

**`demo_methods.py` — shared machinery + stochastic determinism.**
The *same* driver/state/executor/seam run AIMS (`SpawnRule`) and FSSH
(`HopRule`) with only the adaptation rule swapped. The stochastic FSSH run
replays exactly under the same seed because the PRNG lives in `SimState`, not in
a global — the discipline the seam must **enforce**.

## Run it

```bash
PYTHONPATH=. python demo_replay.py
PYTHONPATH=. python demo_methods.py
```

## Map to the fork plan

| Skeleton piece | PySpawn transplant action |
|---|---|
| `es_provider.py` (`ESRequest`/`ESResult`/`compute_one`) | **PR 1**: extract this seam from PySpawn as a pure refactor; validate against a known trajectory. |
| `SimState.rng` / `commit_rng` | **PR 2**: route all randomness through one serialized PRNG. Precondition for replay. |
| `driver.py` + `executor.py` | **PR 3**: replace PySpawn's task-list driver with this replay loop + a cached executor. |
| `propagate_amplitudes` stub | Drop in PySpawn's **validated** H/S saddle-point matrix elements — unchanged. |
| `ParslExecutor` docstring | Swap `ThreadedExecutor` → Parsl for HPC scheduling + durable app-cache. Physics tier unchanged. |

## What is deliberately NOT here

- Real electronic structure (toy 2-state model instead).
- The full FMS H/S matrix elements (norm-preserving stub).
- AIMC cloning and trivial-crossing/state-following (the `BasisAdaptationRule`
  base class is where cloning slots in; state-tracking metadata has a field in
  `ESResult.phase` waiting to be used).
- A real Parsl config (sketched in the `ParslExecutor` docstring).

These are the places the real work goes — but the *architecture* around them is
complete and runs.
