"""
End-to-end demonstration of the resilience story.

Scenario:
  1. Start an AIMS run (SpawnRule). Let it run N1 steps, then SIMULATE A CRASH
     (hard stop) -- but the checkpoint and the ES cache are on disk.
  2. "Restart the job": construct a fresh executor + load the checkpoint, and
     call run() again. It re-enters at the checkpointed step. Every ES call it
     would repeat is a CACHE HIT (served from disk, not recomputed); only genuinely
     new work dispatches.
  3. Show that the restarted trajectory is byte-for-byte identical to an
     uninterrupted reference run -- proving deterministic replay.

The point: run()/advance() never knew a crash happened. No stack was serialized.
"""
import shutil, pickle, hashlib
from pathlib import Path
from nextfms import SimState, TBF, ThreadedExecutor, ToyAvoidedCrossing, Coupling
from nextfms.driver import run, advance, Checkpoint
from nextfms.physics import SpawnRule
import numpy as np


def fresh_state(seed=1234):
    s = SimState(max_steps=60, dt=10.0, method="toy2state")
    s.seed(seed)
    t = TBF(id=s.new_id(), x=np.array([-3.0]), p=np.array([25.0]),
            gamma=0.0, amp=1.0 + 0j, state=0)
    s.basis.append(t)
    return s


def traj_fingerprint(state):
    """A hash of the full trajectory record -> lets us assert exact identity."""
    payload = [(h["step"], h["n_live"], tuple(h["events"])) for h in state.history]
    payload += [(t.id, round(float(t.x[0]), 9), round(float(t.p[0]), 9),
                 t.state, t.parent) for t in state.basis]
    return hashlib.sha256(pickle.dumps(payload)).hexdigest()[:16]


def run_dir(name):
    d = Path(name)
    if d.exists():
        shutil.rmtree(d)
    d.mkdir(parents=True)
    return d


# --- (A) Reference: one uninterrupted run --------------------------------
print("=" * 68)
print("(A) REFERENCE RUN (uninterrupted)")
d = run_dir("run_ref")
be = ToyAvoidedCrossing()
ex = ThreadedExecutor(be, cache_dir=str(d / "cache"), max_in_flight=8)
st = fresh_state()
st = run(st, ex, SpawnRule(threshold=0.02), Checkpoint(str(d / "ckpt.pkl")),
         coupling=Coupling.NACV)
ex.shutdown()
ref_fp = traj_fingerprint(st)
ref_cache_files = len(list((d / "cache").glob("*.pkl")))
print(f"    finished at step {st.step}, {len(st.basis)} TBFs total")
print(f"    trajectory fingerprint = {ref_fp}")
print(f"    ES evaluations cached  = {ref_cache_files}")

# --- (B) Crashing run: stop hard at step 25 ------------------------------
print("=" * 68)
print("(B) CRASHING RUN -- hard stop at step 25, then restart")
d2 = run_dir("run_crash")
ck = Checkpoint(str(d2 / "ckpt.pkl"))

# phase 1: run 25 steps then 'crash' (just stop calling advance)
be = ToyAvoidedCrossing()
ex = ThreadedExecutor(be, cache_dir=str(d2 / "cache"), max_in_flight=8)
st = fresh_state()
rule = SpawnRule(threshold=0.02)
for _ in range(25):
    st = advance(st, ex, rule, Coupling.NACV)
    ck.save(st)
ex.shutdown()
cache_after_crash = len(list((d2 / "cache").glob("*.pkl")))
print(f"    [crash] last checkpoint at step {st.step}; "
      f"{cache_after_crash} ES results on disk")
del st, ex, be   # simulate process death: in-memory state is GONE

# phase 2: 'restart the job' -- new process, reload checkpoint, resume
be = ToyAvoidedCrossing()
ex = ThreadedExecutor(be, cache_dir=str(d2 / "cache"), max_in_flight=8)  # same cache dir

# instrument the backend to COUNT real computations after restart
real_computes = {"n": 0}
_orig = be.compute_one
def counting_compute(req):
    real_computes["n"] += 1
    return _orig(req)
be.compute_one = counting_compute

st = ck.load()                      # <-- resume from serialized physics state
print(f"    [restart] reloaded checkpoint at step {st.step}")
st = run(st, ex, SpawnRule(threshold=0.02), ck, coupling=Coupling.NACV)
ex.shutdown()
restart_fp = traj_fingerprint(st)
print(f"    finished at step {st.step}, {len(st.basis)} TBFs total")
print(f"    trajectory fingerprint = {restart_fp}")
print(f"    NEW ES computes after restart (steps 26-60) = {real_computes['n']}")

# --- (C) The assertions that make it a proof -----------------------------
print("=" * 68)
print("(C) VERIFICATION")
print(f"    reference fingerprint  = {ref_fp}")
print(f"    restarted fingerprint  = {restart_fp}")
print(f"    identical trajectory   = {ref_fp == restart_fp}")
assert ref_fp == restart_fp, "replay diverged -- determinism broken!"
print("    PASS: crash+restart reproduced the uninterrupted trajectory exactly.")
print(f"    (restart recomputed only the {real_computes['n']} steps that never "
      f"finished; steps 1-25 were served from the persistent cache.)")
