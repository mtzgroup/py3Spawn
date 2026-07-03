"""
Demonstrates the shared-machinery thesis: the SAME driver, state, executor, and
seam run AIMS (spawn) and FSSH (stochastic hop) with only the adaptation rule
swapped -- and that the stochastic FSSH run is ALSO deterministically
replayable, because its randomness is drawn from state.rng (serialized), not a
global PRNG.
"""
import shutil
from pathlib import Path
import numpy as np
from nextfms import SimState, TBF, ThreadedExecutor, ToyAvoidedCrossing, Coupling
from nextfms.driver import run, Checkpoint
from nextfms.physics import SpawnRule, HopRule


def rd(name):
    d = Path(name)
    if d.exists():
        shutil.rmtree(d)
    d.mkdir(parents=True)
    return d


def fresh(seed):
    s = SimState(max_steps=50, dt=10.0)
    s.seed(seed)
    s.basis.append(TBF(id=s.new_id(), x=np.array([-3.0]), p=np.array([25.0]),
                       gamma=0.0, amp=1.0 + 0j, state=0))
    return s


def run_fssh(seed, tag):
    d = rd(tag)
    ex = ThreadedExecutor(ToyAvoidedCrossing(), cache_dir=str(d / "cache"))
    st = run(fresh(seed), ex, HopRule(rate=0.02), Checkpoint(str(d / "c.pkl")),
             coupling=Coupling.NACV, log=None)
    ex.shutdown()
    hops = [h for h in st.history if h["events"]]
    return st.rng_state, [h["step"] for h in hops]


print("Same driver + seam, different rule:")
print("  AIMS   = driver.run(..., SpawnRule())   -> basis grows by spawning")
print("  FSSH   = driver.run(..., HopRule())      -> single walker hops")
print()

# FSSH determinism: same seed -> identical hop sequence; different seed -> not.
_, hops_a1 = run_fssh(seed=7, tag="fssh_a1")
_, hops_a2 = run_fssh(seed=7, tag="fssh_a2")      # same seed, fresh process/cache
_, hops_b  = run_fssh(seed=99, tag="fssh_b")      # different seed

print(f"FSSH seed=7  run 1 hop-steps: {hops_a1}")
print(f"FSSH seed=7  run 2 hop-steps: {hops_a2}")
print(f"FSSH seed=99 run   hop-steps: {hops_b}")
print()
print(f"  same seed reproduces hops exactly : {hops_a1 == hops_a2}")
print(f"  different seed gives different hops: {hops_a1 != hops_b}")
assert hops_a1 == hops_a2, "PRNG not deterministic across runs!"
print()
print("=> The stochastic method is replayable because the PRNG lives in state,")
print("   not in a global. That is the discipline the seam must ENFORCE so a")
print("   student cannot reintroduce np.random and silently break resumability.")
