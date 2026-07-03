"""
The physics tier (student-approachable).

Every function here is PURE and reads top-to-bottom. There is no `async`, no
future, no callback, no cache, no reference to the executor's internals. A
physics author adds a spawning criterion, a cloning rule, a hopping rule, or a
new integrator by editing THIS file, and cannot break the resilience machinery
because the code here cannot even name it.

The unifying abstraction is `BasisAdaptationRule`: spawn (AIMS), clone (AIMC),
and hop (FSSH) are all "a criterion over the state + an action on the basis."
Swapping the rule swaps the method; everything else is shared.
"""
from __future__ import annotations

import numpy as np
from .state import SimState, TBF
from .es_provider import ESRequest, Coupling


# ---------------------------------------------------------------------------
# 1. Declare what electronic structure the current basis needs (pure).
# ---------------------------------------------------------------------------
def es_requests(state: SimState, coupling: Coupling) -> dict[int, ESRequest]:
    """One request per live TBF. When the basis grows (spawn/clone) or shrinks
    (prune), this dict changes automatically -- the dynamic task set is a
    non-problem because we re-declare it every step."""
    return {t.id: ESRequest.from_tbf(t, state.method, states=(0, 1),
                                     coupling=coupling)
            for t in state.live_basis}


# ---------------------------------------------------------------------------
# 2. Classical propagation of the Gaussian centers (pure, cheap, serial).
# ---------------------------------------------------------------------------
def propagate_centers(state: SimState, results, req_of) -> None:
    """Velocity-Verlet on each live TBF's phase-space center. Half-kick uses
    the gradient on the TBF's active state."""
    dt, m = state.dt, state.mass
    for t in state.live_basis:
        g = results[req_of[t.id]].gradients[t.state]
        t.p = t.p - 0.5 * dt * g
        t.x = t.x + dt * t.p / m
        # (a real integrator re-evaluates the force at x(t+dt) for the second
        #  half-kick; omitted here to keep the skeleton to one ES pass/step)
        e = results[req_of[t.id]].energies[t.state]
        t.gamma += -e * dt


# ---------------------------------------------------------------------------
# 3. Coupled amplitude propagation (pure, cheap). AIMS builds H/S here.
# ---------------------------------------------------------------------------
def propagate_amplitudes(state: SimState, results, req_of) -> None:
    """Placeholder for the FMS H/S saddle-point matrix elements + TDSE solve in
    the non-orthogonal Gaussian basis. In the transplant this body is the
    VALIDATED PySpawn code, moved across unchanged. Left as a norm-preserving
    stub so the skeleton runs."""
    live = state.live_basis
    if not live:
        return
    norm = np.sqrt(sum(abs(t.amp) ** 2 for t in live))
    if norm > 0:
        for t in live:
            t.amp = t.amp / norm


# ---------------------------------------------------------------------------
# 4. The basis-adaptation abstraction: spawn / clone / hop share this shape.
# ---------------------------------------------------------------------------
class BasisAdaptationRule:
    """A criterion + an action on the basis. Subclass to define a method."""
    def apply(self, state: SimState, results, req_of) -> list[str]:
        raise NotImplementedError


class SpawnRule(BasisAdaptationRule):
    """AIMS: when coupling to another state is strong, spawn a child TBF on
    that state. This is the exact kind of pure predicate a student writes."""
    def __init__(self, threshold=0.05):
        self.threshold = threshold

    def apply(self, state, results, req_of):
        events = []
        for t in list(state.live_basis):
            res = results[req_of[t.id]]
            other = 1 - t.state
            coup = res.nacv.get((t.state, other))
            strength = float(abs(coup[0])) if coup is not None else 0.0
            if strength > self.threshold and t.state == 0:
                child = t.copy()
                child.id = state.new_id()
                child.state = other
                child.amp = 0.0 + 0j       # newborn starts with zero amplitude
                child.parent = t.id
                child.born_step = state.step
                state.basis.append(child)
                events.append(f"spawn {child.id}<-{t.id} (|d|={strength:.3f})")
        return events


class HopRule(BasisAdaptationRule):
    """FSSH: stochastic hop of a single walker's active state. RNG is drawn
    from state.rng and committed back -- this is the line whose determinism the
    whole replay model depends on."""
    def __init__(self, rate=0.02):
        self.rate = rate

    def apply(self, state, results, req_of):
        events = []
        g = state.rng                      # restore serialized bit-state
        for t in state.live_basis:
            res = results[req_of[t.id]]
            other = 1 - t.state
            coup = res.nacv.get((t.state, other))
            strength = float(abs(coup[0])) if coup is not None else 0.0
            if g.random() < self.rate * strength * 20:
                t.state = other
                events.append(f"hop {t.id} -> state {other}")
        state.commit_rng(g)                # advance & persist the PRNG
        return events
