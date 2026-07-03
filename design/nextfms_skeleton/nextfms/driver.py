"""
The driver: the serial-reading top level.

This is the whole point of the design. Read `run()` and `advance()` below: they
are a straight-line narrative of an AIMS/AIMC/SH step -- declare what ES is
needed, get it, propagate centers, propagate amplitudes, adapt the basis,
checkpoint. No async, no callback, no inversion is visible. The concurrency and
crash-resilience are entirely inside `executor.evaluate(...)` (the hard core)
and the PRNG discipline inside `state` (the durable object).

Resilience model here is pure Pattern A (matching FMS v2): checkpoint the
physics state at each step boundary; on restart, re-enter `run()` from the last
checkpoint and let the executor's persistent cache absorb every ES call that
already completed. Nothing reconnects to running jobs because -- as in v2 --
there is no in-flight job registry to reconnect to. (Upgrading to reconnect is a
choice, localized entirely to the executor; the driver and physics never change.)
"""
from __future__ import annotations

import pickle
from pathlib import Path

from .state import SimState
from .es_provider import Coupling
from . import physics


class Checkpoint:
    """Serialize the durable state at step boundaries. Atomic write so a crash
    mid-checkpoint leaves the previous good checkpoint intact."""
    def __init__(self, path: str):
        self.path = Path(path)

    def save(self, state: SimState) -> None:
        tmp = self.path.with_suffix(".tmp")
        with open(tmp, "wb") as fh:
            pickle.dump(state, fh)
        tmp.replace(self.path)

    def load(self) -> SimState | None:
        if self.path.exists():
            with open(self.path, "rb") as fh:
                return pickle.load(fh)
        return None


def advance(state: SimState, executor, rule, coupling: Coupling) -> SimState:
    """One step. Reads forward, top to bottom. THIS is what a student reads and
    edits. Every line is pure physics except the one `executor.evaluate` call,
    which looks like a blocking function returning a dict."""
    reqs = physics.es_requests(state, coupling)              # what we need
    req_of = reqs                                            # id -> request

    # reference-state work (active surfaces) before speculative work
    def prio(req):
        return 0

    results = executor.evaluate(list(reqs.values()), priority=prio)  # blocks
    results = {req: results[req] for req in reqs.values()}   # key by request

    physics.propagate_centers(state, results, req_of)        # pure
    physics.propagate_amplitudes(state, results, req_of)     # pure (FMS H/S)
    events = rule.apply(state, results, req_of)              # spawn/clone/hop

    state.step += 1
    state.time += state.dt
    state.history.append({
        "step": state.step, "time": round(state.time, 2),
        "n_live": len(state.live_basis), "events": events,
    })
    return state


def run(state: SimState, executor, rule, checkpoint: Checkpoint,
        coupling: Coupling = Coupling.NACV, log=print) -> SimState:
    """The top-level loop. A first-year student can read this."""
    while not state.done:
        state = advance(state, executor, rule, coupling)
        checkpoint.save(state)                               # the ONLY durable act
        if log and (state.step % 20 == 0 or state.history[-1]["events"]):
            h = state.history[-1]
            log(f"  step {h['step']:4d} t={h['time']:7.1f} "
                f"n_live={h['n_live']:2d} {h['events'] or ''}")
    return state
