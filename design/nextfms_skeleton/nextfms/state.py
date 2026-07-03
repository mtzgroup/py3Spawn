"""
Durable simulation state.

This is the ONE object that gets serialized at a checkpoint. Everything the
run needs to resume lives here; NOTHING important lives in a call stack.

Two disciplines make deterministic replay work, and both are enforced by the
*shape* of this object rather than by documentation:

  1. The PRNG state is a field of `SimState` (`rng_state`). All randomness in
     the physics tier (surface-hopping draws, Wigner sampling, stochastic
     spawn tests) is drawn from `state.rng`, which is advanced explicitly and
     serialized. A restart re-seeds to the exact bit-state, so a replayed run
     reproduces the original call sequence byte-for-byte.

  2. State is only ever serialized at a STEP BOUNDARY (top of the driver loop),
     where it is trivially consistent. The concurrent electronic-structure
     fan-out that happens *inside* a step is ephemeral: created fresh each
     step, joined at a barrier, never persisted.

A physics author edits the fields below freely. They never touch the executor,
the cache, or the checkpoint machinery -- those cannot even be named from here.
"""
from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field, replace


@dataclass
class TBF:
    """A trajectory basis function (frozen Gaussian) -- or, with one instance,
    a surface-hopping walker. Position/momentum are the phase-space center;
    `amp` is the complex expansion coefficient; `state` is the active adiabatic
    electronic state index.

    This is the nuclear-representation object. AIMS = a coupled set of these;
    FSSH = an independent swarm of these. The rest of the framework does not
    care which, so both variants share every other line of code.
    """
    id: int
    x: np.ndarray          # position(s) of the Gaussian center
    p: np.ndarray          # momentum
    gamma: float           # nuclear phase
    amp: complex           # complex expansion amplitude
    state: int             # active adiabatic electronic state
    alive: bool = True     # pruned TBFs stay in the record but stop propagating
    parent: int | None = None   # id of the TBF this was spawned/cloned from
    born_step: int = 0

    def copy(self) -> "TBF":
        return replace(self, x=self.x.copy(), p=self.p.copy())


@dataclass
class SimState:
    """The full durable state of an AIMS/AIMC/SH run."""
    step: int = 0
    time: float = 0.0
    dt: float = 10.0                       # a.u.
    mass: float = 2000.0                   # a.u. (model)
    method: str = "toy2state"              # electronic-structure method tag
    basis: list[TBF] = field(default_factory=list)
    next_id: int = 0

    # --- determinism: the PRNG lives IN the state, not in a global ----------
    rng_state: dict = field(default_factory=dict)

    # bookkeeping for analysis / stopping
    max_steps: int = 200
    history: list = field(default_factory=list)   # light per-step summaries

    # ---- PRNG plumbing (the load-bearing invisible discipline) -------------
    def seed(self, seed: int) -> None:
        g = np.random.default_rng(seed)
        self.rng_state = g.bit_generator.state

    @property
    def rng(self) -> np.random.Generator:
        """Return a Generator restored to the serialized bit-state. After
        drawing, callers MUST write the advanced state back via `commit_rng`.
        """
        g = np.random.default_rng()
        g.bit_generator.state = self.rng_state
        return g

    def commit_rng(self, g: np.random.Generator) -> None:
        self.rng_state = g.bit_generator.state

    # ---- convenience -------------------------------------------------------
    def new_id(self) -> int:
        i = self.next_id
        self.next_id += 1
        return i

    @property
    def live_basis(self) -> list[TBF]:
        return [t for t in self.basis if t.alive]

    @property
    def done(self) -> bool:
        return self.step >= self.max_steps
