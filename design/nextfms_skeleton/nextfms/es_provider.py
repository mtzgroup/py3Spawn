"""
The electronic-structure seam.

THIS is the backbone of the whole design -- not the integrator, not ASE. It is
the single narrow contract between the physics tier (which decides *what* to
compute) and the driver tier (which decides *how* and *where* to compute it,
and how to survive a crash while doing so).

A physics author sees only:

    request  = ESRequest(geom, states, want)     # "what I need"
    results  = executor.evaluate([req, req, ...]) # "give it to me" (blocks)

`evaluate` LOOKS synchronous and returns results keyed by request id. The fact
that underneath it fans out concurrently, caches, retries, applies backpressure,
and reconnects after a restart is invisible here. That invisibility is the
entire point: a student adding a new ES backend implements `compute_one` below
and never learns the word `async`.

The COUPLING LADDER lives in `want`: a backend advertises which coupling mode
it can supply, and the physics asks for the best available. This is what keeps
the framework portable across analytic-NACV codes, overlap-based codes, and
energy-gap-only ML surrogates.
"""
from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from enum import Enum


class Coupling(Enum):
    """The rungs of the coupling ladder, best-to-cheapest."""
    NACV = "analytic_nacv"        # analytic nonadiabatic coupling vectors
    OVERLAP = "wf_overlap"        # time-derivative couplings from WF overlaps
    ENERGY_GAP = "energy_gap"     # Baeck-An / gap-based (energies+grads only)


@dataclass(frozen=True)
class ESRequest:
    """An idempotent request for electronic-structure data at one geometry.

    Frozen + hashable so it can be a cache key directly. Two requests with the
    same geometry/states/method/want are the SAME computation and must never be
    run twice within a run -- that is what makes replay cheap and what keeps the
    NAC phase from silently flipping on recompute.
    """
    geom_key: tuple           # rounded geometry -> exact, hashable cache key
    method: str
    states: tuple[int, ...]
    want_coupling: Coupling
    want_gradients: bool = True

    @staticmethod
    def from_tbf(tbf, method, states, coupling, decimals=8) -> "ESRequest":
        gk = tuple(np.round(tbf.x, decimals).ravel().tolist())
        return ESRequest(gk, method, tuple(states), coupling)


@dataclass
class ESResult:
    """What every backend returns. A pure data bag -- no methods, no state."""
    energies: np.ndarray                         # (nstates,)
    gradients: dict[int, np.ndarray] = field(default_factory=dict)   # state -> grad
    nacv: dict = field(default_factory=dict)     # (i,j) -> coupling vector
    overlaps: np.ndarray | None = None           # <psi_i(t)|psi_j(t+dt)>
    phase: np.ndarray | None = None              # state-tracking metadata
    meta: dict = field(default_factory=dict)


class ElectronicStructureBackend:
    """Base class for a concrete QM code / ML model adapter.

    A physics author who wants a new backend subclasses this and implements
    `compute_one`. NOTHING about concurrency, caching, or resilience appears
    here -- the backend computes one geometry and returns one result. The
    executor is responsible for calling it many times, in parallel, at most
    once per distinct request, and for surviving crashes.
    """
    #: which coupling rungs this backend can actually supply
    supports: tuple[Coupling, ...] = (Coupling.ENERGY_GAP,)

    def compute_one(self, req: ESRequest) -> ESResult:   # pragma: no cover
        raise NotImplementedError

    def best_coupling(self, wanted: Coupling) -> Coupling:
        """Descend the ladder to the best rung this backend can supply."""
        order = [Coupling.NACV, Coupling.OVERLAP, Coupling.ENERGY_GAP]
        for rung in order[order.index(wanted):]:
            if rung in self.supports:
                return rung
        return Coupling.ENERGY_GAP
