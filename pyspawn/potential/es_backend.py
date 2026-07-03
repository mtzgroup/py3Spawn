"""The electronic-structure seam (PR1 / PR1b).

Formalizes the boundary between the physics and the electronic-structure codes:

  * ``ESState``   -- opaque, backend-defined continuation state threaded across a
                     trajectory's ES calls (cone: the wavefunction ``wf``). The
                     physics never inspects it.
  * ``ESRequest`` -- inputs for one ES evaluation. A migrated backend reads only
                     the value fields (geometry, states, prior_state, dt) and
                     never touches a traj. (``traj``/``zbackprop`` are a
                     transitional escape hatch for un-migrated legacy backends;
                     see ``LegacyMutatingBackend``.)
  * ``ESResult``  -- what a backend returns: the physics-facing outputs
                     (energies, forces, timederivcoups) plus the opaque
                     continuation ``state``.
  * ``ElectronicStructureBackend`` -- base class; implement
                     ``compute_one(request)``; inherit ``apply_to_traj``.
  * ``register_backend`` / ``get_backend`` -- name registry replacing the old
                     string-eval backend dispatch.

PR1b narrows ``compute_one`` from ``(traj, zbackprop)`` to ``(request)``: a
migrated backend computes from values alone. Continuation state is explicit
(``ESState``) instead of the write-only ``prev_wf`` attribute (which was dead
post-PR1 and is removed here). Durably relocating ``ESState`` off the traj (into
the executor) stays deferred to PR3; for now ``apply_to_traj`` writes it back onto
the traj, so serialization/restart are unchanged. Gated by tests/pr0_oracle.py at
max|diff|=0.
"""

from dataclasses import dataclass, field


# --- registry: replaces the old string-name backend dispatch. --------------
_REGISTRY = {}


def register_backend(name):
    """Class decorator: register a backend under ``name`` for get_backend()."""
    def _decorator(cls):
        _REGISTRY[name] = cls
        return cls
    return _decorator


def get_backend(name):
    """Return a fresh instance of the backend registered under ``name``."""
    return _REGISTRY[name]()


@dataclass
class ESState:
    """Opaque electronic-structure continuation state, threaded across a
    trajectory's ES calls. Backend-defined; the physics never reads a field.
    For the cone model this is just the wavefunction ``wf``.
    """
    wf: object = None
    extra: dict = field(default_factory=dict)


@dataclass
class ESRequest:
    """Inputs for one ES evaluation.

    A migrated backend reads ONLY the value fields below and never touches a
    traj. ``traj`` and ``zbackprop`` are a transitional escape hatch used solely
    by ``LegacyMutatingBackend`` for the un-migrated QM backends; they are
    removed once the last legacy backend gains a native ``compute_one``.
    """
    positions: object
    istate: int
    numstates: int
    numdims: int
    length_wf: int
    dt: float
    prior_state: object = None       # ESState | None -- this chain's prev output
    # --- transitional legacy escape hatch (LegacyMutatingBackend only) ---
    traj: object = None
    zbackprop: bool = False


@dataclass
class ESResult:
    """What a backend returns: physics-facing outputs + opaque continuation.

    Fields left ``None`` are not written back (a backend writes only what it
    computed). ``extra`` carries backend-specific data for later use.
    """
    energies: object = None
    forces: object = None            # (nstates, ndims) -- traj "forces_i"
    timederivcoups: object = None
    state: object = None             # ESState | None -- opaque continuation (cone: wf)
    extra: dict = field(default_factory=dict)


class ElectronicStructureBackend:
    """Base class for a concrete QM code / analytic-model adapter.

    Subclass and implement ``compute_one(request) -> ESResult``. It must not
    mutate the trajectory; ``apply_to_traj`` writes the result back, through the
    same setters the old in-place code used.
    """

    def compute_one(self, request):   # pragma: no cover - abstract
        raise NotImplementedError

    def apply_to_traj(self, res, traj, zbackprop):
        """Write an ``ESResult`` back onto the trajectory.

        Uses the same setters, in the same order, as the original in-place
        ``compute_elec_struct`` -- keeping the refactor byte-for-byte identical
        (pr0_oracle.py, max|diff|=0). The opaque continuation state lands on the
        traj (``traj.wf``) for now; PR3 relocates it into the executor's store.
        """
        cb = "backprop_" if zbackprop else ""
        if res.energies is not None:
            getattr(traj, "set_" + cb + "energies")(res.energies)
        if res.forces is not None:
            getattr(traj, "set_" + cb + "forces")(res.forces)
        if res.timederivcoups is not None:
            getattr(traj, "set_" + cb + "timederivcoups")(res.timederivcoups)
        if res.state is not None and res.state.wf is not None:
            getattr(traj, "set_" + cb + "wf")(res.state.wf)


class LegacyMutatingBackend(ElectronicStructureBackend):
    """Adapter for backends not yet migrated to native ``compute_one``.

    The real QM backends (TeraChem / OpenMolcas) still carry an in-place
    ``compute_elec_struct(traj, zbackprop)`` that mutates the trajectory and has
    external side effects (sockets, scratch files) that cannot be exercised in
    this hermetic environment. Rather than rewrite validated physics blind, they
    flow through the same seam via this adapter: ``compute_one`` runs the legacy
    function (reached through the request's transitional ``traj`` escape hatch)
    for its side effects and returns an empty result; ``apply_to_traj`` is a
    no-op. Migrating each to a native ``compute_one`` is a later PR, done when
    live QM validation is available.
    """

    def __init__(self, legacy_fn):
        self._legacy_fn = legacy_fn

    def compute_one(self, request):
        self._legacy_fn(request.traj, request.zbackprop)
        return ESResult()

    def apply_to_traj(self, res, traj, zbackprop):
        return
