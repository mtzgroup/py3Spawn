"""The electronic-structure seam (PR1).

This formalizes the seam that already existed by convention: every backend in
``pyspawn/potential/`` defined a ``compute_elec_struct(self, zbackprop)`` that was
monkeypatched onto ``traj`` and mutated it in place. That injection is replaced
here by a real interface:

  * ``ESResult`` -- a pure data bag: what a backend produces, no methods/state.
  * ``ElectronicStructureBackend`` -- base class a backend subclasses; it
    implements ``compute_one`` (compute one geometry, RETURN an ``ESResult``)
    and inherits ``apply_to_traj`` (write the result back through the existing
    trajectory setters).
  * a name registry (``register_backend`` / ``get_backend``) replacing the old
    ``compute_elec_struct_<software>_<method>`` string-eval dispatch.

The physics reads serially at the call site (``traj.compute_elec_struct``):
``result = backend.compute_one(traj, zbackprop); backend.apply_to_traj(...)``.

Mirrors the target contract in
``design/nextfms_skeleton/nextfms/es_provider.py`` (ESResult / compute_one),
adapted to the real trajectory state (energies, forces, timederivcoups, wf).

NOTE (PR1 scope): this is a pure refactor -- zero behavior change, gated by
tests/pr0_oracle.py at max|diff|=0. Concurrency/caching/an ``ESRequest`` cache
key belong to PR3/PR4 and are deliberately NOT introduced here.
"""

from dataclasses import dataclass, field


# --- registry: replaces the old string-name ("compute_elec_struct_<sw>_<method>")
#     dispatch with an explicit name -> backend-class map. -------------------
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
class ESResult:
    """What a backend returns for one geometry. A pure data bag -- no methods.

    Fields are ``None`` when a backend does not produce them, and
    ``apply_to_traj`` skips those (so a backend only writes what it computed).
    ``extra`` carries backend-specific restart state (civecs/orbs/phases/...)
    for backends that need it; unused by the hermetic cone model.
    """
    energies: object = None            # (nstates,)
    forces: object = None              # (nstates, ndims) -- traj's "forces_i"
    timederivcoups: object = None      # (nstates,)
    wf: object = None                  # (nstates, length_wf)
    prev_wf: object = None             # previous-step wf (state-tracking)
    extra: dict = field(default_factory=dict)


class ElectronicStructureBackend:
    """Base class for a concrete QM code / analytic-model adapter.

    Subclass and implement ``compute_one``. Nothing about concurrency, caching,
    or resilience appears here -- the backend computes one geometry and RETURNS
    one ``ESResult``. It must not mutate the trajectory; ``apply_to_traj`` does
    that, through the same setters the old in-place code used.
    """

    def compute_one(self, traj, zbackprop):   # pragma: no cover - abstract
        raise NotImplementedError

    def apply_to_traj(self, res, traj, zbackprop):
        """Write an ``ESResult`` back onto the trajectory.

        Uses the same setters, in the same top-to-bottom order, as the original
        in-place ``compute_elec_struct`` -- this is what keeps the refactor
        byte-for-byte identical (pr0_oracle.py, max|diff|=0).
        """
        cb = "backprop_" if zbackprop else ""
        if res.prev_wf is not None:
            getattr(traj, "set_" + cb + "prev_wf")(res.prev_wf)
        if res.energies is not None:
            getattr(traj, "set_" + cb + "energies")(res.energies)
        if res.forces is not None:
            getattr(traj, "set_" + cb + "forces")(res.forces)
        if res.timederivcoups is not None:
            getattr(traj, "set_" + cb + "timederivcoups")(res.timederivcoups)
        if res.wf is not None:
            getattr(traj, "set_" + cb + "wf")(res.wf)


class LegacyMutatingBackend(ElectronicStructureBackend):
    """Adapter for backends not yet migrated to native ``compute_one``.

    The real QM backends (TeraChem / OpenMolcas) still carry an in-place
    ``compute_elec_struct(self, zbackprop)`` that mutates the trajectory and has
    external side effects (sockets, scratch files) which cannot be exercised in
    this hermetic environment. Rather than rewrite validated physics blind, they
    flow through the same seam via this adapter: ``compute_one`` runs the legacy
    function for its side effects and returns an empty result; ``apply_to_traj``
    is a no-op. Migrating each to a native ``compute_one`` is a later PR, done
    when live QM validation is available.
    """

    def __init__(self, legacy_fn):
        self._legacy_fn = legacy_fn

    def compute_one(self, traj, zbackprop):
        self._legacy_fn(traj, zbackprop)
        return ESResult()

    def apply_to_traj(self, res, traj, zbackprop):
        return
