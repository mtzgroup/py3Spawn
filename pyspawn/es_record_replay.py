"""Record/replay harness for the electronic-structure seam.

Turns ONE real run (e.g. TeraChem / OpenMolcas) into a hermetic regression
fixture. ``RecordingBackend`` wraps the real backend and snapshots the ES
*outputs* it produces, keyed by ``(label, direction, rounded geometry)``.
``ReplayBackend`` plays those snapshots back with no QM program, so the whole AIMS
trajectory reproduces byte-for-byte in an environment without the QM code.

This is the QM analogue of ``tests/pr0_oracle.py``: the cone oracle validates the
framework on an analytic model; a captured fixture + ``ReplayBackend`` validates
the framework on a *real* QM trajectory, hermetically, so QM-touching refactors
(the backend migration off ``LegacyMutatingBackend``, the PR3 driver, ...) get a
regression gate they otherwise lack (`current_architecture.md` section 4e:
the cone oracle is blind to S_elec / DGAS / QM state).

Which ES outputs? The dynamics consumes ``energies, forces, timederivcoups,
S_elec_flat`` (and ``wf`` for the ``wf0/wf1`` log); the integrator re-derives
positions/momenta from them (`current_architecture.md` section 4). The QM backends
*additionally* write ``civecs``/``orbs`` (CI vectors + orbitals) to the durable
HDF5 (``terachem_cas.init_h5_datasets``), so those are recorded too -- not because
anything recomputes them (replay never calls QM), but so the replayed HDF5 matches
the golden byte-for-byte. Their scalar sizes ``ncivecs``/``norbs`` are rebuilt on
restore. The analytic cone has no civecs/orbs, so on the cone these fields are
simply absent from the snapshot (``getattr`` -> None -> skipped).

Keying: ``(label, direction, round(geometry, 10))``. Geometry is order-independent
(so the fixture survives a driver reorder -- exactly what PR3 needs to validate),
at the cost of one assumption: a given (label, direction) does not revisit the
same rounded geometry with a *different* ES output. For continuous trajectories
this holds; see ``es_seam_contract.md`` section 4.1 for the provenance-key
discussion.

Validated hermetically by ``tests/test_es_replay.py`` (record a cone run, replay
it, max|diff|=0). Capture instructions: ``design/qm_fixture_capture.md``.
"""

import pickle

import numpy as np

from .potential.es_backend import ElectronicStructureBackend, ESResult

#: ES output attributes the dynamics consumes (plus wf, which is logged). Their
#: ``backprop_`` twins are handled via the channel prefix. ``civecs``/``orbs`` are
#: QM continuation state (absent on the analytic cone), but the QM backends write
#: them to the durable HDF5 (``terachem_cas.init_h5_datasets``), so the tape must
#: carry them for a byte-for-byte replay; ``_restore`` rebuilds the derived scalar
#: sizes ``ncivecs``/``norbs`` those datasets are shaped by.
ES_OUTPUT_FIELDS = ("energies", "forces", "timederivcoups", "S_elec_flat", "wf",
                    "civecs", "orbs")

_DECIMALS = 10


def _key(request):
    direction = "backprop" if request.zbackprop else "forward"
    label = request.traj.get_label()
    pos = np.round(np.asarray(request.positions, dtype=float), _DECIMALS)
    return (label, direction, pos.tobytes())


def _snapshot(traj, zbackprop):
    cb = "backprop_" if zbackprop else ""
    snap = {}
    for name in ES_OUTPUT_FIELDS:
        attr = cb + name
        val = getattr(traj, attr, None)
        if isinstance(val, np.ndarray):
            snap[attr] = val.copy()
    return snap


def _restore(traj, snap):
    for attr, val in snap.items():
        setattr(traj, attr, np.asarray(val).copy())
    # QM backends size their civecs/orbs HDF5 datasets from the scalar
    # ncivecs/norbs, which the real backend derives from the array sizes during
    # compute (terachem_cas set_ncivecs/set_norbs). Reproduce those scalars so
    # init_h5_datasets() works when the arrays are supplied from the tape.
    for arr_name, n_name in (("civecs", "ncivecs"), ("orbs", "norbs")):
        for attr, val in snap.items():
            if attr.endswith(arr_name):
                setattr(traj, n_name, int(np.asarray(val).size))


class RecordingBackend(ElectronicStructureBackend):
    """Wrap a real backend; run it, apply it, then snapshot the ES outputs.

    Fully handles compute + apply + snapshot and returns an empty result, so the
    seam's outer ``apply_to_traj`` is a no-op regardless of whether ``inner`` is
    migrated (returns an ``ESResult``) or legacy (mutates the traj in place). The
    recorded trajectory is therefore identical to an un-wrapped run.
    """

    def __init__(self, inner, tape):
        self._inner = inner
        self.tape = tape           # dict: key -> {attr: ndarray}

    def compute_one(self, request):
        result = self._inner.compute_one(request)
        self._inner.apply_to_traj(result, request.traj, request.zbackprop)
        self.tape[_key(request)] = _snapshot(request.traj, request.zbackprop)
        return ESResult()

    def apply_to_traj(self, res, traj, zbackprop):
        return


class ReplayBackend(ElectronicStructureBackend):
    """Replay ES outputs from a tape -- no QM program.

    Keyed identically to the recording. A missing key means the replayed
    trajectory diverged from the recorded one (should never happen for a
    deterministic run) -- raised loudly rather than silently mishandled.
    """

    def __init__(self, tape):
        self.tape = tape

    def compute_one(self, request):
        key = _key(request)
        try:
            snap = self.tape[key]
        except KeyError:
            raise KeyError(
                "replay diverged: no recorded ES output for %r "
                "(the replayed trajectory visited a geometry the recording "
                "did not)" % (key,))
        _restore(request.traj, snap)
        return ESResult()

    def apply_to_traj(self, res, traj, zbackprop):
        return


def save_tape(tape, path):
    with open(path, "wb") as f:
        pickle.dump(tape, f, protocol=4)


def load_tape(path):
    with open(path, "rb") as f:
        return pickle.load(f)


# --- hdf5 trajectory diff (shared by the cone self-test and a QM replay oracle) ---

def _load_hdf5(path):
    import h5py
    d = {}
    with h5py.File(path, "r") as f:
        f.visititems(
            lambda n, o: d.__setitem__(n, o[()])
            if isinstance(o, h5py.Dataset) else None)
    return d


def max_hdf5_diff(path1, path2):
    """Return (n_datasets, max_abs_diff, worst_key, n_shape_mismatch)."""
    a, b = _load_hdf5(path1), _load_hdf5(path2)
    keys = sorted(set(a) & set(b))
    maxd, worst, mism = 0.0, None, 0
    for k in keys:
        x, y = np.asarray(a[k]), np.asarray(b[k])
        if x.shape != y.shape:
            mism += 1
            continue
        if np.issubdtype(x.dtype, np.number) and x.size:
            d = np.nanmax(np.abs(x - y))
            if d > maxd:
                maxd, worst = d, k
    return len(keys), maxd, worst, mism
