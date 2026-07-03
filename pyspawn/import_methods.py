from .traj import traj
from .hessian import hessian
from .simulation import simulation
from .potential.es_backend import LegacyMutatingBackend

def into_hessian(x):
    for method in x.__dict__:
        if method[0] != "_":
            setattr(hessian, method, getattr(x, method))

def into_traj(x):
    for method in x.__dict__:
        # The electronic-structure seam is no longer monkeypatched onto traj
        # (see potential/es_backend.py). traj owns a native compute_elec_struct
        # shim that dispatches to the selected backend, wired below.
        if method[0] != "_" and method != "compute_elec_struct":
            setattr(traj, method, getattr(x, method))
    # Select the ES backend for this potential module via the registry: a
    # migrated backend exposes BACKEND; a legacy in-place compute_elec_struct is
    # wrapped in an adapter. Modules with neither (e.g. the vv integrator) leave
    # the current backend untouched.
    if hasattr(x, "BACKEND"):
        traj._es_backend = x.BACKEND()
    elif "compute_elec_struct" in x.__dict__:
        traj._es_backend = LegacyMutatingBackend(x.compute_elec_struct)

def into_simulation(x):
    for method in x.__dict__:
        if method[0] != "_":
            setattr(simulation, method, getattr(x, method))
            
