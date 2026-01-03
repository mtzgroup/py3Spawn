from .traj import traj
from .hessian import hessian
from .simulation import simulation

def into_hessian(x):
    for method in x.__dict__:
        if method[0] != "_":
            setattr(hessian, method, getattr(x, method))
            
def into_traj(x):
    for method in x.__dict__:
        if method[0] != "_":
            setattr(traj, method, getattr(x, method))
            
def into_simulation(x):
    for method in x.__dict__:
        if method[0] != "_":
            setattr(simulation, method, getattr(x, method))
            
