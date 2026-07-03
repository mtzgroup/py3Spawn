import math
import numpy as np

from .es_backend import ElectronicStructureBackend, ESResult, register_backend


#################################################
### electronic structure routines go here #######
#################################################

# each electronic structure method requires at least two routines:
# 1) compute_elec_struct_, which computes energies, forces, and wfs
# 2) init_h5_datasets_, which defines the datasets to be output to hdf5
# 3) potential_specific_traj_copy, which copies data that is potential specific
#   from one traj data structure to another
# other ancillary routines may be included as well


### pyspawn_cone electronic structure ###
@register_backend("test_cone")
class TestConeBackend(ElectronicStructureBackend):
    """Analytic Jahn-Teller conical-intersection model (hermetic; no external QM).

    PR1: this holds the same computation that used to live in the module-level
    ``compute_elec_struct(self, zbackprop)`` monkeypatched onto ``traj``. It now
    RETURNS an ``ESResult`` instead of mutating the trajectory; the caller
    (traj.compute_elec_struct) writes it back via the inherited ``apply_to_traj``.
    The math and wavefunction phasing are unchanged -- pr0_oracle.py stays at
    max|diff|=0.
    """

    def compute_one(self, traj, zbackprop):
        if not zbackprop:
            cbackprop = ""
        else:
            cbackprop = "backprop_"

        # the current wf becomes the previous wf (state-tracking); we read it
        # directly instead of writing it back first, then re-reading it.
        prev_wf = getattr(traj, "get_" + cbackprop + "wf")()

        pos = getattr(traj, "get_" + cbackprop + "positions")()
        x = pos[0]
        y = pos[1]
        r = math.sqrt(x * x + y * y)
        theta = (math.atan2(y, x)) / 2.0

        e = np.zeros(traj.numstates)
        e[0] = (r - 1.0) * (r - 1.0) - 1.0
        e[1] = (r + 1.0) * (r + 1.0) - 1.0

        f = np.zeros((traj.numstates, traj.numdims))
        ftmp = -2.0 * (r - 1.0)
        f[0, 0] = (x / r) * ftmp
        f[0, 1] = (y / r) * ftmp
        ftmp = -2.0 * (r + 1.0)
        f[1, 0] = (x / r) * ftmp
        f[1, 1] = (y / r) * ftmp

        wf = np.zeros((traj.numstates, traj.length_wf))
        wf[0, 0] = math.sin(theta)
        wf[0, 1] = math.cos(theta)
        wf[1, 0] = math.cos(theta)
        wf[1, 1] = -math.sin(theta)
        # phasing wave funciton to match previous time step
        W = np.matmul(prev_wf, wf.T)
        if W[0, 0] < 0.0:
            wf[0, :] = -1.0 * wf[0, :]
            W[:, 0] = -1.0 * W[:, 0]
        if W[1, 1] < 0.0:
            wf[1, :] = -1.0 * wf[1, :]
            W[:, 1] = -1.0 * W[:, 1]
        # computing NPI derivative coupling
        tmp = traj.compute_tdc(W)
        tdc = np.zeros(traj.numstates)
        if traj.istate == 1:
            jstate = 0
        else:
            jstate = 1
        tdc[jstate] = tmp

        return ESResult(energies=e, forces=f, timederivcoups=tdc,
                        wf=wf, prev_wf=prev_wf)


#: selected by import_methods.into_traj(pyspawn.potential.test_cone)
BACKEND = TestConeBackend


def init_h5_datasets(self):
    self.h5_datasets["time"] = 1
    self.h5_datasets["energies"] = self.numstates
    self.h5_datasets["positions"] = self.numdims
    self.h5_datasets["momenta"] = self.numdims
    self.h5_datasets["forces_i"] = self.numdims
    self.h5_datasets["wf0"] = self.numstates
    self.h5_datasets["wf1"] = self.numstates
    self.h5_datasets_half_step["time_half_step"] = 1
    self.h5_datasets_half_step["timederivcoups"] = self.numstates


def potential_specific_traj_copy(self, from_traj):
    return


def get_wf0(self):
    return self.wf[0, :].copy()


def get_wf1(self):
    return self.wf[1, :].copy()


def get_backprop_wf0(self):
    return self.backprop_wf[0, :].copy()


def get_backprop_wf1(self):
    return self.backprop_wf[1, :].copy()

###end pyspawn_cone electronic structure section###
