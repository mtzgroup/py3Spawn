"""
A toy 2-state, 1-D electronic-structure backend (student tier).

This stands in for a real quantum-chemistry adapter (Molpro/OpenMolcas/BAGEL/
TeraChem) or an ML surrogate. It shows the shape a physics author fills in: a
pure `compute_one(request) -> ESResult`. No concurrency, no caching -- the
executor owns all of that.

Physics: a 1-D avoided crossing (a Tully-like model). Two adiabatic states with
a coupling region near x=0. Real backends replace this body with a call out to a
QC code; the SIGNATURE is the contract that never changes.
"""
from __future__ import annotations

import numpy as np
from .es_provider import ElectronicStructureBackend, ESRequest, ESResult, Coupling


class ToyAvoidedCrossing(ElectronicStructureBackend):
    # This toy can supply the top rung; an ML surrogate would advertise only
    # ENERGY_GAP and the coupling ladder would descend automatically.
    supports = (Coupling.NACV, Coupling.OVERLAP, Coupling.ENERGY_GAP)

    A, B, C, D = 0.01, 1.6, 0.005, 1.0

    def compute_one(self, req: ESRequest) -> ESResult:
        x = np.array(req.geom_key, dtype=float)[0]
        # diabatic 2x2, then diagonalize to adiabatic
        v11 = np.sign(x) * self.A * (1.0 - np.exp(-self.B * abs(x)))
        v22 = -v11
        v12 = self.C * np.exp(-self.D * x * x)
        H = np.array([[v11, v12], [v12, v22]])
        w, U = np.linalg.eigh(H)                      # adiabatic energies/vecs
        # analytic gradient of each adiabatic surface (Hellmann-Feynman on H)
        dx = 1e-6
        H2 = H.copy()
        v11b = np.sign(x + dx) * self.A * (1 - np.exp(-self.B * abs(x + dx)))
        v12b = self.C * np.exp(-self.D * (x + dx) ** 2)
        H2 = np.array([[v11b, v12b], [v12b, -v11b]])
        w2, _ = np.linalg.eigh(H2)
        grad = {int(s): np.array([(w2[s] - w[s]) / dx]) for s in req.states}
        res = ESResult(energies=w, gradients=grad)
        if req.want_coupling == Coupling.NACV:
            # crude finite-difference derivative coupling d_01
            d01 = float(U[:, 0] @ ((H2 - H) / dx) @ U[:, 1]) / max(w[1] - w[0], 1e-6)
            res.nacv = {(0, 1): np.array([d01]), (1, 0): np.array([-d01])}
        return res
