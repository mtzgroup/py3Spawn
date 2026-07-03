"""Nonadiabatic coupling kernels shared across the physics tiers.

``compute_npi_tdc`` computes a time-derivative (nonadiabatic) coupling from a
wavefunction-overlap matrix ``W`` using the norm-preserving-interpolation (NPI)
scheme. It is a PURE function of ``(W, dt)`` -- no trajectory, no ``self`` -- so
the electronic-structure backends and the DGAS Hamiltonian can compute coupling
without reaching into a ``traj`` object.

Extracted verbatim from ``traj.compute_tdc`` in PR1b step 1 (behavior-identical:
``traj.compute_tdc`` now delegates here, passing ``self.get_timestep()`` as
``dt``). Validated by ``tests/pr0_oracle.py`` (max|diff|=0) and
``tests/test_compute_tdc.py``.
"""
import numpy as np


def compute_npi_tdc(Win, dt):
    """NPI time-derivative coupling from overlap matrix ``W`` and timestep ``dt``.

    ``W[0,0]``, ``W[1,1]`` are cos-like state overlaps (arccos arguments);
    ``W[0,1]``, ``W[1,0]`` are sin-like (arcsin arguments). All four are clamped
    into ``[-1, 1]`` before the angle formula (the off-diagonal clamp is the
    physics fix in the preceding commit).
    """
    W = Win.copy()
    if 1.0 < W[0, 0]: # < 1.01:
        W[0, 0] = 1.0
    if -1.0 > W[0, 0]: # > -1.01:
        W[0, 0] = -1.0
    if 1.0 < W[1, 1]: # < 1.01:
        W[1, 1] = 1.0
    if -1.0 > W[1, 1]: # > -1.01:
        W[1, 1] = -1.0
    if 1.0 < W[0, 1]: # < 1.01:
        W[0, 1] = 1.0
    if -1.0 > W[0, 1]: # > -1.01:
        W[0, 1] = -1.0
    if 1.0 < W[1, 0]: # < 1.01:
        W[1, 0] = 1.0
    if -1.0 > W[1, 0]: # > -1.01:
        W[1, 0] = -1.0
    Atmp = np.arccos(W[0, 0]) - np.arcsin(W[0, 1])
    Btmp = np.arccos(W[0, 0]) + np.arcsin(W[0, 1])
    Ctmp = np.arccos(W[1, 1]) - np.arcsin(W[1, 0])
    Dtmp = np.arccos(W[1, 1]) + np.arcsin(W[1, 0])
    Wlj = np.sqrt(1 - W[0, 0] * W[0, 0] - W[1, 0] * W[1, 0])
    if Wlj != Wlj:
        Wlj = 0.0
    if np.absolute(Atmp) < 1.0e-6:
        A = -1.0
    else:
        A = -1.0 * np.sin(Atmp) / Atmp
    if np.absolute(Btmp) < 1.0e-6:
        B = 1.0
    else:
        B = np.sin(Btmp) / Btmp
    if np.absolute(Ctmp) < 1.0e-6:
        C = 1.0
    else:
        C = np.sin(Ctmp) / Ctmp
    if np.absolute(Dtmp) < 1.0e-6:
        D = 1.0
    else:
        D = np.sin(Dtmp) / Dtmp
    if Wlj < 1.0e-6:
        E = 0.0
    else:
        Wlk = -1.0 * (W[0, 1] * W[0, 0] + W[1, 1] * W[1, 0]) / Wlj
        sWlj = np.sin(Wlj)
        sWlk = np.sin(Wlk)
        Etmp = np.sqrt((1 - Wlj * Wlj) * (1 - Wlk * Wlk))
        denom = sWlj * sWlj - sWlk * sWlk
        E = 2.0 * Wlj * (Wlj * Wlk * sWlj + (Etmp - 1.0) * sWlk) / denom
    tdc = 0.5 / dt * (np.arccos(W[0, 0]) * (A + B)
                      + np.arcsin(W[1, 0]) * (C + D) + E)
    return tdc
