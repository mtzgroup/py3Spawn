#!/usr/bin/env python
"""Hermetic unit test for traj.compute_tdc (NPI time-derivative coupling).

Guards the copy-paste bug in the W-clamping block (traj.py ~1054-1061): the four
lines meant to clamp the OFF-diagonal overlaps W[0,1], W[1,0] into [-1,1] (the
arguments of arcsin in the NPI angle formula) instead tested the off-diagonal but
wrote the DIAGONAL, so the off-diagonal was never clamped and arcsin(|W|>1) -> NaN.

The cone oracle (pr0_oracle.py) NEVER drives W out of [-1,1] (measured peak
|W_offdiag| ~= 0.9998), so it cannot see this bug. This test does, by feeding an
out-of-range off-diagonal directly. Pure function of (W, dt): no ES backend, no
TeraChem.

Input choice: the matrices below are deliberately simple (diagonal ~0, one
out-of-range off-diagonal) so that clamping the off-diagonal to +-1 leaves ALL
downstream NPI intermediates finite. A richer matrix like [[0.9,1.4],[0.1,0.9]]
would clamp correctly yet still NaN via a SEPARATE, unguarded path -- the derived
Wlk = -(W01*W00 + W11*W10)/Wlj can exceed 1, and Etmp = sqrt((1-Wlj^2)(1-Wlk^2))
(traj.py ~1091) then takes sqrt of a negative. That Wlk/Etmp fragility is a
distinct latent issue (only reachable with unphysical W) and is intentionally NOT
exercised here; this test isolates the clamp fix only.

Run: PYTHONPATH=<repo> python tests/test_compute_tdc.py   (exit 0 = PASS)
"""
import sys
import numpy as np
import pyspawn


def _traj(dt=0.02):
    tr = pyspawn.traj(2, 2)      # compute_tdc is native on traj; needs only dt
    tr.set_timestep(dt)
    return tr


def check(name, cond):
    print("  [%s] %s" % ("PASS" if cond else "FAIL", name))
    return bool(cond)


def main():
    tr = _traj()
    ok = True

    # (1) out-of-range W[0,1] must not NaN. BUG: arcsin(1.5)=NaN -> tdc=NaN.
    ok &= check("out-of-range W[0,1]=1.5 -> finite tdc",
                np.isfinite(tr.compute_tdc(np.array([[0.0, 1.5],
                                                     [0.0, 0.0]]))))

    # (2) same for the other off-diagonal W[1,0].
    ok &= check("out-of-range W[1,0]=-1.3 -> finite tdc",
                np.isfinite(tr.compute_tdc(np.array([[0.0, 0.0],
                                                     [-1.3, 0.0]]))))

    # (3) the clamp must target the OFF-diagonal: W[0,1]=1.5 clamps to 1.0, so it
    #     must give the same result as W[0,1]=1.0 (proves the right element is
    #     clamped, not the diagonal). BUG clamps the diagonal -> a is NaN.
    a = tr.compute_tdc(np.array([[0.0, 1.5], [0.0, 0.0]]))
    b = tr.compute_tdc(np.array([[0.0, 1.0], [0.0, 0.0]]))
    ok &= check("W[0,1]=1.5 clamps to the off-diagonal (== W[0,1]=1.0 result)",
                np.isfinite(a) and np.isclose(a, b))

    # (4) sanity: an in-range W (what the cone actually sees) stays finite and is
    #     untouched by the clamp block.
    ok &= check("in-range W -> finite tdc",
                np.isfinite(tr.compute_tdc(np.array([[0.7, -0.3],
                                                     [0.3, 0.7]]))))

    print("OVERALL:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
