"""Trajectory generation & time-parameterization.

quintic / blend        : minimum-jerk point-to-point profiles (zero vel/acc at ends)
PathTrajectory         : waypoints (path) → smooth trajectory q(t) respecting per-joint
                         velocity and acceleration limits.

PathTrajectory method (simple, robust, easy to explain in an interview):
 1. parameterize waypoints by cumulative joint-space distance s
 2. fit a C² natural cubic spline q(s) (scipy CubicSpline) on the normalized parameter
 3. choose a single time scale T for a minimum-jerk timing law s(t) such that
    max |q'(s) ṡ| ≤ v_max and max |q'(s) s̈ + q''(s) ṡ²| ≤ a_max  (checked densely)
This is *not* time-optimal (TOPP-RA is), but it is smooth, never violates limits and
starts/stops at rest — good enough for manipulation demos.
"""
from __future__ import annotations

import numpy as np
from scipy.interpolate import CubicSpline


def quintic(t: float, T: float):
    """Minimum-jerk timing law on [0, T] → (s, ṡ, s̈) with s∈[0,1]."""
    if T <= 0:
        return 1.0, 0.0, 0.0
    tau = np.clip(t / T, 0.0, 1.0)
    s = 10 * tau**3 - 15 * tau**4 + 6 * tau**5
    sd = (30 * tau**2 - 60 * tau**3 + 30 * tau**4) / T
    sdd = (60 * tau - 180 * tau**2 + 120 * tau**3) / T**2
    if t <= 0 or t >= T:
        sd, sdd = 0.0, 0.0
    return s, sd, sdd


def blend(p0, p1, t, t0, T):
    """Min-jerk interpolation from p0 (at t0) to p1 (at t0+T). Returns (p, v, a)."""
    p0, p1 = np.asarray(p0, float), np.asarray(p1, float)
    s, sd, sdd = quintic(t - t0, T)
    dp = p1 - p0
    return p0 + s * dp, sd * dp, sdd * dp


class PathTrajectory:
    def __init__(self, waypoints: np.ndarray, vmax, amax, n_check: int = 400):
        W = np.asarray(waypoints, float)
        # drop duplicate consecutive waypoints
        keep = [0] + [i for i in range(1, len(W)) if np.linalg.norm(W[i] - W[i - 1]) > 1e-9]
        W = W[keep]
        if len(W) == 1:
            W = np.vstack([W, W])
        seg = np.linalg.norm(np.diff(W, axis=0), axis=1)
        u = np.concatenate([[0.0], np.cumsum(seg)])
        u = u / u[-1] if u[-1] > 0 else np.linspace(0, 1, len(W))
        self.W = W
        self.spline = CubicSpline(u, W, axis=0, bc_type="clamped") if len(W) > 2 else None
        vmax = np.broadcast_to(np.asarray(vmax, float), (W.shape[1],))
        amax = np.broadcast_to(np.asarray(amax, float), (W.shape[1],))
        # find minimal T (bisection) such that limits hold on a dense grid
        lo, hi = 1e-3, 1.0
        while not self._ok(hi, vmax, amax, n_check):
            hi *= 2.0
            if hi > 1e4:
                break
        for _ in range(40):
            mid = 0.5 * (lo + hi)
            if self._ok(mid, vmax, amax, n_check):
                hi = mid
            else:
                lo = mid
        self.T = hi

    def _qs(self, s):
        if self.spline is None:  # straight segment
            return self.W[0] + s * (self.W[1] - self.W[0]), self.W[1] - self.W[0], np.zeros_like(self.W[0])
        return self.spline(s), self.spline(s, 1), self.spline(s, 2)

    def _ok(self, T, vmax, amax, n):
        for t in np.linspace(0, T, n):
            s, sd, sdd = quintic(t, T)
            q, dq, ddq = self._qs(s)
            if np.any(np.abs(dq * sd) > vmax + 1e-9) or np.any(np.abs(dq * sdd + ddq * sd**2) > amax + 1e-9):
                return False
        return True

    def __call__(self, t: float):
        s, sd, sdd = quintic(t, self.T)
        q, dq, ddq = self._qs(s)
        return q, dq * sd, dq * sdd + ddq * sd**2
