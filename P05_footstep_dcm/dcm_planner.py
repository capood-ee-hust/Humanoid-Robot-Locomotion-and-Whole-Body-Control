from __future__ import annotations

import numpy as np
from scipy.linalg import solve_discrete_are

G = 9.81


class DCMPlanner:
    def __init__(self, z_c=0.66, T_step=0.6, dt=0.005):
        self.z, self.T, self.dt = z_c, T_step, dt
        self.w = np.sqrt(G / z_c)

    def plan(self, footsteps, c0):
       
        P = np.array([[f[0], f[1]] for f in footsteps])
        # step i: stance foot = footsteps[i+1]  (first foot to move is footsteps[1+1] ...)
        # ZMP sequence: start between feet, then on each stance foot, end between the last two
        zmps = [0.5 * (P[0] + P[1])] + [P[i] for i in range(1, len(P) - 1)] + [0.5 * (P[-1] + P[-2])]
        zmps = np.array(zmps)
        n = len(zmps)
        eos = np.zeros_like(zmps)
        eos[-1] = zmps[-1]
        ewT = np.exp(-self.w * self.T)
        for i in range(n - 1, 0, -1):
            ini = zmps[i] + ewT * (eos[i] - zmps[i])
            eos[i - 1] = ini
        # simulate
        ts, Z, X, C, V, S = [], [], [], [], [], []
        c = np.asarray(c0[:2], float).copy()
        steps_per = int(round(self.T / self.dt))
        for i in range(n):
            for k in range(steps_per if i < n - 1 else 2 * steps_per):
                t_loc = k * self.dt
                xi = zmps[i] + np.exp(self.w * (min(t_loc, self.T) - self.T)) * (eos[i] - zmps[i]) if i < n - 1 else zmps[i]
                cd = self.w * (xi - c)
                ts.append((i * steps_per + k) * self.dt); Z.append(zmps[i]); X.append(xi); C.append(c.copy()); V.append(cd); S.append(i)
                c = c + self.dt * cd
        return dict(t=np.array(ts), zmp=np.array(Z), dcm=np.array(X), com=np.array(C), com_v=np.array(V),
                    step=np.array(S), zmps=zmps, dcm_eos=eos)


class PreviewController:
    """Kajita ZMP preview control (one axis; apply per axis)."""

    def __init__(self, z_c=0.66, dt=0.005, preview_s=1.6, Qe=1.0, R=1e-6):
        self.dt = dt
        A = np.array([[1, dt, dt**2 / 2], [0, 1, dt], [0, 0, 1]])
        B = np.array([[dt**3 / 6], [dt**2 / 2], [dt]])
        C = np.array([[1, 0, -z_c / G]])
        # augmented system with integrator on ZMP error
        At = np.block([[np.eye(1), C @ A], [np.zeros((3, 1)), A]])
        Bt = np.vstack([C @ B, B])
        Q = np.zeros((4, 4)); Q[0, 0] = Qe
        P = solve_discrete_are(At, Bt, Q, R * np.eye(1))
        K = np.linalg.solve(R * np.eye(1) + Bt.T @ P @ Bt, Bt.T @ P @ At)
        self.Ke, self.Kx = K[0, 0], K[0, 1:]
        R = np.array([[R]])
        N = int(preview_s / dt)
        Ac = At - Bt @ K
        X = -Ac.T @ P @ np.array([[1], [0], [0], [0]])
        Gp = [-self.Ke]
        for _ in range(1, N):
            Gp.append(np.linalg.solve(R + Bt.T @ P @ Bt, Bt.T @ X).item())
            X = Ac.T @ X
        self.Gp = np.array(Gp)
        self.A, self.B, self.C, self.N = A, B, C, N

    def run(self, p_ref, x0):
        """p_ref: ZMP reference samples (one axis). Returns com, zmp."""
        x = np.array([x0, 0.0, 0.0])
        e_sum = 0.0
        pr = np.concatenate([p_ref, np.full(self.N, p_ref[-1])])
        com, zmp = [], []
        for k in range(len(p_ref)):
            p = (self.C @ x).item()
            e_sum += p - p_ref[k]
            u = -self.Ke * e_sum - self.Kx @ x - self.Gp[1:] @ pr[k + 1:k + self.N]
            x = self.A @ x + self.B.ravel() * u
            com.append(x[0]); zmp.append(p)
        return np.array(com), np.array(zmp)


def swing_trajectory(p0, p1, t, T, height=0.06):
    tau = np.clip(t / T, 0, 1)
    s = 10 * tau**3 - 15 * tau**4 + 6 * tau**5
    xy = (1 - s) * np.asarray(p0[:2]) + s * np.asarray(p1[:2])
    return np.array([xy[0], xy[1], height * np.sin(np.pi * tau)])
