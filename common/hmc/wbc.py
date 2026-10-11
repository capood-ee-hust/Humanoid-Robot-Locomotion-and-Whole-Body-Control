"""QP-based Whole-Body Controller (weighted, TSID-style) for a floating-base humanoid.

Decision variables  x = [ a (nv) , f (3 × n_contact_points) ]
  a : generalized acceleration (base 6 + joints)
  f : contact forces at the foot contact points, world frame

Cost      Σ_k w_k ‖J_k a + (dJ_k v) − a*_k‖² + w_f ‖f‖² + w_a ‖a‖²
Subject to
  (1) floating-base dynamics  M_u a + h_u = J_{c,u}ᵀ f            (6 rows)
  (2) stance feet            J_foot a + dJ_foot v = −K_c J_foot v (6-D, no slip / no rotation)
  (3) friction pyramid        |f_x|, |f_y| ≤ μ̃ f_z
  (4) normal force bounds     f_min ≤ f_z ≤ f_max   (0 ≤ f_z ≤ 0 if swing)
  (5) torque limits           τ_min ≤ M_a a + h_a − J_{c,a}ᵀ f ≤ τ_max
Output: τ = M_a a* + h_a − J_{c,a}ᵀ f*  (feed-forward joint torques)
"""
from __future__ import annotations

from dataclasses import dataclass, field

import mujoco
import numpy as np

from .qp import DenseQP
from .robot import G1, full_mass_matrix, quat_to_mat, rot_log


@dataclass
class Task:
    name: str
    J: np.ndarray            # (k, nv)
    dJv: np.ndarray          # (k,)
    acc: np.ndarray          # desired task acceleration (k,)
    w: float | np.ndarray    # scalar or per-row weight


@dataclass
class WBCGains:
    com_kp: float = 60.0
    com_kd: float = 16.0
    ori_kp: float = 150.0
    ori_kd: float = 25.0
    post_kp: float = 60.0
    post_kd: float = 15.0
    foot_kp: float = 300.0
    foot_kd: float = 35.0
    hand_kp: float = 150.0
    hand_kd: float = 25.0
    w_com: float = 20.0
    w_pelvis: float = 5.0
    w_torso: float = 2.0
    w_post: float = 0.2
    w_foot: float = 50.0
    w_hand: float = 10.0
    w_force: float = 1e-4
    w_acc: float = 1e-5
    mu: float = 0.6
    fz_min: float = 2.0
    fz_max: float = 400.0
    contact_kd: float = 20.0


@dataclass
class WBCState:
    """Everything the WBC computed this step (for logging/diagnostics)."""
    ok: bool = False
    tau: np.ndarray | None = None
    a: np.ndarray | None = None
    f: np.ndarray | None = None
    iters: int = 0
    info: dict = field(default_factory=dict)


class WholeBodyController:
    NPTS = 8  # 4 contact spheres × 2 feet

    def __init__(self, robot: G1, gains: WBCGains | None = None):
        self.r = robot
        self.g = gains or WBCGains()
        nv = robot.nv
        self.nx = nv + 3 * self.NPTS
        self.nrow = 6 + 12 + 4 * self.NPTS + self.NPTS + robot.nu
        self.qp = DenseQP(self.nx, self.nrow)
        self.M = np.zeros((nv, nv))
        self.tau_scale = np.ones(robot.nu)     # used by fault injection / derating
        self.force_fail = False                # fault injection: emulate solver crash/timeout
        self.last = WBCState()

    # ---------------------------------------------------------------- task builders
    def com_task(self, c_ref, v_ref=None, a_ref=None, w=None, acc_override=None) -> Task:
        """CoM task. `acc_override[i]` (not NaN) replaces the PD law on axis i —
        used by the DCM balance law, which computes the CoM acceleration itself."""
        r, g = self.r, self.g
        v_ref = np.zeros(3) if v_ref is None else v_ref
        a_ref = np.zeros(3) if a_ref is None else a_ref
        J = r.com_jac()
        c, cd = r.com(), J @ r.d.qvel
        acc = a_ref + g.com_kp * (c_ref - c) + g.com_kd * (v_ref - cd)
        if acc_override is not None:
            ov = np.asarray(acc_override, float)
            acc = np.where(np.isnan(ov), acc, ov)
        return Task("com", J, None, acc, g.w_com if w is None else w)

    def orientation_task(self, body: int, R_ref, w, name) -> Task:
        r, g = self.r, self.g
        J6 = r.body_jac(body)
        Jr = J6[:3]
        R = r.d.xmat[body].reshape(3, 3)
        e = rot_log(R_ref @ R.T)
        acc = g.ori_kp * e - g.ori_kd * (Jr @ r.d.qvel)
        return Task(name, Jr, None, acc, w)

    def posture_task(self, q_ref_act, w=None, qd_ref=None, qdd_ref=None) -> Task:
        r, g = self.r, self.g
        J = np.zeros((r.nu, r.nv))
        J[np.arange(r.nu), r.act_dof] = 1.0
        q = r.d.qpos[r.act_qpos]
        qd = r.d.qvel[r.act_dof]
        qd_ref = np.zeros(r.nu) if qd_ref is None else qd_ref
        qdd_ref = np.zeros(r.nu) if qdd_ref is None else qdd_ref
        acc = qdd_ref + g.post_kp * (q_ref_act - q) + g.post_kd * (qd_ref - qd)
        return Task("posture", J, np.zeros(r.nu), acc, g.w_post if w is None else w)

    def foot_task(self, foot: int, p_ref, R_ref, v_ref=None, a_ref=None) -> Task:
        r, g = self.r, self.g
        b = r.foot_body[foot]
        J6 = r.body_jac(b)
        v6 = J6 @ r.d.qvel
        p = r.d.xpos[b]
        R = r.d.xmat[b].reshape(3, 3)
        v_ref = np.zeros(3) if v_ref is None else v_ref
        a_ref = np.zeros(3) if a_ref is None else a_ref
        acc_lin = a_ref + g.foot_kp * (p_ref - p) + g.foot_kd * (v_ref - v6[3:])
        acc_ang = g.foot_kp * rot_log(R_ref @ R.T) - g.foot_kd * v6[:3]
        return Task(f"foot{foot}", J6, None, np.concatenate([acc_ang, acc_lin]), g.w_foot)

    def point_task(self, body: int, local_point, p_ref, v_ref=None, a_ref=None, w=None, name="hand") -> Task:
        r, g = self.r, self.g
        R = r.d.xmat[body].reshape(3, 3)
        p = r.d.xpos[body] + R @ local_point
        J = r.point_jac(body, p)
        v_ref = np.zeros(3) if v_ref is None else v_ref
        a_ref = np.zeros(3) if a_ref is None else a_ref
        acc = a_ref + g.hand_kp * (p_ref - p) + g.hand_kd * (v_ref - J @ r.d.qvel)
        t = Task(name, J, None, acc, g.w_hand if w is None else w)
        lp = np.asarray(local_point, dtype=float)
        t.jac_fn = lambda dd, b=body, lp=lp: r.point_jac(b, dd.xpos[b] + dd.xmat[b].reshape(3, 3) @ lp, dd)
        return t

    # ---------------------------------------------------------------- main solve
    def solve(self, tasks: list[Task], stance=(True, True), fz_max_scale=(1.0, 1.0)) -> WBCState:
        r, g, m, d = self.r, self.g, self.r.m, self.r.d
        nv, nu, nx = r.nv, r.nu, self.nx
        npts = self.NPTS

        full_mass_matrix(m, d, self.M)
        M, h = self.M, d.qfrc_bias.copy()

        # contact point Jacobians (world frame) at the 8 contact spheres -> used for Jcᵀ f
        pts = [(f, gid) for f in (0, 1) for gid in r.foot_geoms[f]]
        Jc = np.vstack([r.point_jac(r.foot_body[f], d.geom_xpos[gid]) for f, gid in pts])  # (24, nv)
        # 6-D foot Jacobians -> used for the no-slip constraint (rank-6 per foot, no redundancy)
        ffns = [(lambda dd, b=b: r.body_jac(b, dd)) for b in r.foot_body]
        tfns = [self._jac_fn_for(t) for t in tasks if t.dJv is None]
        drifts = r.drift_terms(ffns + tfns)
        Jf = [fn(d) for fn in ffns]
        dJf_v = drifts[:2]
        k = 2
        for t in tasks:
            if t.dJv is None:
                t.dJv = drifts[k]; k += 1

        # ------------------------------------------------ cost
        P = np.zeros((nx, nx)); q = np.zeros(nx)
        for t in tasks:
            W = np.broadcast_to(np.asarray(t.w, dtype=float), (t.J.shape[0],))
            JW = t.J.T * W
            P[:nv, :nv] += JW @ t.J
            q[:nv] += JW @ (t.dJv - t.acc)
        P[:nv, :nv] += g.w_acc * np.eye(nv)
        P[nv:, nv:] += g.w_force * np.eye(3 * npts)

        # ------------------------------------------------ constraints
        A = np.zeros((self.nrow, nx)); lo = np.zeros(self.nrow); hi = np.zeros(self.nrow)
        row = 0
        # (1) floating base dynamics
        A[0:6, :nv] = M[:6]
        A[0:6, nv:] = -Jc[:, :6].T
        lo[0:6] = hi[0:6] = -h[:6]
        row = 6
        # (2) stance feet: zero 6-D acceleration (+ velocity damping)
        active = np.repeat([stance[0], stance[1]], 4)
        for foot in (0, 1):
            sl = slice(row + 6 * foot, row + 6 * foot + 6)
            if stance[foot]:
                A[sl, :nv] = Jf[foot]
                lo[sl] = hi[sl] = -dJf_v[foot] - g.contact_kd * (Jf[foot] @ d.qvel)
        row += 12
        # (3) friction pyramid
        mu = g.mu / np.sqrt(2.0)
        for i in range(npts):
            c = nv + 3 * i
            for ax in (0, 1):
                for s in (1.0, -1.0):
                    A[row, c + ax] = s
                    A[row, c + 2] = -mu
                    lo[row], hi[row] = -np.inf, 0.0
                    row += 1
        # (4) normal force bounds
        for i in range(npts):
            A[row, nv + 3 * i + 2] = 1.0
            if active[i]:
                hi[row] = g.fz_max * fz_max_scale[i // 4]
                lo[row] = min(g.fz_min, hi[row])       # allow full unloading
            row += 1
        # (5) torque limits on actuated dofs
        Ma = M[r.act_dof]
        A[row:row + nu, :nv] = Ma
        A[row:row + nu, nv:] = -Jc[:, r.act_dof].T
        tmax = r.tau_max * self.tau_scale
        lo[row:row + nu] = -tmax - h[r.act_dof]
        hi[row:row + nu] = tmax - h[r.act_dof]

        x = None if self.force_fail else self.qp.solve(P, q, A, lo, hi)
        st = WBCState(ok=x is not None, iters=self.qp.iters,
                      info={"status": "forced_fail" if self.force_fail else self.qp.status})
        if x is not None:
            a, f = x[:nv], x[nv:]
            tau = Ma @ a + h[r.act_dof] - Jc[:, r.act_dof].T @ f
            st.a, st.f, st.tau = a, f.reshape(npts, 3), np.clip(tau, -tmax, tmax)
        self.last = st
        return st

    def _jac_fn_for(self, t: Task):
        r = self.r
        if t.name == "com":
            return lambda dd: r.com_jac(dd)
        if t.name.startswith("foot"):
            b = r.foot_body[int(t.name[4:])]
            return lambda dd, b=b: r.body_jac(b, dd)
        if t.name in ("pelvis", "torso"):
            b = r.pelvis if t.name == "pelvis" else r.torso
            return lambda dd, b=b: r.body_jac(b, dd)[:3]
        if hasattr(t, "jac_fn"):
            return t.jac_fn
        # fallback: treat Jacobian as constant (dJv ≈ 0)
        J = t.J
        return lambda dd, J=J: J

    # ---------------------------------------------------------------- utilities
    @staticmethod
    def cop(f: np.ndarray, pts: np.ndarray) -> np.ndarray:
        """Center of pressure (xy) of a set of point forces."""
        fz = f[:, 2].sum()
        return (pts[:, :2] * f[:, 2:3]).sum(0) / max(fz, 1e-9)
