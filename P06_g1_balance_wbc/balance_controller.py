from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "common"))
from hmc.robot import G1, polygon_halfspaces              # noqa: E402
from hmc.safety import Mode, SafetyConfig, SafetySupervisor  # noqa: E402
from hmc.sim import tilt_angle                               # noqa: E402
from hmc.timing import blend                                 # noqa: E402
from hmc.wbc import WBCGains, WholeBodyController            # noqa: E402


@dataclass
class Reference:
    com: np.ndarray
    com_v: np.ndarray
    com_a: np.ndarray
    stance: tuple = (True, True)
    fz_scale: tuple = (1.0, 1.0)
    swing_foot: int | None = None
    swing_p: np.ndarray | None = None
    swing_v: np.ndarray | None = None
    swing_a: np.ndarray | None = None


class BalanceController:
    def __init__(self, robot: G1, gains: WBCGains | None = None, safety_cfg: SafetyConfig | None = None):
        self.r = robot
        self.wbc = WholeBodyController(robot, gains)
        d = robot.d
        self.q_ref = d.qpos[robot.act_qpos].copy()
        self.c0 = robot.com()
        self.foot_p0 = [d.xpos[b].copy() for b in robot.foot_body]
        self.foot_R0 = [d.xmat[b].reshape(3, 3).copy() for b in robot.foot_body]
        # posture weights: legs almost free (CoM/feet tasks decide them), waist & arms held
        self.w_post = np.full(robot.nu, self.wbc.g.w_post)
        for i, n in enumerate(robot.act_names):
            if "hip" in n or "knee" in n or "ankle" in n:
                self.w_post[i] = 1e-3
        self.sup = SafetySupervisor(robot.nu, tau_cont=0.6 * robot.tau_max, cfg=safety_cfg or SafetyConfig())
        self.reference_fn = lambda t: Reference(self.c0, np.zeros(3), np.zeros(3))
        self.use_dcm = False     
        self.k_xi = 3.0
        self.cop_margin = 0.008          
        self.cop_saturated = False
        self.dcm_info = {}
        self.sit_start = None
        self.degraded_since = None
        self.log = []

    # main
    def compute(self, t: float) -> np.ndarray:
        r, d, sup = self.r, self.r.d, self.sup
        valid = sup.check_sensors(t, d.qpos, d.qvel)
        if valid:
            sup.check_tilt(t, tilt_angle(d, r.pelvis))

        mode = sup.mode
        if mode == Mode.ESTOP:
            return np.zeros(r.nu)
        if mode == Mode.DAMPING:
            qv = np.nan_to_num(d.qvel[r.act_dof])
            return sup.damping_torque(qv)
        if not valid:                      # hold last good command for a few cycles
            return sup.last_good_tau

        if mode == Mode.DEGRADED:
            if self.degraded_since is None:
                self.degraded_since = t
            if t - self.degraded_since > 0.3:
                sup.request(t, Mode.SAFE_SIT, "DEGRADED → hạ thấp có kiểm soát")
        ref = self.reference_fn(t)
        if sup.mode == Mode.SAFE_SIT:
            ref = self._sit_reference(t)

        # DCM law only in loaded double support; single support uses the CoM PD law
        # (on a 6 cm-wide foot the clamped DCM law leaves almost no lateral authority)
        dbl = ref.stance == (True, True) and min(ref.fz_scale) > 0.5
        acc_xy = self.dcm_law(ref) if (self.use_dcm and dbl) else None
        tasks = [
            self.wbc.com_task(ref.com, ref.com_v, ref.com_a,
                              acc_override=None if acc_xy is None else [acc_xy[0], acc_xy[1], np.nan]),
            self.wbc.orientation_task(r.pelvis, np.eye(3), self.wbc.g.w_pelvis, "pelvis"),
            self.wbc.orientation_task(r.torso, np.eye(3), self.wbc.g.w_torso, "torso"),
            self.wbc.posture_task(self.q_ref, w=self.w_post),
        ]
        if ref.swing_foot is not None:
            tasks.append(self.wbc.foot_task(ref.swing_foot, ref.swing_p, self.foot_R0[ref.swing_foot],
                                            ref.swing_v, ref.swing_a))
        st = self.wbc.solve(tasks, stance=ref.stance, fz_max_scale=ref.fz_scale)
        sup.check_qp(t, st.ok, st.tau if st.ok else None)
        if sup.mode == Mode.DAMPING:
            return sup.damping_torque(d.qvel[r.act_dof])
        tau = st.tau if st.ok else sup.last_good_tau
        sup.update_thermal(t, r.m.opt.timestep, tau)
        self.log.append(dict(t=t, com=r.com(), com_ref=ref.com.copy(), ok=st.ok,
                             f=None if st.f is None else st.f.copy(), mode=sup.mode.value))
        return tau

    # ------------------------------------------------------------------ DCM balance law
    def dcm_law(self, ref: Reference) -> np.ndarray:
        """DCM feedback with CoP clamping (ankle strategy that never asks the impossible).

        ξ = c + ċ/ω,  p_cmd = ξ_ref − ξ̇_ref/ω + (1 + k_ξ/ω)(ξ − ξ_ref),
        p_cmd is projected into the support polygon shrunk by `cop_margin`,
        then  c̈_xy = ω² (c − p_cmd).
        """
        r, d = self.r, self.r.d
        c = r.com(); cd = r.com_jac() @ d.qvel
        w = np.sqrt(9.81 / max(c[2], 0.3))
        xi = c[:2] + cd[:2] / w
        xi_ref = ref.com[:2] + ref.com_v[:2] / w
        xid_ref = ref.com_v[:2] + ref.com_a[:2] / w
        p = xi_ref - xid_ref / w + (1.0 + self.k_xi / w) * (xi - xi_ref)
        feet = tuple(i for i in (0, 1) if ref.stance[i] and ref.fz_scale[i] > 0.5) or (0, 1)
        hull = r.support_polygon(feet)
        p_cl = project_to_polygon(p, hull, self.cop_margin)
        self.cop_saturated = np.linalg.norm(p_cl - p) > 1e-4
        self.dcm_info = dict(xi=xi, p_cmd=p_cl, p_raw=p)
        return w**2 * (c[:2] - p_cl)

    def _sit_reference(self, t):
        if self.sit_start is None:
            self.sit_start = t
            self.sit_from = self.r.com()
        target = self.sit_from.copy()
        target[:2] = self.c0[:2]
        target[2] = self.c0[2] - 0.18
        p, v, a = blend(self.sit_from, target, t, self.sit_start, 1.5)
        if t - self.sit_start > 1.8:
            self.sup.request(t, Mode.DAMPING, "đã hạ thấp xong → DAMPING")
        return Reference(p, v, a)


# reference
def stand_reference(ctrl: BalanceController):
    
    return lambda t: Reference(ctrl.c0, np.zeros(3), np.zeros(3))


def project_to_polygon(p, hull, margin):
    
    A, b = polygon_halfspaces(hull, margin)
    x = np.array(p, float)
    for _ in range(20):                      # cyclic projections onto violated half-planes
        viol = A @ x - b
        i = int(np.argmax(viol))
        if viol[i] <= 1e-9:
            break
        x = x - viol[i] * A[i]
    return x
