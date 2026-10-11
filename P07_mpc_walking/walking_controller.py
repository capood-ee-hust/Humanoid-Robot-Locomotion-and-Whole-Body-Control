from __future__ import annotations

import sys
from pathlib import Path

import mujoco
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "common"))
sys.path.insert(0, str(HERE))
from hmc.qp import DenseQP                                   # noqa: E402
from hmc.robot import G1                                     # noqa: E402
from hmc.safety import Mode, SafetyConfig, SafetySupervisor  # noqa: E402
from hmc.sim import tilt_angle                               # noqa: E402
from hmc.wbc import Task, WBCGains, WholeBodyController      # noqa: E402
from gait import GaitSchedule, ankle_pos, quintic, yaw_R    # noqa: E402
from lipm_mpc import LipmMPC                                 # noqa: E402

SIDE_TO_FOOT = {+1: 0, -1: 1}      # side của bước chân (+1 trái, −1 phải) → chỉ số bàn chân trong G1/WBC


class TolerantQP(DenseQP):
 

    def solve(self, P, q, A, l, u):
        self.solver.update(Px=P[self._pr, self._pc], Ax=A.flatten(order="F"), q=q, l=l, u=u)
        if self.last_x is not None:
            self.solver.warm_start(x=self.last_x)
        res = self.solver.solve()
        self.status = str(res.info.status).lower()
        self.iters = res.info.iter
        self.ok = "solved" in self.status and bool(np.all(np.isfinite(res.x)))
        if self.ok:
            self.last_x = res.x.copy()
        return res.x if self.ok else None


class MPCWalkingController:
    def __init__(self, robot: G1, schedule: GaitSchedule, t_start: float = 0.0,
                 mpc_period: int = 10, gains: WBCGains | None = None, safety_cfg: SafetyConfig | None = None,
                 z_drop: float = 0.05, t_drop: float = 0.8):
        self.r, self.gs, self.t_start = robot, schedule, t_start
        d = robot.d
        # trọng tâm ưu tiên cao (100) để gia tốc thật bám gia tốc MPC yêu cầu; chân đung đưa 30
        self.wbc = WholeBodyController(robot, gains or WBCGains(com_kp=80.0, com_kd=18.0, w_com=100.0, w_foot=30.0))
        self.wbc.qp = TolerantQP(self.wbc.nx, self.wbc.nrow)
        self.q_ref = d.qpos[robot.act_qpos].copy()
        # Hạ trọng tâm z_drop (gối gập thêm) trong t_drop giây đầu rồi giữ KHÔNG ĐỔI (giả thiết LIPM).
        # Chân co hơn → với được xa hơn khi phải bước rộng để chống đẩy (chân duỗi thẳng thì không bước thêm được).
        self.z0 = robot.com()[2]
        self.z_walk, self.t_drop = self.z0 - z_drop, t_drop
        self.mpc = LipmMPC(z_c=self.z_walk)
        self.mpc_period = mpc_period                   # 10 × 2 ms = 20 ms (50 Hz)
        self.w_post = np.full(robot.nu, self.wbc.g.w_post)
        for i, n in enumerate(robot.act_names):
            if "hip" in n or "knee" in n or "ankle" in n:
                self.w_post[i] = 1e-3                  # chân gần như tự do: CoM và bàn chân quyết định
        self.sup = SafetySupervisor(robot.nu, tau_cont=0.6 * robot.tau_max, cfg=safety_cfg or SafetyConfig())
        self.k = 0
        self.p0 = robot.com()[:2].copy()               # ZMP đang áp dụng
        self.mpc_mode = "hard"
        self.upd = None                                # (t_cập nhật, c, ċ) lúc MPC giải gần nhất
        self.swing_offset = None                       # lệch giữa vị trí chân thật lúc nhấc và kế hoạch
        self.swing_phase_id = None
        self.w_angmom, self.angmom_kd = 1.0, 2.0       # task mô-men động lượng (xem _angmom_task)
        if not hasattr(mujoco, "mj_angmomMat"):        # MuJoCo quá cũ (< 3.1): bỏ task này
            self.w_angmom = 0.0
        # Chỉnh THỜI ĐIỂM đặt chân bằng cách cho "đồng hồ dáng đi" ts chạy nhanh / chậm hơn thời gian thật
        # trong pha một chân (xem _update_rate). MPC, chân đung đưa, điểm neo ZMP đều đọc theo ts.
        self.ts = 0.0
        self.rate = 1.0                                # dts/dt hiện tại
        self.rate_max = 1.6                            # nhanh nhất: pha một chân 0.45 s → ~0.28 s
        self.rate_min = 0.7                            # chậm nhất: 0.45 s → ~0.64 s
        self.rate_adapt = (0.08, 0.20)                 # độ dời bước [m]: bắt đầu đổi nhịp / đổi tối đa
        self._t_last = None
        self._rate_phase = None
        self.log = []

    # ------------------------------------------------------------------ helpers
    def com_state(self):
        r = self.r
        c = r.com()
        cd = r.com_jac() @ r.d.qvel
        return c, cd

    def _swing_reference(self, g):
        """Tham chiếu chân đung đưa, bù lệch giữa vị trí chân thật lúc nhấc và kế hoạch (giảm dần về 0)."""
        sw = g["swing"]
        ph = g["phase"]
        foot = SIDE_TO_FOOT[sw["side"]]
        if self.swing_phase_id is not id(ph):          # vừa bắt đầu pha một chân: ghi vị trí chân thật
            self.swing_phase_id = id(ph)
            b = self.r.foot_body[foot]
            self.swing_offset = self.r.d.xpos[b] - ankle_pos(self.gs.steps[ph.swing_from])
        s, ds, dds = quintic(sw["tau"])
        fade = 1.0 - s
        k = self.rate                                  # đạo hàm theo thời gian thật = đạo hàm theo ts × rate
        pos = sw["pos"] + fade * self.swing_offset
        vel = k * (sw["vel"] - (ds / ph.T) * self.swing_offset)
        acc = k**2 * (sw["acc"] - (dds / ph.T**2) * self.swing_offset)
        return foot, pos, vel, acc, sw["R"]

    def _update_rate(self, g):
       
        ph = g["phase"]
        if ph.kind != "SS" or (self.rate_max <= 1.0 and self.rate_min >= 1.0):
            self.rate, self._rate_phase = 1.0, None
            return
        if self._rate_phase is not id(ph):
            self.rate, self._rate_phase = 1.0, id(ph)
        if g["swing"]["tau"] > 0.85:
            return
        j = ph.swing_to
        psi = self.gs.goal[j][2]
        dx, dy = np.subtract(self.gs.steps[j][:2], self.gs.goal[j][:2])
        fwd = abs(np.cos(psi) * dx + np.sin(psi) * dy)                      # dời trước/sau
        out = self.gs.steps[j][3] * (-np.sin(psi) * dx + np.cos(psi) * dy)  # dời RA NGOÀI (>0) / vào trong (<0)
        e0, e1 = self.rate_adapt
        if out < -e0:
            want = 1.0 - (1.0 - self.rate_min) * float(np.clip((-out - e0) / (e1 - e0), 0.0, 1.0))
            self.rate = min(self.rate, want)
        else:
            e = max(fwd, out)
            want = 1.0 + (self.rate_max - 1.0) * float(np.clip((e - e0) / (e1 - e0), 0.0, 1.0))
            self.rate = max(self.rate, want)

    def _angmom_task(self):
        
        r = self.r

        def jac(dd):
            H = np.zeros((3, r.nv))
            mujoco.mj_angmomMat(r.m, dd, H, r.pelvis)
            return H[:2]                               # L̇_z (xoay quanh trục đứng) không ảnh hưởng tâm áp lực
        J = jac(r.d)
        t = Task("angmom", J, None, -self.angmom_kd * (J @ r.d.qvel), self.w_angmom)
        t.jac_fn = jac                                 # để WBC tính đạo hàm J̇·q̇ bằng sai phân
        return t

    # ------------------------------------------------------------------ main
    def compute(self, t: float) -> np.ndarray:
        r, d, sup = self.r, self.r.d, self.sup
        valid = sup.check_sensors(t, d.qpos, d.qvel)
        if valid:
            sup.check_tilt(t, tilt_angle(d, r.pelvis))
        if sup.mode == Mode.ESTOP:
            return np.zeros(r.nu)
        if sup.mode in (Mode.DAMPING, Mode.SAFE_SIT, Mode.DEGRADED):
            if sup.mode != Mode.DAMPING:
                sup.request(t, Mode.DAMPING, "đang đi: chuyển thẳng sang giảm chấn")
            return sup.damping_torque(np.nan_to_num(d.qvel[r.act_dof]))
        if not valid:
            return sup.last_good_tau if sup.last_good_tau is not None else np.zeros(r.nu)

        # ---- đồng hồ dáng đi
        if self._t_last is None:
            self.ts = max(t - self.t_start, 0.0)
        else:
            self.ts += (t - self._t_last) * self.rate
        self._t_last = t
        ts = self.ts
        g = self.gs.query(ts)
        c, cd = self.com_state()

        # MPC: cập nhật mỗi mpc_period bước
        if self.k % self.mpc_period == 0 or self.upd is None:
            self.p0, info = self.mpc.solve(c[:2], cd[:2], ts, self.gs, self.p0)
            self.upd = (t, c[:2].copy(), cd[:2].copy())
            self.mpc_mode = info["mode"]
            self._update_rate(g)
        self.k += 1
        tu, cu, cdu = self.upd
        cr, vr, ar = self.mpc.com_reference(cu, cdu, self.p0, t - tu)

        #  task WBC
        R_yaw = yaw_R(g["pelvis_yaw"])
        s, ds, dds = quintic(ts / self.t_drop)          # hạ trọng tâm êm lúc bắt đầu
        dz = self.z_walk - self.z0
        zr = (self.z0 + dz * s, dz * ds / self.t_drop, dz * dds / self.t_drop**2)
        tasks = [
            self.wbc.com_task(np.r_[cr, zr[0]], np.r_[vr, zr[1]], np.r_[ar, zr[2]]),
            self.wbc.orientation_task(r.pelvis, R_yaw, self.wbc.g.w_pelvis, "pelvis"),
            self.wbc.orientation_task(r.torso, R_yaw, self.wbc.g.w_torso, "torso"),
            self.wbc.posture_task(self.q_ref, w=self.w_post),
        ]
        if self.w_angmom > 0:
            tasks.append(self._angmom_task())
        swing_foot = None
        if g["swing"] is not None:
            swing_foot, sp, sv, sa, sR = self._swing_reference(g)
            tasks.append(self.wbc.foot_task(swing_foot, sp, sR, sv, sa))
        stance = (g["stance"][+1], g["stance"][-1])
        fz = (g["fz_scale"][+1], g["fz_scale"][-1])
        st = self.wbc.solve(tasks, stance=stance, fz_max_scale=fz)
        sup.check_qp(t, st.ok, st.tau if st.ok else None)
        tau = st.tau if st.ok else (sup.last_good_tau if sup.last_good_tau is not None else np.zeros(r.nu))
        sup.update_thermal(t, r.m.opt.timestep, tau)
        self.log.append(dict(t=t, ts=ts, com=c.copy(), com_ref=np.r_[cr, zr[0]], zmp_mpc=self.p0.copy(),
                             zmp_ref=g["zmp_ref"].copy(), kind=g["kind"], qp_ok=st.ok, mode=sup.mode.value,
                             mpc_mode=self.mpc_mode, swing_foot=swing_foot, rate=self.rate,
                             cop=self._measured_cop()))
        return tau

    def _measured_cop(self):
        """Tâm áp lực thật (từ lực tiếp xúc MuJoCo giữa bàn chân và sàn)."""
        m, d, r = self.r.m, self.r.d, self.r
        feet = set(r.foot_geoms[0]) | set(r.foot_geoms[1])
        num, fz_sum = np.zeros(2), 0.0
        f6 = np.zeros(6)
        for i in range(d.ncon):
            con = d.contact[i]
            if con.geom1 in feet or con.geom2 in feet:
                mujoco.mj_contactForce(m, d, i, f6)
                fn = f6[0]                              # lực pháp tuyến trong khung tiếp xúc
                num += fn * con.pos[:2]
                fz_sum += fn
        return num / fz_sum if fz_sum > 1e-6 else np.full(2, np.nan)
