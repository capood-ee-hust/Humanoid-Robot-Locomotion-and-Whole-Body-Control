from __future__ import annotations

import sys
import time
from pathlib import Path

import mujoco
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "common"))
from hmc.robot import G1, load_g1, rot_log   # noqa: E402
from hmc.viz import Recorder                 # noqa: E402
from dcm_planner import swing_trajectory      # noqa: E402

ANKLE_TO_CENTER = 0.035   # tâm tiếp xúc bàn chân G1 nằm trước cổ chân 3.5 cm
FRAME_DT = 0.01           # bước thời gian của các khung hình tính sẵn [s]
KEY_ENTER = 257


def yaw_R(psi):
    c, s = np.cos(psi), np.sin(psi)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


class LegIK:
    def __init__(self, r: G1, q0):
        self.r, self.m = r, r.m
        self.d = mujoco.MjData(r.m)
        self.q0 = q0.copy()
        legs = r.LEG_L + r.LEG_R
        self.cols = np.r_[np.arange(6), [r.act_dof[r.aid(n)] for n in legs]]
        self.leg_qpos = np.array([r.act_qpos[r.aid(n)] for n in legs])
        self.lo = np.array([r.q_lo[r.aid(n)] for n in legs]); self.hi = np.array([r.q_hi[r.aid(n)] for n in legs])

    def solve(self, q, feet, com_xy, psi, iters=60):
        r, d = self.r, self.d
        q = q.copy()
        for _ in range(iters):
            d.qpos[:] = q; mujoco.mj_kinematics(self.m, d); mujoco.mj_comPos(self.m, d)
            e, J = [], []
            for (p, R), b in zip(feet, r.foot_body):
                e += [rot_log(R @ d.xmat[b].reshape(3, 3).T), p - d.xpos[b]]
                J.append(r.body_jac(b, d))
            e.append(com_xy - d.subtree_com[r.pelvis][:2]); J.append(r.com_jac(d)[:2])
            e.append(rot_log(yaw_R(psi) @ d.xmat[r.pelvis].reshape(3, 3).T)); J.append(r.body_jac(r.pelvis, d)[:3])
            e, J = np.concatenate(e), np.vstack(J)[:, self.cols]
            if np.linalg.norm(e) < 1e-5:
                break
            dx = J.T @ np.linalg.solve(J @ J.T + 1e-4 * np.eye(len(e)), e)
            N = np.eye(J.shape[1]) - np.linalg.pinv(J) @ J
            pull = np.zeros(J.shape[1]); pull[6:] = 0.1 * (self.q0[self.leg_qpos] - q[self.leg_qpos])
            step = dx + N @ pull
            nrm = np.linalg.norm(step)
            if nrm > 0.15:                      # trust region: avoids Newton blow-ups near reach limits
                step *= 0.15 / nrm
            dv = np.zeros(self.m.nv); dv[self.cols] = step
            mujoco.mj_integratePos(self.m, q, dv, 1.0)
            q[self.leg_qpos] = np.clip(q[self.leg_qpos], self.lo, self.hi)
        return q, float(np.linalg.norm(e))


# ---------------------------------------------------------------------- cảnh
def _scene_editor(world, steps):
    """Thêm vào cảnh G1: vật cản và 3 chấm mocap (CoM, DCM, ZMP). Không vẽ ô bước chân trên sàn."""
    def edit(spec):
        for k, o in enumerate(world.obstacles if world else []):
            if o.kind == "circle":
                spec.worldbody.add_geom(name=f"obs{k}", type=mujoco.mjtGeom.mjGEOM_CYLINDER, size=[o.p[2], 0.25, 0],
                                        pos=[o.p[0], o.p[1], 0.25], rgba=[0.5, 0.5, 0.5, 1], contype=0, conaffinity=0)
            else:
                x0, y0, x1, y1 = o.p
                spec.worldbody.add_geom(name=f"obs{k}", type=mujoco.mjtGeom.mjGEOM_BOX,
                                        size=[(x1 - x0) / 2, (y1 - y0) / 2, 0.3], pos=[(x0 + x1) / 2, (y0 + y1) / 2, 0.3],
                                        rgba=[0.5, 0.5, 0.5, 1], contype=0, conaffinity=0)
        for name, rgba in (("com_marker", [0.1, 0.8, 0.2, 1]), ("dcm_marker", [0.8, 0.2, 0.8, 1]),
                           ("zmp_marker", [0.05, 0.05, 0.05, 1])):
            b = spec.worldbody.add_body(name=name, mocap=True, pos=[0, 0, 0.004])
            b.add_geom(type=mujoco.mjtGeom.mjGEOM_SPHERE, size=[0.025, 0, 0], rgba=rgba, contype=0, conaffinity=0)
    return edit


def start_qpos(r: G1, steps):
    """Tư thế ban đầu của G1 đặt đúng chỗ hai bàn chân xuất phát steps[0], steps[1].

    Lấy tư thế đứng danh định rồi dời base (x, y) và xoay quanh trục đứng cho khớp với
    điểm giữa và hướng của hai bàn chân xuất phát; IK ở khung hình đầu sẽ chỉnh nốt phần còn lại.
    """
    q = r.nominal_qpos()
    (xl, yl, pl, _), (xr, yr, pr, _) = steps[0], steps[1]
    psi = 0.5 * (pl + pr)
    mid = np.array([(xl + xr) / 2, (yl + yr) / 2])
    q[0:2] = mid - ANKLE_TO_CENTER * np.array([np.cos(psi), np.sin(psi)])
    q[3:7] = [np.cos(psi / 2), 0.0, 0.0, np.sin(psi / 2)]
    return q


def _markers(m):
    ids = {}
    for name in ("com_marker", "dcm_marker", "zmp_marker"):
        b = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, name)
        ids[name] = m.body_mocapid[b]
    return ids


# ---------------------------------------------------------------------- tính trước các khung hình
def compute_frames(steps, tr, dp, r: G1, q_start):
    """Giải IK cho mọi khung hình (mỗi FRAME_DT). Trả dict: t, Q, step, phase, com, dcm, zmp, ik_err."""
    m, d = r.m, r.d
    d.qpos[:] = q_start
    mujoco.mj_forward(m, d)
    ik = LegIK(r, q_start)
    z_foot = [d.xpos[b][2] for b in r.foot_body]

    def ankle(step):
        x, y, psi, _ = step
        return np.array([x - np.cos(psi) * ANKLE_TO_CENTER, y - np.sin(psi) * ANKLE_TO_CENTER])

    n_int = len(tr["zmps"])
    every = max(1, int(round(FRAME_DT / dp.dt)))
    q = q_start.copy()
    out = dict(t=[], Q=[], step=[], phase=[], com=[], dcm=[], zmp=[])
    worst = 0.0
    for k in range(0, len(tr["t"]), every):
        i = int(tr["step"][k])
        t_loc = tr["t"][k] - i * dp.T
        foot_now = {}
        for j in range(min(max(i, 1), len(steps) - 1) + 1):  # bàn chân đã tiếp đất (luôn gồm 2 chân ban đầu)
            foot_now[steps[j][3]] = steps[j]
        feet = {s: (np.r_[ankle(foot_now[s]), z_foot[0 if s > 0 else 1]], yaw_R(foot_now[s][2])) for s in (+1, -1)}
        if 1 <= i <= n_int - 2 and i + 1 < len(steps):
            nxt = steps[i + 1]; side = nxt[3]
            sw = swing_trajectory(ankle(steps[i - 1]), ankle(nxt), t_loc, dp.T, height=0.06)
            psi = steps[i - 1][2] + (nxt[2] - steps[i - 1][2]) * min(t_loc / dp.T, 1)
            feet[side] = (np.r_[sw[:2], z_foot[0 if side > 0 else 1] + sw[2]], yaw_R(psi))
            phase = "tru chan trai" if steps[i][3] > 0 else "tru chan phai"
        else:
            phase = "dung hai chan"
        psi_p = 0.5 * (foot_now[+1][2] + foot_now[-1][2])
        q, err = ik.solve(q, [feet[+1], feet[-1]], tr["com"][k], psi_p)
        worst = max(worst, err)
        out["t"].append(tr["t"][k]); out["Q"].append(q.copy()); out["step"].append(i); out["phase"].append(phase)
        out["com"].append(tr["com"][k]); out["dcm"].append(tr["dcm"][k]); out["zmp"].append(tr["zmp"][k])
    for key in ("t", "Q", "com", "dcm", "zmp"):
        out[key] = np.array(out[key])
    out["ik_err"] = worst
    out["n_steps"] = n_int
    return out


def _apply_frame(m, d, F, j, mk):
    d.qpos[:] = F["Q"][j]
    for name, key in (("com_marker", "com"), ("dcm_marker", "dcm"), ("zmp_marker", "zmp")):
        d.mocap_pos[mk[name]] = [F[key][j][0], F[key][j][1], 0.004]
    mujoco.mj_forward(m, d)


def _joint_deg(r, d, name):
    return float(np.degrees(d.qpos[r.act_qpos[r.aid(name)]]))


# ---------------------------------------------------------------------- xuất GIF
def export_gif(steps, tr, dp, gif, world=None, label="P05 G1"):
    m = load_g1(_scene_editor(world, steps)); r = G1(m); d = r.d
    F = compute_frames(steps, tr, dp, r, start_qpos(r, steps))
    mk = _markers(m)
    rec = Recorder(m, gif, fps=12, width=400, height=300, track_body=r.pelvis, azimuth=120, elevation=-15, distance=3.0)
    for j in range(len(F["t"])):
        _apply_frame(m, d, F, j, mk)
        d.time = F["t"][j]
        rec.capture(d, f"{label}  t={F['t'][j]:.2f}s  buoc {F['step'][j]}")
    rec.save()
    print(f"[G1] sai số IK lớn nhất = {F['ik_err']:.2e} (≈0 nghĩa là mọi tư thế khả thi về động học)")


# ---------------------------------------------------------------------- cửa sổ MuJoCo tương tác
def play_interactive(steps, tr, dp, world=None, label="P05 G1", test_frames: int | None = None,
                     speed: float = 1.0):
    """Mở cửa sổ MuJoCo, phát lại lặp đi lặp lại cho tới khi đóng cửa sổ.

    HƯỚNG DẪN SỬ DỤNG CỬA SỔ (chỉ ghi ở đây, không hiển thị trên màn hình)
      Chuột : kéo chuột trái = xoay camera · kéo chuột phải = dịch camera · lăn chuột = zoom
      Space : tạm dừng / chạy tiếp
      Enter : chạy lại từ đầu
      6 / 7 : phát chậm lại / nhanh lên (0.125x – 4x)
      8     : camera bám theo robot <-> camera tự do
      Góc trên trái: thời gian, bước, pha, tốc độ phát, góc gối/hông; chấm xanh lá = CoM, tím = DCM, đen = ZMP.

    `speed`: tốc độ phát ban đầu (1.0 = thời gian thật).
    `test_frames`: chạy không màn hình N vòng lặp để kiểm tra (không mở cửa sổ).
    """
    m = load_g1(_scene_editor(world, steps)); r = G1(m); d = r.d
    print("[G1] đang tính tư thế cho từng khung hình...")
    F = compute_frames(steps, tr, dp, r, start_qpos(r, steps))
    print(f"[G1] {len(F['t'])} khung hình, sai số IK lớn nhất = {F['ik_err']:.2e}")
    mk = _markers(m)
    T_end = F["t"][-1]
    st = dict(paused=False, restart=False, speed=float(speed), follow=True)

    def on_key(key):
        if key == ord(" "):
            st["paused"] = not st["paused"]
        elif key == KEY_ENTER:
            st["restart"] = True
        elif key == ord("6"):
            st["speed"] = max(0.125, st["speed"] / 2)
        elif key == ord("7"):
            st["speed"] = min(4.0, st["speed"] * 2)
        elif key == ord("8"):
            st["follow"] = not st["follow"]

    if test_frames is None:
        from mujoco import viewer as mjviewer
        v = mjviewer.launch_passive(m, d, key_callback=on_key)
        v.cam.distance, v.cam.elevation, v.cam.azimuth = 3.0, -15, 120
        print("[G1] cửa sổ đã mở — đóng cửa sổ để kết thúc.")
    else:
        v = _DummyViewer(test_frames)

    t_play, t_wall, hold_until = 0.0, time.time(), None
    while v.is_running():
        now = time.time(); dt_wall = now - t_wall; t_wall = now
        if st["restart"]:
            t_play, hold_until, st["restart"] = 0.0, None, False
        elif not st["paused"]:
            if t_play < T_end:
                t_play = min(T_end, t_play + dt_wall * st["speed"])
            elif hold_until is None:
                hold_until = now + 1.5                     # dừng 1.5 s ở cuối rồi lặp lại
            elif now >= hold_until:
                t_play, hold_until = 0.0, None
        j = min(int(round(t_play / FRAME_DT)), len(F["t"]) - 1)
        with v.lock():
            _apply_frame(m, d, F, j, mk)
            d.time = F["t"][j]
        if st["follow"]:
            v.cam.lookat[:] = [d.xpos[r.pelvis][0], d.xpos[r.pelvis][1], 0.6]
        if hasattr(v, "set_texts"):
            v.set_texts([
                (None, mujoco.mjtGridPos.mjGRID_TOPLEFT,
                 "kich ban\nthoi gian\nbuoc\npha\ntoc do phat\ncamera\ngoi trai/phai [deg]\nhong trai/phai [deg]\ncham",
                 f"{label}\n{F['t'][j]:5.2f} / {T_end:.2f} s{'  (tam dung)' if st['paused'] else ''}\n"
                 f"{F['step'][j]} / {F['n_steps'] - 1}\n{F['phase'][j]}\n{st['speed']:.3g}x\n"
                 f"{'bam theo robot' if st['follow'] else 'tu do'}\n"
                 f"{_joint_deg(r, d, 'left_knee_joint'):5.1f} / {_joint_deg(r, d, 'right_knee_joint'):5.1f}\n"
                 f"{_joint_deg(r, d, 'left_hip_pitch_joint'):5.1f} / {_joint_deg(r, d, 'right_hip_pitch_joint'):5.1f}\n"
                 "xanh la = CoM, tim = DCM, den = ZMP")])
            # Hướng dẫn phím/chuột KHÔNG hiển thị trong cửa sổ — xem ghi chú ở docstring của hàm này.
        v.sync()
        time.sleep(1 / 60)
    v.close()


class _DummyViewer:
    """Thay cho cửa sổ thật khi tự kiểm tra không màn hình."""

    def __init__(self, n):
        self.n = n
        self.cam = type("Cam", (), {"lookat": np.zeros(3), "distance": 3.0, "elevation": -15, "azimuth": 120})()

    def is_running(self):
        self.n -= 1
        return self.n >= 0

    def lock(self):
        import contextlib
        return contextlib.nullcontext()

    def set_texts(self, texts):
        assert all(len(x) == 4 for x in texts)

    def sync(self):
        pass

    def close(self):
        pass