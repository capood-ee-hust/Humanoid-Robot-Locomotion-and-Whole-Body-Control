from __future__ import annotations

import sys
import time
from pathlib import Path

import mujoco
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "common"))
sys.path.insert(0, str(HERE))

from hmc.robot import G1, load_g1          # noqa: E402,F401  (load_g1 dùng lại trong 2 script chạy)
from hmc.sim import tilt_angle             # noqa: E402
from gait import ANKLE_TO_CENTER, GaitParams, GaitSchedule, sole_corners   # noqa: E402
from walking_controller import MPCWalkingController   # noqa: E402

# In tiếng Việt an toàn cả khi Windows dùng bảng mã cũ
for _s in (sys.stdout, sys.stderr):
    if (getattr(_s, "encoding", "") or "").lower().replace("-", "") != "utf8" and hasattr(_s, "reconfigure"):
        try:
            _s.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

KEY_ENTER = 257
PUSH_KEYS = {ord("8"): 0.0, ord("9"): 180.0, ord("6"): 90.0, ord("7"): 270.0}   # góc so với hướng robot
PUSH_DUR = 0.1
RECOVER_DELAY = 2.0
HOLD_AT_END = 1.5


# ---------------------------------------------------------------------- dựng mô hình và tư thế xuất phát
def scene_editor(world):
    """Thêm vật cản (nếu có) vào cảnh G1 — chỉ để nhìn, robot không va chạm với chúng."""
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
    return edit


def start_qpos(r: G1, x: float, y: float, yaw_deg: float) -> np.ndarray:
    """Tư thế đứng chuẩn, dời hông tới (x, y) và xoay mặt yaw_deg độ."""
    q = r.nominal_qpos()
    psi = np.radians(yaw_deg)
    q[0], q[1] = x, y
    q[3:7] = [np.cos(psi / 2), 0.0, 0.0, np.sin(psi / 2)]
    return q


def feet_centers(r: G1):
    """Tâm đế hai bàn chân (x, y, ψ) đọc từ mô hình — dùng làm 2 bước xuất phát."""
    out = []
    for f in (0, 1):
        b = r.foot_body[f]
        R = r.d.xmat[b].reshape(3, 3)
        psi = float(np.arctan2(R[1, 0], R[0, 0]))
        p = r.d.xpos[b][:2] + ANKLE_TO_CENTER * np.array([np.cos(psi), np.sin(psi)])
        out.append((float(p[0]), float(p[1]), psi))
    return out[0], out[1]


class WalkSession:
   

    def __init__(self, m, steps, gait_params: GaitParams, q_start, validator=None, adapt=True):
        self.m, self.steps0, self.gp, self.q_start = m, list(steps), gait_params, q_start
        self.validator, self.adapt = validator, adapt
        self.r = G1(m)
        self.d = self.r.d
        self.falls = 0
        self.reset(0.0)

    def reset(self, t_keep: float):
        m, d = self.m, self.d
        mujoco.mj_resetData(m, d)
        d.qpos[:] = self.q_start
        d.time = t_keep
        mujoco.mj_forward(m, d)
        self.gs = GaitSchedule(list(self.steps0), self.gp)
        self.ctrl = MPCWalkingController(self.r, self.gs, t_start=t_keep)
        self.ctrl.mpc.adapt = self.adapt
        self.ctrl.mpc.validator = self.validator
        self.fallen_since = None
        self.end_since = None

    def step(self):
        
        m, d = self.m, self.d
        mujoco.mj_step1(m, d)
        d.ctrl[:] = self.ctrl.compute(d.time)
        mujoco.mj_step2(m, d)
        if self.ctrl.sup.mode.value != "STAND":
            if self.fallen_since is None:
                self.fallen_since = d.time
                self.falls += 1
            return d.time - self.fallen_since > RECOVER_DELAY
        if self.ctrl.ts > self.gs.T_end:                 # đồng hồ dáng đi (có thể nhanh/chậm hơn thời gian thật)
            if self.end_since is None:
                self.end_since = d.time
            return d.time - self.end_since > HOLD_AT_END
        return False

    def heading(self) -> float:
        R = self.d.xmat[self.r.pelvis].reshape(3, 3)
        return float(np.arctan2(R[1, 0], R[0, 0]))

    def max_adapt_cm(self) -> float:
        return 100 * max((np.hypot(s[0] - s0[0], s[1] - s0[1]) for s, s0 in zip(self.gs.steps, self.gs.steps_ref)),
                         default=0.0)


# ---------------------------------------------------------------------- cửa sổ tương tác
def run_viewer(m, session: WalkSession, label: str, push_force: float = 60.0, speed: float = 1.0,
               test_seconds: float | None = None, test_push=None):
    
    d, r = session.d, session.r
    pelvis = r.pelvis
    st = dict(paused=False, reset=False, push=None, follow=True)

    def on_key(key):
        if key == ord(" "):
            st["paused"] = not st["paused"]
        elif key == KEY_ENTER:
            st["reset"] = True
        elif key in PUSH_KEYS:
            st["push"] = PUSH_KEYS[key]
        elif key == ord("0"):
            st["follow"] = not st["follow"]

    viewer = None
    if test_seconds is None:
        from mujoco import viewer as mjviewer
        viewer = mjviewer.launch_passive(m, d, key_callback=on_key)
        viewer.cam.distance, viewer.cam.elevation, viewer.cam.azimuth = 3.2, -15, 135
        print(f"[{label}] cửa sổ đã mở — đóng cửa sổ để kết thúc.")
    user_force = np.zeros((m.nbody, 6))
    key_push = None
    t_wall0, t_sim0 = time.time(), d.time
    k = 0
    peak_force = 0.0
    try:
        while (viewer is None and d.time < test_seconds) or (viewer is not None and viewer.is_running()):
            if st["reset"]:
                st["reset"] = False
                session.reset(d.time)
                key_push = None
                print(f"  t={d.time:6.2f}s  ↺ đặt lại, đi lại từ đầu")
            if st["paused"]:
                if viewer is not None:
                    viewer.sync()
                    time.sleep(0.02)
                    t_wall0, t_sim0 = time.time(), d.time
                continue
            if st["push"] is not None:
                ang = session.heading() + np.radians(st["push"])
                key_push = (d.time + PUSH_DUR, push_force * np.array([np.cos(ang), np.sin(ang), 0.0]))
                print(f"  t={d.time:6.2f}s  → phím đẩy {push_force:.0f} N")
                st["push"] = None
            if test_push is not None:                     # cú đẩy lập trình sẵn (chế độ --test)
                t0, F, ang_deg = test_push
                if abs(d.time - t0) < 1e-6:
                    ang = session.heading() + np.radians(ang_deg)
                    key_push = (d.time + PUSH_DUR, F * np.array([np.cos(ang), np.sin(ang), 0.0]))
            d.xfrc_applied[:] = user_force
            f_key = np.zeros(3)
            if key_push is not None and d.time < key_push[0]:
                f_key = key_push[1]
                d.xfrc_applied[pelvis, :3] += f_key
            f_tot = float(np.linalg.norm(user_force[:, :3], axis=1).max() + np.linalg.norm(f_key))
            peak_force = max(peak_force, f_tot)
            was_standing = session.ctrl.sup.mode.value == "STAND"
            need_reset = session.step()
            if was_standing and session.ctrl.sup.mode.value != "STAND":
                print(f"  t={d.time:6.2f}s  ✗ NGÃ → tự đặt lại sau {RECOVER_DELAY:.0f} s")
            if need_reset:
                st["reset"] = True
            k += 1
            if viewer is not None and k % 8 == 0:
                if st["follow"]:
                    viewer.cam.lookat[:] = [d.xpos[pelvis][0], d.xpos[pelvis][1], 0.6]
                ts = session.ctrl.ts
                q = session.gs.phase_at(ts)
                n_steps = len(session.gs.steps) - 2
                done = sum(1 for j, tl in session.gs.land_time.items() if j >= 2 and tl <= ts)
                viewer.set_texts([(None, mujoco.mjtGridPos.mjGRID_TOPLEFT,
                                   "Mode\nTime\nStep\npha\nStep\nController\nMPC\nNext step max [cm]\n"
                                   "Fm / Fk [N]\nNghieng [deg]\nFalls",
                                   f"{label}\n{ts:5.2f} / {session.gs.T_end:.2f} s\n{done} / {n_steps}\n"
                                   f"{'mot chan' if q.kind == 'SS' else 'hai chan'}\nx{session.ctrl.rate:.2f}\n"
                                   f"{session.ctrl.sup.mode.value}\n"
                                   f"{session.ctrl.mpc_mode}\n{session.max_adapt_cm():5.1f}\n"
                                   f"{np.linalg.norm(user_force[:, :3], axis=1).max():5.0f} / {np.linalg.norm(f_key):5.0f}\n"
                                   f"{np.degrees(tilt_angle(d, pelvis)):5.1f}\n{session.falls}")])
                d.xfrc_applied[:] = 0                     # xóa lực cũ (kể cả cú đẩy phím) trước khi viewer ghi lực chuột
                viewer.sync()
                user_force = d.xfrc_applied.copy()        # viewer vừa ghi lực chuột vào đây
                lag = (d.time - t_sim0) / speed - (time.time() - t_wall0)
                if lag > 0:
                    time.sleep(lag)
    except KeyboardInterrupt:
        pass
    finally:
        if viewer is not None:
            viewer.close()
    return session


# ---------------------------------------------------------------------- xuất file
def save_outputs(m, session: WalkSession, out_dir: Path, name: str, title: str, world=None, test_push=None):
    """Chạy một lượt không màn hình, ghi GIF + biểu đồ (bước chân kế hoạch vs MPC chỉnh, CoM, ZMP)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from hmc.viz import Recorder
    out_dir.mkdir(exist_ok=True)
    d, r = session.d, session.r
    rec = Recorder(m, str(out_dir / f"{name}.gif"), fps=15, width=400, height=300, track_body=r.pelvis)
    T = session.gs.T_end + 1.0
    key_push = None
    while d.time < T:
        d.xfrc_applied[:] = 0
        if test_push is not None:
            t0, F, ang_deg = test_push
            if abs(d.time - t0) < 1e-6:
                ang = session.heading() + np.radians(ang_deg)
                key_push = (d.time + PUSH_DUR, F * np.array([np.cos(ang), np.sin(ang), 0.0]))
            if key_push is not None and d.time < key_push[0]:
                d.xfrc_applied[r.pelvis, :3] = key_push[1]
        session.step()
        rec.capture(d, f"{title}  t={d.time:.2f}s  {session.ctrl.sup.mode.value}")
        if session.ctrl.sup.mode.value != "STAND" and d.time - session.fallen_since > 1.0:
            break
    rec.save()
    L = session.ctrl.log
    t = np.array([e["t"] for e in L])
    com = np.array([e["com"] for e in L])
    zmp = np.array([e["zmp_mpc"] for e in L])
    cop = np.array([e["cop"] for e in L])
    fig = plt.figure(figsize=(13, 8))
    ax = fig.add_subplot(2, 1, 1)
    if world is not None:
        from matplotlib.patches import Circle, Rectangle
        for o in world.obstacles:
            if o.kind == "circle":
                ax.add_patch(Circle(o.p[:2], o.p[2], color="0.6"))
            else:
                ax.add_patch(Rectangle((o.p[0], o.p[1]), o.p[2] - o.p[0], o.p[3] - o.p[1], color="0.6"))
    for k_, (s, s0) in enumerate(zip(session.gs.steps, session.gs.steps_ref)):
        col = "tab:blue" if s[3] > 0 else "tab:red"
        c0 = sole_corners(s0)
        ax.fill(c0[:, 0], c0[:, 1], fc="none", ec=col, ls=":", lw=0.8)
        c1 = sole_corners(s)
        ax.fill(c1[:, 0], c1[:, 1], fc=col, alpha=0.25, ec=col, lw=1.2)
    ax.plot(zmp[:, 0], zmp[:, 1], "k-", lw=0.7, label="ZMP do MPC chọn")
    ax.plot(com[:, 0], com[:, 1], "g-", lw=2, label="CoM thật")
    ax.set_aspect("equal")
    ax.grid(alpha=0.3)
    ax.legend(loc="upper left", fontsize=8)
    ax.set_title(f"{title} — nét chấm = bước kế hoạch, tô màu = bước thật sau khi MPC chỉnh "
                 f"(xanh = trái, đỏ = phải); dời lớn nhất {session.max_adapt_cm():.1f} cm")
    for i_, lab in enumerate(("x", "y")):
        a2 = fig.add_subplot(2, 2, 3 + i_)
        a2.plot(t, com[:, i_], "g", lw=2, label="CoM")
        a2.plot(t, zmp[:, i_], "k", lw=0.8, label="ZMP MPC")
        a2.plot(t, cop[:, i_], "c", lw=0.6, alpha=0.7, label="CoP đo")
        if test_push is not None:
            a2.axvspan(test_push[0], test_push[0] + PUSH_DUR, color="orange", alpha=0.3, label="cú đẩy")
        a2.set_xlabel("t [s]")
        a2.set_ylabel(f"{lab} [m]")
        a2.grid(alpha=0.3)
        a2.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out_dir / f"{name}.png", dpi=110)
    plt.close(fig)
    return session


def summary(session: WalkSession) -> str:
    L = session.ctrl.log
    tilt = np.degrees(tilt_angle(session.d, session.r.pelvis))
    soft = sum(1 for e in L if e["mpc_mode"] != "hard")
    return (f"chế độ cuối {session.ctrl.sup.mode.value}, số lần ngã {session.falls}, "
            f"CoM cuối {np.round(session.r.com()[:2], 3)}, nghiêng cuối {tilt:.1f}°, "
            f"dời bước lớn nhất {session.max_adapt_cm():.1f} cm, bước MPC chế độ mềm {soft // 10}")
