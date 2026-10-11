from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt   # noqa: E402
import mujoco                     # noqa: E402
import numpy as np                # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "common"))
from hmc.robot import G1, load_g1                          # noqa: E402
from hmc.safety import Mode                                # noqa: E402
from hmc.sim import tilt_angle                             # noqa: E402
from balance_controller import BalanceController, stand_reference  # noqa: E402

KEY_ENTER = 257
PUSH_KEYS = {ord("8"): (0.0, "trước"), ord("9"): (180.0, "sau"), ord("6"): (90.0, "trái"), ord("7"): (270.0, "phải")}
RECOVER_DELAY = 2.0        # s nằm ở DAMPING trước khi tự đứng lại
PUSH_DUR = 0.1


class StandingRobot:

    def __init__(self, m, controller: str):
        self.m, self.kind = m, controller
        self.r = G1(m)
        self.d = self.r.d
        self.q0 = self.r.nominal_qpos()
        self.reset(t_keep=0.0)

    def reset(self, t_keep: float):
        d = self.d
        mujoco.mj_resetData(self.m, d)
        d.qpos[:] = self.q0
        d.time = t_keep                      # keep a monotonic clock for logs/pacing
        mujoco.mj_forward(self.m, d)
        self.ctrl = None
        if self.kind in ("wbc", "dcm"):
            self.ctrl = BalanceController(self.r)
            self.ctrl.use_dcm = self.kind == "dcm"
            self.ctrl.reference_fn = stand_reference(self.ctrl)
        self.fallen_since = None

    @property
    def mode(self) -> str:
        if self.ctrl is not None:
            return self.ctrl.sup.mode.value
        return "DAMPING" if self.fallen() else "STAND"

    def fallen(self) -> bool:
        return np.degrees(tilt_angle(self.d, self.r.pelvis)) > 25 or self.d.qpos[2] < 0.45

    def torque(self):
        r, d = self.r, self.d
        if self.ctrl is not None:
            tau = self.ctrl.compute(d.time)
            self.ctrl.sup.last_cmd_time = d.time
            return tau
        # baseline: stiff joint PD towards the nominal posture (≈ Menagerie position actuators)
        if self.fallen():
            return -4.0 * d.qvel[r.act_dof]
        tau = 300.0 * (self.q0[r.act_qpos] - d.qpos[r.act_qpos]) - 12.0 * d.qvel[r.act_dof]
        return np.clip(tau, -r.tau_max, r.tau_max)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--controller", choices=["wbc", "dcm", "pd"], default="wbc")
    ap.add_argument("--push", type=float, default=100.0, help="lực của phím đẩy [N], kéo dài 0.1 s")
    ap.add_argument("--test", action="store_true",
                    help="tự kiểm tra không cần màn hình: mô phỏng 'chuột' giữ 30 N rồi đẩy 400 N")
    a = ap.parse_args()

    m = load_g1()
    bot = StandingRobot(m, a.controller)
    d = bot.d
    pelvis = bot.r.pelvis
    state = dict(paused=False, reset=False, push=None)     # shared with the key callback

    def on_key(keycode: int):
        if keycode == ord(" "):
            state["paused"] = not state["paused"]
        elif keycode == KEY_ENTER:
            state["reset"] = True
        elif keycode in PUSH_KEYS:
            state["push"] = PUSH_KEYS[keycode]

    log = dict(t=[], f_user=[], f_key=[], tilt=[], com_err=[], mode=[])
    falls, max_survived, last_fall_force = [], 0.0, None
    peak_since_stand = 0.0
    last_force_t = -1e9
    key_push = None                                          # (t_end, force vector)
    fmt = lambda x: f"{x:6.1f}"                             # noqa: E731

    viewer = None
    if not a.test:
        from mujoco import viewer as mjviewer        # (không dùng `import mujoco.viewer` trong hàm)
        viewer = mjviewer.launch_passive(m, d, key_callback=on_key)
        viewer.cam.distance, viewer.cam.elevation, viewer.cam.azimuth = 3.0, -15, 150
    print(f"[interactive] controller = {a.controller}. Đóng cửa sổ để kết thúc." if viewer else
          "[interactive --test] chạy kiểm tra không màn hình...")

    t_wall0, sim0 = time.time(), d.time
    last_t, k = d.time, 0
    user_force = np.zeros((m.nbody, 6))
    T_TEST = 9.0
    try:
        while (viewer is None and d.time < T_TEST) or (viewer is not None and viewer.is_running()):
            # ---------------- UI-requested reset (our Enter key, or MuJoCo's own Reset/Backspace)
            if state["reset"] or d.time < last_t - 1e-9:
                state["reset"] = False
                bot.reset(t_keep=last_t)
                peak_since_stand = 0.0
                print(f"  t={d.time:6.2f}s  ↺ đặt lại tư thế đứng")
            if state["paused"]:
                if viewer is not None:
                    viewer.sync(); time.sleep(0.02)
                continue

            # ---------------- forces: viewer perturbation (written into xfrc_applied at sync) + key pushes
            if state["push"] is not None:
                ang, name = state["push"]; state["push"] = None
                th = np.deg2rad(ang)
                key_push = (d.time + PUSH_DUR, a.push * np.array([np.cos(th), np.sin(th), 0.0]))
                print(f"  t={d.time:6.2f}s  → phím đẩy {a.push:.0f} N về phía {name}")
            if a.test:                                       # "chuột" giả lập: lực nhỏ (đứng được), rồi lực lớn (ngã)
                user_force[:] = 0
                if 1.0 <= d.time < 1.5:
                    user_force[pelvis, :3] = [30.0, 0, 0]
                if 3.0 <= d.time < 3.3:
                    user_force[pelvis, :3] = [0, 400.0, 0]
            d.xfrc_applied[:] = user_force
            f_key = np.zeros(3)
            if key_push is not None and d.time < key_push[0]:
                f_key = key_push[1]
                d.xfrc_applied[pelvis, :3] += f_key

            # control step (mj_step1 → controller → mj_step2)
            mujoco.mj_step1(m, d)
            d.ctrl[:] = bot.torque()
            mujoco.mj_step2(m, d)
            last_t = d.time
            k += 1

            # bookkeeping
            f_user = float(np.linalg.norm(user_force[:, :3], axis=1).max())
            f_tot = f_user + float(np.linalg.norm(f_key))
            mode = bot.mode
            if f_tot > 0:
                last_force_t = d.time
            if mode == "STAND":
                peak_since_stand = max(peak_since_stand, f_tot)
                # chỉ tính là "chịu được" khi đã hết lực và robot vẫn đứng thêm 1 s
                if f_tot == 0 and peak_since_stand > 0 and d.time - last_force_t > 1.0:
                    max_survived = max(max_survived, peak_since_stand)
            if mode in ("DAMPING", "ESTOP") or (bot.ctrl is None and bot.fallen()):
                if bot.fallen_since is None:
                    bot.fallen_since = d.time
                    last_fall_force = peak_since_stand
                    falls.append((d.time, peak_since_stand))
                    print(f"  t={d.time:6.2f}s  ✗ NGÃ (lực lớn nhất trước khi ngã ≈ {peak_since_stand:.0f} N) "
                          f"→ tự đứng lại sau {RECOVER_DELAY:.0f} s")
                elif d.time - bot.fallen_since > RECOVER_DELAY:
                    state["reset"] = True
            com_err = 0.0 if bot.ctrl is None else 1000 * float(np.linalg.norm(bot.r.com() - bot.ctrl.c0))
            log["t"].append(d.time); log["f_user"].append(f_user); log["f_key"].append(float(np.linalg.norm(f_key)))
            log["tilt"].append(np.degrees(tilt_angle(d, pelvis))); log["com_err"].append(com_err); log["mode"].append(mode)

            # viewer: sync (this also writes the NEW mouse force into xfrc_applied)
            if viewer is not None and k % 8 == 0:
                if hasattr(viewer, "set_texts"):
                    viewer.set_texts([(None, mujoco.mjtGridPos.mjGRID_TOPLEFT,
                                       "Controller\nMode\nFm [N]\nFk [N]\nNghieng [deg]\nLech CoM [mm]\n"
                                       "F max [N]\nFalls",
                                       f"{a.controller}\n{mode}\n{fmt(f_user)}\n{fmt(np.linalg.norm(f_key))}\n"
                                       f"{fmt(log['tilt'][-1])}\n{fmt(com_err)}\n{fmt(max_survived)}\n{len(falls)}")])
                    
                viewer.sync()
                user_force = d.xfrc_applied.copy()            # viewer just wrote the perturbation here
                # pace to (at most) real time
                lag = (d.time - sim0) - (time.time() - t_wall0)
                if lag > 0:
                    time.sleep(lag)
    except KeyboardInterrupt:
        pass
    finally:
        if viewer is not None:
            viewer.close()

    # ---------------- summary
    print("\n=== Tóm tắt ===")
    print(f"Controller: {a.controller} | thời gian mô phỏng {d.time:.1f} s | số lần ngã: {len(falls)}")
    print(f"Lực lớn nhất robot đã chịu được mà vẫn đứng: ≈ {max_survived:.0f} N")
    for t, f in falls:
        print(f"  ngã lúc t={t:.2f}s sau lực ≈ {f:.0f} N")
    if log["t"]:
        L = {kk: np.array(v) for kk, v in log.items() if kk != "mode"}
        fig, ax = plt.subplots(3, 1, figsize=(9, 7), sharex=True)
        ax[0].plot(L["t"], L["f_user"], label="lực chuột"); ax[0].plot(L["t"], L["f_key"], label="lực phím")
        ax[0].set_ylabel("lực [N]"); ax[0].legend(); ax[0].grid(alpha=.3)
        ax[1].plot(L["t"], L["tilt"]); ax[1].axhline(25, color="r", ls=":"); ax[1].set_ylabel("nghiêng [°]"); ax[1].grid(alpha=.3)
        ax[2].plot(L["t"], L["com_err"]); ax[2].set_ylabel("lệch CoM [mm]"); ax[2].set_xlabel("t [s]"); ax[2].grid(alpha=.3)
        for t, _ in falls:
            for x in ax:
                x.axvline(t, color="k", ls="--", lw=.8)
        fig.suptitle(f"P06 interactive — {a.controller} (đường nét đứt = lúc ngã)")
        fig.tight_layout(); (HERE / "out").mkdir(exist_ok=True)
        fig.savefig(HERE / "out" / "interactive.png", dpi=110); plt.close(fig)
        print("Đã lưu out/interactive.png")


if __name__ == "__main__":
    main()