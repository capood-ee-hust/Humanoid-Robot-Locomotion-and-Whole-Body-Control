from __future__ import annotations

import argparse
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "P05_footstep_dcm"))

from footstep_planner import straight_footsteps     # noqa: E402
from gait import GaitParams                          # noqa: E402
from mpc_walk_viewer import (G1, WalkSession, feet_centers, load_g1, run_viewer, save_outputs,   # noqa: E402
                             start_qpos, summary)
import mujoco                                        # noqa: E402



START_X = 0.0          # vị trí bắt đầu 
START_Y = 0.0         
START_YAW_DEG = 0.0    # hướng mặt lúc bắt đầu [độ]
DISTANCE = 2.0         # quãng đường đi thẳng [m]
STEP_LENGTH = 0.15     # chiều dài mỗi bước 
T_SS = 0.45            # thời gian đứng một chân [s] 
T_DS = 0.20            # thời gian đứng hai chân [s]
STEP_ADAPT = True      # dời chỗ đặt chân + đổi nhịp 
PUSH_FORCE = 60.0      # lực một cú đẩy bằng phím 8/9/6/7 [N], kéo dài 0.1 s
PLAYBACK_SPEED = 1.0   # tốc độ chạy so với thời gian thật 



def main():
    ap = argparse.ArgumentParser(description="G1 đi thẳng bằng MPC + WBC")
    ap.add_argument("--start", type=float, nargs=3, default=[START_X, START_Y, START_YAW_DEG],
                    metavar=("X", "Y", "YAW_DEG"))
    ap.add_argument("--distance", type=float, default=DISTANCE)
    ap.add_argument("--step", type=float, default=STEP_LENGTH)
    ap.add_argument("--t-ss", type=float, default=T_SS)
    ap.add_argument("--t-ds", type=float, default=T_DS)
    ap.add_argument("--no-adapt", action="store_true", help="tắt chỉnh chỗ đặt chân")
    ap.add_argument("--push-force", type=float, default=PUSH_FORCE)
    ap.add_argument("--speed", type=float, default=PLAYBACK_SPEED)
    ap.add_argument("--save", action="store_true", help="xuất GIF + biểu đồ vào out/ thay vì mở cửa sổ")
    ap.add_argument("--push", type=float, nargs=2, metavar=("F", "GOC_DEG"), default=None,
                    help="(--save/--test) một cú đẩy F [N] theo góc so với hướng robot, giữa quãng đi")
    ap.add_argument("--test", action="store_true", help="tự kiểm tra không màn hình (có một cú đẩy 60 N sang ngang)")
    a = ap.parse_args()

    m = load_g1()
    r = G1(m)
    q0 = start_qpos(r, *a.start)
    r.d.qpos[:] = q0
    mujoco.mj_forward(m, r.d)
    left, right = feet_centers(r)
    steps = straight_footsteps(left, right, a.distance, a.step)
    gp = GaitParams(t_ss=a.t_ss, t_ds=a.t_ds)
    adapt = STEP_ADAPT and not a.no_adapt
    session = WalkSession(m, steps, gp, q0, adapt=adapt)
    n = len(steps) - 3
    print(f"[MPC walk] đi thẳng {a.distance:.2f} m: {n} bước {a.step:.2f} m + 1 bước khép chân, "
          f"{session.gs.T_end:.1f} s, chỉnh bước {'BẬT' if adapt else 'TẮT'}")
    t_push_mid = gp.t_init + 0.5 * (session.gs.T_end - gp.t_init - gp.t_final) + 0.1
    push = (round(t_push_mid / 0.002) * 0.002, *a.push) if a.push else None
    if a.save:
        save_outputs(m, session, HERE / "out", "walk_straight_mpc", "G1 di thang MPC", test_push=push)
        print("[out] out/walk_straight_mpc.gif, out/walk_straight_mpc.png |", summary(session))
    elif a.test:
        push = push or (round(t_push_mid / 0.002) * 0.002, 60.0, 90.0)
        run_viewer(m, session, "di thang MPC", test_seconds=session.gs.T_end + 1.0, test_push=push)
        print(f"[test] đẩy {push[1]:.0f} N, góc {push[2]:.0f}° lúc t={push[0]:.2f} s →", summary(session))
    else:
        run_viewer(m, session, "di thang MPC", push_force=a.push_force, speed=a.speed)
        print(summary(session))


if __name__ == "__main__":
    main()
