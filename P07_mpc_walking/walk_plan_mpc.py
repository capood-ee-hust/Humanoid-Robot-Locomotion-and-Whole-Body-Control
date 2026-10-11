from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "P05_footstep_dcm"))

from footstep_planner import FootstepPlanner          # noqa: E402
from gait import GaitParams                            # noqa: E402
from mpc_walk_viewer import (G1, WalkSession, feet_centers, load_g1, run_viewer, save_outputs,   # noqa: E402
                             scene_editor, start_qpos, summary)
import run_walk_plan as p05                            # noqa: E402  (OBSTACLES, default_world dùng chung)
import mujoco                                          # noqa: E402


START_X = 0.0          # vị trí bắt đầu 
START_Y = 0.0          
START_YAW_DEG = 0.0    # hướng mặt lúc bắt đầu [độ]
GOAL_X = 3.6           
GOAL_Y = 0.2           
GOAL_YAW_DEG = 0.0     # hướng mặt khi tới đích [độ]
T_SS = 0.45            # thời gian đứng một chân [s]
T_DS = 0.20            # thời gian đứng hai chân [s]
STEP_ADAPT = True      # dời chỗ đặt chân + đổi nhịp (không dời vào vật cản)
PUSH_FORCE = 60.0      # lực một cú đẩy bằng phím 8/9/6/7 [N], kéo dài 0.1 s
PLAYBACK_SPEED = 1.0   # tốc độ chạy so với thời gian thật



def main():
    ap = argparse.ArgumentParser(description="G1 đi theo kế hoạch A* bằng MPC + WBC")
    ap.add_argument("--start", type=float, nargs=3, default=[START_X, START_Y, START_YAW_DEG],
                    metavar=("X", "Y", "YAW_DEG"))
    ap.add_argument("--goal", type=float, nargs=3, default=[GOAL_X, GOAL_Y, GOAL_YAW_DEG],
                    metavar=("X", "Y", "YAW_DEG"))
    ap.add_argument("--t-ss", type=float, default=T_SS)
    ap.add_argument("--t-ds", type=float, default=T_DS)
    ap.add_argument("--no-adapt", action="store_true", help="tắt chỉnh chỗ đặt chân")
    ap.add_argument("--push-force", type=float, default=PUSH_FORCE)
    ap.add_argument("--speed", type=float, default=PLAYBACK_SPEED)
    ap.add_argument("--save", action="store_true", help="xuất GIF + biểu đồ vào out/ thay vì mở cửa sổ")
    ap.add_argument("--push", type=float, nargs=2, metavar=("F", "GOC_DEG"), default=None,
                    help="(--save/--test) một cú đẩy F [N] theo góc so với hướng robot, giữa quãng đi")
    ap.add_argument("--test", action="store_true", help="tự kiểm tra không màn hình")
    a = ap.parse_args()

    sx, sy, syaw = a.start
    gx, gy, gyaw = a.goal
    world = p05.default_world((sx, sy), (gx, gy))
    m = load_g1(scene_editor(world))
    r = G1(m)
    q0 = start_qpos(r, sx, sy, syaw)
    r.d.qpos[:] = q0
    mujoco.mj_forward(m, r.d)
    left, right = feet_centers(r)
    print("[A*] đang tìm chuỗi bước chân...")
    fp = FootstepPlanner(world)
    steps = fp.plan(left, right, (gx, gy, np.radians(gyaw)))
    if steps is None:
        sys.exit(f"A* thất bại (điểm đầu/đích trong vật cản hoặc không có đường): {fp.stats}")
    print(f"[A*] {fp.stats['steps']} bước, mở {fp.stats['expanded']} node, {1000 * fp.stats['time']:.0f} ms")
    gp = GaitParams(t_ss=a.t_ss, t_ds=a.t_ds)
    adapt = STEP_ADAPT and not a.no_adapt
    validator = (lambda x, y, psi: world.foot_free(x, y, psi))
    session = WalkSession(m, steps, gp, q0, validator=validator, adapt=adapt)
    print(f"[MPC walk] {session.gs.T_end:.1f} s, chỉnh bước {'BẬT' if adapt else 'TẮT'}")
    t_push_mid = gp.t_init + 0.5 * (session.gs.T_end - gp.t_init - gp.t_final) + 0.1
    push = (round(t_push_mid / 0.002) * 0.002, *a.push) if a.push else None
    if a.save:
        save_outputs(m, session, HERE / "out", "walk_plan_mpc", "G1 theo A* MPC", world=world, test_push=push)
        print("[out] out/walk_plan_mpc.gif, out/walk_plan_mpc.png |", summary(session))
    elif a.test:
        run_viewer(m, session, "A* + MPC", test_seconds=session.gs.T_end + 1.0, test_push=push)
        print("[test]", summary(session))
    else:
        run_viewer(m, session, "A* + MPC", push_force=a.push_force, speed=a.speed)
        print(summary(session))


if __name__ == "__main__":
    main()
