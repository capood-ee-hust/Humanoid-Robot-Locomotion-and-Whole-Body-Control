from __future__ import annotations

import argparse
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

# In tiếng Việt an toàn cả khi Windows dùng bảng mã cũ
for _s in (sys.stdout, sys.stderr):
    if (getattr(_s, "encoding", "") or "").lower().replace("-", "") != "utf8" and hasattr(_s, "reconfigure"):
        try:
            _s.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

from dcm_planner import DCMPlanner, PreviewController   # noqa: E402
from footstep_planner import World, straight_footsteps  # noqa: E402

import numpy as np  # noqa: E402

# ============================ THAM SỐ — SỬA TRỰC TIẾP Ở ĐÂY ============================
# (tham số dòng lệnh, nếu có, sẽ ghi đè các giá trị này)
START_X = 0.0          # vị trí bắt đầu: điểm giữa hai bàn chân, tọa độ x [m]
START_Y = 0.0          # vị trí bắt đầu: tọa độ y [m]
START_YAW_DEG = 0.0    # hướng mặt lúc bắt đầu [độ]; 0 = nhìn theo trục x, 90 = nhìn theo trục y
DISTANCE = 2.0         # quãng đường đi thẳng [m] → đích = điểm bắt đầu + DISTANCE theo hướng mặt
STEP_LENGTH = 0.20     # chiều dài mỗi bước [m] (nên ≤ 0.30)
STEP_TIME = 1.2        # thời gian mỗi bước [s]: nhỏ hơn = đi nhanh hơn (0.6 = nhanh gấp đôi)
PLAYBACK_SPEED = 1.0   # tốc độ phát ban đầu trong cửa sổ (1.0 = thời gian thật; phím 6/7 vẫn đổi được)
FOOT_HALF_WIDTH = 0.118  # nửa khoảng cách giữa hai bàn chân [m]
# =======================================================================================
# Vận tốc đi trung bình ≈ STEP_LENGTH / STEP_TIME  (mặc định 0.20 / 1.2 ≈ 0.17 m/s khi đang bước)


def start_feet(x, y, yaw_deg, half_w=FOOT_HALF_WIDTH):
    """Hai bàn chân xuất phát (x, y, ψ) đặt hai bên điểm (x, y), vuông góc với hướng mặt."""
    psi = np.deg2rad(yaw_deg)
    left = (x - np.sin(psi) * half_w, y + np.cos(psi) * half_w, psi)
    right = (x + np.sin(psi) * half_w, y - np.cos(psi) * half_w, psi)
    return left, right


def main():
    ap = argparse.ArgumentParser(description="Unitree G1 đi thẳng")
    ap.add_argument("--start", type=float, nargs=3, default=[START_X, START_Y, START_YAW_DEG],
                    metavar=("X", "Y", "YAW_DEG"), help="vị trí bắt đầu x y [m] và hướng [độ]")
    ap.add_argument("--distance", type=float, default=DISTANCE, help="quãng đường [m]")
    ap.add_argument("--step", type=float, default=STEP_LENGTH, help="chiều dài mỗi bước [m]")
    ap.add_argument("--step-time", type=float, default=STEP_TIME, help="thời gian mỗi bước [s]")
    ap.add_argument("--speed", type=float, default=PLAYBACK_SPEED, help="tốc độ phát ban đầu trong cửa sổ")
    ap.add_argument("--save", action="store_true", help="xuất PNG + 2 GIF vào out/ thay vì mở cửa sổ")
    ap.add_argument("--test", action="store_true", help="tự kiểm tra không mở cửa sổ")
    a = ap.parse_args()

    # 1) chuỗi bước chân đi thẳng
    sx, sy, syaw = a.start
    left, right = start_feet(sx, sy, syaw)
    gx = sx + a.distance * np.cos(np.deg2rad(syaw))
    gy = sy + a.distance * np.sin(np.deg2rad(syaw))
    world = World(xlim=(min(sx, gx) - 0.8, max(sx, gx) + 0.8), ylim=(min(sy, gy) - 0.8, max(sy, gy) + 0.8))
    steps = straight_footsteps(left, right, a.distance, a.step, world=world)
    if steps is None:
        sys.exit("Chuỗi bước đi thẳng không hợp lệ")
    print(f"[bước chân] từ ({sx:.2f}, {sy:.2f}) hướng {syaw:.0f}° đi thẳng {a.distance:.2f} m tới ({gx:.2f}, {gy:.2f}): "
          f"{len(steps) - 3} bước dài {a.step:.2f} m + 1 bước khép chân")

    # 2) quỹ đạo trọng tâm theo DCM
    dp = DCMPlanner(z_c=0.66, T_step=a.step_time)
    tr = dp.plan(steps, (sx, sy))                     # trọng tâm bắt đầu ở giữa hai bàn chân
    v_avg = (tr["com"][-1, 0] - tr["com"][0, 0]) / tr["t"][-1]
    print(f"[DCM] {len(tr['zmps'])} pha, tổng {tr['t'][-1]:.1f} s, vận tốc trọng tâm trung bình {v_avg:.2f} m/s")

    # 3) cửa sổ MuJoCo hoặc xuất file
    if a.save:
        from g1_playback import export_gif
        from run_walk_plan import OUT, save_outputs
        OUT.mkdir(exist_ok=True)
        pc = PreviewController(z_c=0.66, dt=dp.dt)
        cx, zx = pc.run(tr["zmp"][:, 0], sx)
        cy, zy = pc.run(tr["zmp"][:, 1], sy)
        title = f"P05 — Đi thẳng {a.distance:.1f} m, bước {a.step:.2f} m (xanh = chân trái, đỏ = chân phải) + mẫu đi DCM"
        save_outputs(world, steps, (gx, gy), title, tr, cx, cy, zx, zy,
                     "walk_plan_straight.png", "walk_topview_straight.gif")
        export_gif(steps, tr, dp, str(OUT / "g1_walk_kinematic_straight.gif"), world=world, label="P05 G1 di thang")
        print("[out] walk_plan_straight.png, walk_topview_straight.gif, g1_walk_kinematic_straight.gif")
    else:
        from g1_playback import play_interactive
        play_interactive(steps, tr, dp, world=world, label="di thang", test_frames=300 if a.test else None,
                         speed=a.speed)


if __name__ == "__main__":
    main()