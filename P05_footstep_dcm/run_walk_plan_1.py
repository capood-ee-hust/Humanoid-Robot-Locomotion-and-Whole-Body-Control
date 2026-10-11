from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                  # noqa: E402
from matplotlib.animation import FuncAnimation, PillowWriter  # noqa: E402
from matplotlib.patches import Circle, Polygon, Rectangle      # noqa: E402
import numpy as np                               # noqa: E402

HERE = Path(__file__).resolve().parent

# In tiếng Việt an toàn cả khi Windows dùng bảng mã cũ (xem common/hmc/__init__.py)
for _s in (sys.stdout, sys.stderr):
    if (getattr(_s, "encoding", "") or "").lower().replace("-", "") != "utf8" and hasattr(_s, "reconfigure"):
        try:
            _s.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass
sys.path.insert(0, str(HERE))
from dcm_planner import DCMPlanner, PreviewController, swing_trajectory  # noqa: E402
from footstep_planner import FootstepPlanner, Obstacle, World, foot_corners  # noqa: E402

OUT = HERE / "out"

# ============================ THAM SỐ — SỬA TRỰC TIẾP Ở ĐÂY ============================
# (tham số dòng lệnh, nếu có, sẽ ghi đè các giá trị này)
START_X = 0.0          # vị trí bắt đầu: điểm giữa hai bàn chân, tọa độ x [m]
START_Y = 0.0          # vị trí bắt đầu: tọa độ y [m]
START_YAW_DEG = 0.0    # hướng mặt lúc bắt đầu [độ]; 0 = nhìn theo trục x
GOAL_X = 3.6           # đích đến: tọa độ x [m]
GOAL_Y = 0.2           # đích đến: tọa độ y [m]
GOAL_YAW_DEG = 0.0     # hướng mặt khi tới đích [độ]
STEP_TIME = 1.2        # thời gian mỗi bước [s]: nhỏ hơn = đi nhanh hơn (0.6 = nhanh gấp đôi)
PLAYBACK_SPEED = 1.0   # tốc độ phát ban đầu trong cửa sổ (1.0 = thời gian thật; phím 6/7 vẫn đổi được)
FOOT_HALF_WIDTH = 0.118  # nửa khoảng cách giữa hai bàn chân [m]

# VẬT CẢN — thêm / xóa / sửa từng dòng:
#   ("circle", (tâm_x, tâm_y, bán_kính))           hình trụ tròn
#   ("rect",   (x_min, y_min, x_max, y_max))       khối hộp chữ nhật
# Điểm bắt đầu và đích phải cách vật cản ít nhất ~0.15 m (bàn chân dài 20 cm + khoảng hở 4 cm).
OBSTACLES = [
    ("circle", (1.3, 0.1, 0.35)),        # trụ tròn lớn giữa đường
    ("rect",   (2.4, -1.5, 2.7, -0.2)),  # tường dưới  ┐ chừa khe y ∈ (−0.2, 0.5)
    ("rect",   (2.4, 0.5, 2.7, 1.5)),    # tường trên  ┘
    ("circle", (0.7, -0.55, 0.20)),      # trụ nhỏ bên phải lúc xuất phát
    ("rect",   (0.9, 0.75, 1.4, 1.0)),   # hộp thấp bên trái
    ("rect",   (1.8, 0.45, 2.05, 1.1)),  # hộp chắn phía trên trước khe tường
    ("circle", (3.2, -0.35, 0.18)),      # trụ nhỏ sau khe tường
]
# Chiều dài bước của A*: danh sách dxs và max_step trong FootstepPlanner.__init__ (footstep_planner.py).
# =======================================================================================


def default_world(start_xy=(START_X, START_Y), goal_xy=(GOAL_X, GOAL_Y)):
    """Bản đồ: các vật cản trong OBSTACLES; biên bản đồ tự nới ra để chứa điểm bắt đầu, đích và vật cản."""
    xs = [start_xy[0], goal_xy[0]]
    ys = [start_xy[1], goal_xy[1]]
    for kind, p in OBSTACLES:
        if kind == "circle":
            xs += [p[0] - p[2], p[0] + p[2]]; ys += [p[1] - p[2], p[1] + p[2]]
        else:
            xs += [p[0], p[2]]; ys += [p[1], p[3]]
    return World(xlim=(min(-0.5, min(xs) - 0.7), max(4.5, max(xs) + 0.9)),
                 ylim=(min(-1.5, min(ys) - 1.0), max(1.5, max(ys) + 1.0)),
                 obstacles=[Obstacle(kind, tuple(p)) for kind, p in OBSTACLES])


def draw_world(ax, w: World):
    for o in w.obstacles:
        if o.kind == "circle":
            ax.add_patch(Circle(o.p[:2], o.p[2], color="gray", alpha=.6))
        else:
            x0, y0, x1, y1 = o.p
            ax.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, color="gray", alpha=.6))
    ax.set_xlim(*w.xlim); ax.set_ylim(*w.ylim); ax.set_aspect("equal"); ax.grid(alpha=.3)


def draw_steps(ax, steps, alpha=0.8):
    for k, (x, y, psi, side) in enumerate(steps):
        ax.add_patch(Polygon(foot_corners(x, y, psi), closed=True, fill=False,
                             ec="tab:blue" if side > 0 else "tab:red", lw=1.2, alpha=alpha))
        ax.text(x, y, str(k), fontsize=6, ha="center", va="center")


def start_feet(x, y, yaw_deg, half_w=FOOT_HALF_WIDTH):
    """Hai bàn chân xuất phát (x, y, ψ) đặt hai bên điểm (x, y), vuông góc với hướng mặt."""
    psi = np.deg2rad(yaw_deg)
    return ((x - np.sin(psi) * half_w, y + np.cos(psi) * half_w, psi),
            (x + np.sin(psi) * half_w, y - np.cos(psi) * half_w, psi))


def build_plan(a):
    """A* tìm chuỗi bước chân vòng qua vật cản. Trả (world, steps, goal_xy, title)."""
    sx, sy, syaw = a.start
    gx, gy, gyaw = a.goal
    start_l, start_r = start_feet(sx, sy, syaw)
    w = default_world((sx, sy), (gx, gy))
    fp = FootstepPlanner(w)
    steps = fp.plan(start_l, start_r, (gx, gy, np.deg2rad(gyaw)))
    if steps is None:
        print("Footstep planner thất bại (điểm bắt đầu/đích nằm trong vật cản hoặc không có đường):", fp.stats)
        sys.exit(1)
    print(f"[footstep] từ ({sx:.2f}, {sy:.2f}) tới ({gx:.2f}, {gy:.2f}): {fp.stats['steps']} bước, "
          f"mở {fp.stats['expanded']} node, {1000*fp.stats['time']:.0f} ms")
    title = "P05 — A* footstep planner (xanh = chân trái, đỏ = chân phải) + mẫu đi DCM"
    return w, steps, (gx, gy), title


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", type=float, nargs=3, default=[START_X, START_Y, START_YAW_DEG],
                    metavar=("X", "Y", "YAW_DEG"), help="vị trí bắt đầu x y [m] và hướng [độ]")
    ap.add_argument("--goal", type=float, nargs=3, default=[GOAL_X, GOAL_Y, GOAL_YAW_DEG],
                    metavar=("X", "Y", "YAW_DEG"), help="đích x y [m] và hướng [độ]")
    ap.add_argument("--step-time", type=float, default=STEP_TIME, help="thời gian mỗi bước [s]")
    ap.add_argument("--speed", type=float, default=PLAYBACK_SPEED, help="tốc độ phát ban đầu trong cửa sổ")
    ap.add_argument("--save", action="store_true", help="xuất PNG + 2 GIF vào out/ thay vì mở cửa sổ")
    ap.add_argument("--test", action="store_true", help="tự kiểm tra cửa sổ tương tác mà không mở màn hình")
    a = ap.parse_args()
    OUT.mkdir(exist_ok=True)

    w, steps, goal_xy, title = build_plan(a)
    f_plan, f_top, f_g1 = "walk_plan.png", "walk_topview.gif", "g1_walk_kinematic.gif"

    dp = DCMPlanner(z_c=0.66, T_step=a.step_time)
    tr = dp.plan(steps, tuple(a.start[:2]))           # trọng tâm bắt đầu ở giữa hai bàn chân
    pc = PreviewController(z_c=0.66, dt=dp.dt)
    cx, zx = pc.run(tr["zmp"][:, 0], a.start[0])
    cy, zy = pc.run(tr["zmp"][:, 1], a.start[1])
    v_avg = (tr["com"][-1, 0] - tr["com"][0, 0]) / tr["t"][-1]
    print(f"[DCM] {len(tr['zmps'])} pha ZMP, tổng {tr['t'][-1]:.1f} s; CoM cuối = {np.round(tr['com'][-1], 3)}; "
          f"vận tốc CoM trung bình theo x = {v_avg:.2f} m/s")

    label = "di theo ke hoach A*"
    if a.save:
        save_outputs(w, steps, goal_xy, title, tr, cx, cy, zx, zy, f_plan, f_top)
        from g1_playback import export_gif
        export_gif(steps, tr, dp, str(OUT / f_g1), world=w, label=f"P05 G1 {label}")
        print(f"[out] {f_plan}, {f_top}, {f_g1}")
    else:
        from g1_playback import play_interactive
        play_interactive(steps, tr, dp, world=w, label=label, test_frames=300 if a.test else None,
                         speed=a.speed)


def save_outputs(w, steps, goal_xy, title, tr, cx, cy, zx, zy, f_plan, f_top):
    # ---- static figure
    fig = plt.figure(figsize=(13, 7))
    ax = fig.add_subplot(2, 1, 1)
    draw_world(ax, w); draw_steps(ax, steps)
    ax.plot(tr["zmp"][:, 0], tr["zmp"][:, 1], "k-", lw=.8, drawstyle="steps-post", label="ZMP tham chiếu")
    ax.plot(tr["dcm"][:, 0], tr["dcm"][:, 1], "m-", lw=1.2, label="DCM ξ")
    ax.plot(tr["com"][:, 0], tr["com"][:, 1], "g-", lw=2, label="CoM (DCM planner)")
    ax.plot(cx, cy, "c--", lw=1.2, label="CoM (ZMP preview control)")
    ax.plot(*goal_xy, "g*", ms=14)
    ax.legend(fontsize=8, loc="upper left", ncol=2)
    ax.set_title(title)
    for k, (lab, col) in enumerate([("x", 0), ("y", 1)]):
        ax2 = fig.add_subplot(2, 2, 3 + k)
        ax2.plot(tr["t"], tr["zmp"][:, col], "k", drawstyle="steps-post", lw=.8, label="ZMP ref")
        ax2.plot(tr["t"], tr["dcm"][:, col], "m", label="DCM")
        ax2.plot(tr["t"], tr["com"][:, col], "g", label="CoM (DCM)")
        ax2.plot(tr["t"], [cx, cy][col], "c--", label="CoM (preview)")
        ax2.plot(tr["t"], [zx, zy][col], "c:", lw=.8, label="ZMP (preview)")
        ax2.set_xlabel("t [s]"); ax2.set_ylabel(f"{lab} [m]"); ax2.grid(alpha=.3); ax2.legend(fontsize=7)
    fig.tight_layout(); fig.savefig(OUT / f_plan, dpi=110); plt.close(fig)

    # ---- animation (top view)
    fig, ax = plt.subplots(figsize=(10, 6))
    draw_world(ax, w); draw_steps(ax, steps, alpha=.25)
    com_l, = ax.plot([], [], "g-", lw=2); dcm_p, = ax.plot([], [], "mo", ms=6); zmp_p, = ax.plot([], [], "ks", ms=6)
    com_p, = ax.plot([], [], "go", ms=9); ttl = ax.set_title("")
    idx = np.arange(0, len(tr["t"]), 10)

    def upd(k):
        i = idx[k]
        com_l.set_data(tr["com"][:i, 0], tr["com"][:i, 1])
        com_p.set_data([tr["com"][i, 0]], [tr["com"][i, 1]])
        dcm_p.set_data([tr["dcm"][i, 0]], [tr["dcm"][i, 1]])
        zmp_p.set_data([tr["zmp"][i, 0]], [tr["zmp"][i, 1]])
        ttl.set_text(f"t = {tr['t'][i]:.2f} s   ■ ZMP   ● DCM (tím)   ● CoM (xanh)")
        return com_l, com_p, dcm_p, zmp_p, ttl
    anim = FuncAnimation(fig, upd, frames=len(idx), blit=False)
    anim.save(OUT / f_top, writer=PillowWriter(fps=20), dpi=70); plt.close(fig)


if __name__ == "__main__":
    main()