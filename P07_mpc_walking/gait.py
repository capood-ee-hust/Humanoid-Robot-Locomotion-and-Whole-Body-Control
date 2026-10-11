from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

# Hình học bàn chân G1 (đo từ mô hình): tâm đế nằm trước cổ chân 3.5 cm, cổ chân cao 3.5 cm so với sàn
ANKLE_TO_CENTER = 0.035
ANKLE_HEIGHT = 0.035
SOLE_HALF_L = 0.085          # nửa chiều dài vùng tiếp xúc (tâm 4 quả cầu: −0.05 … +0.12 so với cổ chân)
SOLE_HALF_W = 0.027          # nửa chiều rộng (gót ±0.025, mũi ±0.03)


def yaw_R(psi: float) -> np.ndarray:
    c, s = np.cos(psi), np.sin(psi)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def quintic(tau: float):
    """Luật minimum-jerk s(τ), ṡ(τ), s̈(τ) với τ ∈ [0, 1] (đạo hàm theo τ)."""
    tau = min(max(tau, 0.0), 1.0)
    s = 10 * tau**3 - 15 * tau**4 + 6 * tau**5
    ds = 30 * tau**2 - 60 * tau**3 + 30 * tau**4
    dds = 60 * tau - 180 * tau**2 + 120 * tau**3
    return s, ds, dds


def sole_corners(step, shrink_x=0.0, shrink_y=0.0) -> np.ndarray:
    """4 góc (x, y) của vùng tiếp xúc bàn chân, co vào shrink_x / shrink_y [m]."""
    x, y, psi = step[0], step[1], step[2]
    hl, hw = SOLE_HALF_L - shrink_x, SOLE_HALF_W - shrink_y
    c, s = np.cos(psi), np.sin(psi)
    loc = np.array([[hl, hw], [-hl, hw], [-hl, -hw], [hl, -hw]])
    return np.c_[x + c * loc[:, 0] - s * loc[:, 1], y + s * loc[:, 0] + c * loc[:, 1]]


def ankle_pos(step, z=ANKLE_HEIGHT) -> np.ndarray:
    """Vị trí body cổ chân (ankle_roll_link) khi bàn chân đặt tại step."""
    x, y, psi = step[0], step[1], step[2]
    return np.array([x - np.cos(psi) * ANKLE_TO_CENTER, y - np.sin(psi) * ANKLE_TO_CENTER, z])


@dataclass
class Phase:
    kind: str                    # "DS" hoặc "SS"
    t0: float
    T: float
    feet: dict                   # side → chỉ số bước đang chạm đất (DS: 2 chân; SS: chỉ chân trụ)
    anchor_from: dict = None     # điểm neo đầu pha: {chỉ số bước: hệ số}
    anchor_to: dict = None       # điểm neo cuối pha
    load_side: int = 0           # DS: chân vừa chạm đất (nhận lực dần), 0 = không có
    unload_side: int = 0         # DS: chân sắp nhấc (nhả lực dần), 0 = không có
    swing_side: int = 0          # SS: chân đung đưa
    swing_from: int = -1         # SS: chỉ số bước chân đung đưa rời đi
    swing_to: int = -1           # SS: chỉ số bước chân đung đưa đáp xuống
    final: bool = False          # DS cuối: giữ mãi sau khi hết thời gian


@dataclass
class GaitParams:
    t_ss: float = 0.45           # thời gian đứng một chân [s] — ngắn để DCM không kịp khuếch đại nhiều
    t_ds: float = 0.2            # thời gian đứng hai chân giữa hai bước [s]
    t_init: float = 1.2          # DS đầu: dồn trọng tâm sang chân trụ đầu tiên
    t_final: float = 1.5         # DS cuối: đưa trọng tâm về giữa hai chân
    ramp: float = 0.05           # thời gian dốc lực khi nhận / nhả tải [s]
    lift: float = 0.06           # độ cao nhấc chân [m]
    touchdown_dz: float = -0.004 # đặt chân thấp hơn mặt sàn một chút để chắc chắn chạm
    shrink_x: float = 0.015      # co hộp hỗ trợ vào trong theo chiều dọc bàn chân (biên an toàn) [m]
    shrink_y: float = 0.006      # … theo chiều ngang bàn chân [m]


@dataclass
class GaitSchedule:
    steps: list
    p: GaitParams = field(default_factory=GaitParams)

    def __post_init__(self):
        s, p = [tuple(map(float, st[:3])) + (int(st[3]),) for st in self.steps], self.p
        self.steps = list(s)             # vị trí hiện hành (MPC có thể sửa)
        self.steps_ref = list(s)         # kế hoạch gốc (chỉ để so sánh / vẽ)
        self.goal = list(s)              # mục tiêu MPC bám theo = kế hoạch gốc + phần dời "kế thừa" (xem commit)
        self.committed = {0, 1}          # các bước đã chốt vị trí
        n = len(s)
        assert n >= 3, "cần ít nhất 2 chân ban đầu + 1 bước"
        one = lambda i: {i: 1.0}                        # noqa: E731
        mid = lambda i, j: {i: 0.5, j: 0.5}            # noqa: E731
        ph, t = [], 0.0
        ph.append(Phase("DS", t, p.t_init, {s[0][3]: 0, s[1][3]: 1}, mid(0, 1), one(1), unload_side=s[0][3]))
        t += p.t_init
        self.land_time = {0: 0.0, 1: 0.0}             # thời điểm mỗi bước chạm đất
        for i in range(1, n - 1):
            ph.append(Phase("SS", t, p.t_ss, {s[i][3]: i}, one(i), one(i),
                            swing_side=s[i + 1][3], swing_from=i - 1, swing_to=i + 1))
            t += p.t_ss
            self.land_time[i + 1] = t
            if i + 1 == n - 1:   # bước cuối (khép chân) → DS cuối, ZMP về giữa hai chân
                ph.append(Phase("DS", t, p.t_final, {s[i][3]: i, s[i + 1][3]: i + 1}, one(i), mid(i, i + 1),
                                load_side=s[i + 1][3], final=True))
                t += p.t_final
            else:
                ph.append(Phase("DS", t, p.t_ds, {s[i][3]: i, s[i + 1][3]: i + 1}, one(i), one(i + 1),
                                load_side=s[i + 1][3], unload_side=s[i][3]))
                t += p.t_ds
        self.phases = ph
        self.T_end = t
        self._t0 = np.array([q.t0 for q in ph])

    # ------------------------------------------------------------------ helpers
    def phase_at(self, t: float) -> Phase:
        k = int(np.searchsorted(self._t0, t, side="right")) - 1
        return self.phases[min(max(k, 0), len(self.phases) - 1)]

    def progress(self, q: Phase, t: float) -> float:
        """s(t) ∈ [0, 1]: điểm neo dời từ anchor_from sang anchor_to (chỉ trong phần giữa của DS)."""
        if q.kind == "SS":
            return 0.0
        r = self.p.ramp
        span = (q.T - r) if q.final else (q.T - 2 * r)
        return quintic((t - q.t0 - r) / max(span, 1e-6))[0]

    def anchor(self, t: float):
        """Tâm hộp hỗ trợ tại t dưới dạng tổ hợp tuyến tính các bước: ({chỉ số: hệ số}, yaw)."""
        q = self.phase_at(t)
        s = self.progress(q, t)
        coef = {}
        for i, a in q.anchor_from.items():
            coef[i] = coef.get(i, 0.0) + (1 - s) * a
        for i, a in q.anchor_to.items():
            coef[i] = coef.get(i, 0.0) + s * a
        yaw = sum(a * self.steps[i][2] for i, a in coef.items())
        return coef, yaw

    def point(self, coef: dict, steps=None) -> np.ndarray:
        steps = self.steps if steps is None else steps
        return sum(a * np.array(steps[i][:2]) for i, a in coef.items())

    def zmp_ref(self, t: float, steps=None) -> np.ndarray:
        coef, _ = self.anchor(t)
        return self.point(coef, steps)

    def support(self, t: float):
        """Vùng đỡ (ZMP phải nằm trong) tại t:
            ("foot", i)    : một bàn chân — pha một chân, và hai cửa sổ dốc lực trong pha hai chân
            ("hull", i, j) : bao lồi hai bàn chân — phần giữa của pha hai chân"""
        q = self.phase_at(t)
        tl = t - q.t0
        if q.kind == "SS":
            return ("foot", next(iter(q.feet.values())))
        side_idx = dict(q.feet)
        if q.load_side and tl < self.p.ramp:                       # chân vừa chạm đất chưa chịu lực
            return ("foot", next(i for s, i in side_idx.items() if s != q.load_side))
        if q.unload_side and not q.final and tl > q.T - self.p.ramp:   # chân sắp nhấc đã nhả lực
            return ("foot", next(i for s, i in side_idx.items() if s != q.unload_side))
        i, j = sorted(side_idx.values())
        return ("hull", i, j)

    def foot_box(self, i: int):
        """Hộp bàn chân i (đã co vào): (pháp tuyến 4×2, nửa kích thước 4).
        Ràng buộc:  n_k · (p − tâm_bước_i) ≤ h_k  — tuyến tính theo vị trí bước i."""
        psi = self.steps[i][2]
        c, s = np.cos(psi), np.sin(psi)
        ex, ey = np.array([c, s]), np.array([-s, c])
        hl, hw = SOLE_HALF_L - self.p.shrink_x, SOLE_HALF_W - self.p.shrink_y
        return np.array([ex, -ex, ey, -ey]), np.array([hl, hl, hw, hw])

    def hull_points(self, i: int, j: int) -> np.ndarray:
        """8 góc (đã co vào) của hai bàn chân i, j — bao lồi của chúng là vùng đỡ khi đứng hai chân."""
        return np.vstack([sole_corners(self.steps[k], self.p.shrink_x, self.p.shrink_y) for k in (i, j)])

    # ------------------------------------------------------------------ main query
    def query(self, t: float) -> dict:
        p = self.p
        q = self.phase_at(t)
        tl = t - q.t0
        stance = {+1: False, -1: False}
        fz = {+1: 1.0, -1: 1.0}
        for side in q.feet:
            stance[side] = True
        swing = None
        if q.kind == "DS":
            if q.load_side:
                fz[q.load_side] = min(1.0, tl / p.ramp)
            if q.unload_side and not q.final:
                fz[q.unload_side] = min(1.0, max(0.0, (q.T - tl) / p.ramp))
            pelvis_yaw = float(np.mean([self.steps[i][2] for i in q.feet.values()]))
        else:
            tau = tl / q.T
            s, ds, dds = quintic(tau)
            st0, st1 = self.steps[q.swing_from], self.steps[q.swing_to]
            a0 = ankle_pos(st0)
            a1 = ankle_pos(st1, ANKLE_HEIGHT + p.touchdown_dz)
            pos = a0 + s * (a1 - a0)
            vel = (ds / q.T) * (a1 - a0)
            acc = (dds / q.T**2) * (a1 - a0)
            # nhấc chân: hình chuông 16 τ²(1−τ)², vận tốc đứng bằng 0 ở hai đầu
            tc = min(max(tau, 0.0), 1.0)
            b = 16 * tc**2 * (1 - tc) ** 2
            db = 16 * (2 * tc * (1 - tc) ** 2 - 2 * tc**2 * (1 - tc))
            ddb = 16 * (2 * (1 - tc) ** 2 - 8 * tc * (1 - tc) + 2 * tc**2)
            pos[2] += p.lift * b
            vel[2] += p.lift * db / q.T
            acc[2] += p.lift * ddb / q.T**2
            psi = st0[2] + s * (st1[2] - st0[2])
            swing = dict(side=q.swing_side, pos=pos, vel=vel, acc=acc, R=yaw_R(psi), tau=tau)
            stance_i = next(iter(q.feet.values()))
            pelvis_yaw = 0.5 * (self.steps[stance_i][2] + psi)
        return dict(kind=q.kind, stance=stance, fz_scale=fz, zmp_ref=self.zmp_ref(t), swing=swing,
                    pelvis_yaw=pelvis_yaw, phase=q, done=(t >= self.T_end))

    def commit(self, j: int, gamma: float = 0.8, validator=None):
        """Bước j đã chốt vị trí (sắp chạm đất). Nếu nó lệch khỏi mục tiêu một đoạn r thì dời mục tiêu các bước
        sau thêm r·γ^k (k = 1, 2, …): robot bị đẩy sẽ đi tiếp từ chỗ mới rồi TỪ TỪ quay về đường kế hoạch,
        thay vì giật mạnh về đường cũ (dễ vượt lố sang phía bên kia)."""
        if j in self.committed:
            return
        self.committed.add(j)
        r = np.subtract(self.steps[j][:2], self.goal[j][:2])
        if gamma <= 0.0 or np.hypot(*r) < 1e-4:
            return
        for k in range(j + 1, len(self.steps)):
            g = gamma ** (k - j)
            x, y = self.goal[k][0] + g * r[0], self.goal[k][1] + g * r[1]
            if validator is not None and not validator(x, y, self.goal[k][2]):
                continue
            self.goal[k] = (x, y) + self.goal[k][2:]
            if k not in self.committed:
                sx, sy = self.steps[k][0] + g * r[0], self.steps[k][1] + g * r[1]
                if validator is None or validator(sx, sy, self.steps[k][2]):
                    self.steps[k] = (sx, sy) + self.steps[k][2:]

    def adaptable(self, t_now: float, horizon: float, freeze: float):
        """Chỉ số các bước MPC được phép dời: chưa chạm đất, chạm đất trong horizon, còn > freeze giây."""
        return [j for j, tl in self.land_time.items()
                if j >= 2 and t_now + freeze < tl <= t_now + horizon]
