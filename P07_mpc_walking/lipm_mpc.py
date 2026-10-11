from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "common"))
from hmc.qp import DenseQP                       # noqa: E402
from hmc.robot import convex_hull                # noqa: E402

G = 9.81
ROWS_PER_SUPPORT = 8       # số hàng tối đa cho một vùng đỡ (hộp bàn chân: 4; bao lồi hai chân: ≤ 8)


def halfspaces(pts: np.ndarray):
    """Bao lồi của các điểm 2D → (A, b) với A p ≤ b (mỗi cạnh một hàng)."""
    hull = convex_hull(pts)
    A, b = [], []
    for i in range(len(hull)):
        p0, p1 = hull[i], hull[(i + 1) % len(hull)]
        e = p1 - p0
        nrm = np.array([e[1], -e[0]])            # bao lồi ngược chiều kim đồng hồ → pháp tuyến hướng ra ngoài
        nrm /= np.linalg.norm(nrm) + 1e-12
        A.append(nrm)
        b.append(nrm @ p0)
    return np.array(A), np.array(b)


class LipmMPC:
    def __init__(self, z_c: float, dt: float = 0.05, N: int = 32, M: int = 3,
                 w_zmp: float = 3.0, w_dzmp: float = 0.5, w_step: float = 3.0, w_move: float = 1.0,
                 w_stab_soft: float = 1e4, tail: float = 1.5, freeze: float = 0.12, adapt: bool = True,
                 step_dx=(-0.12, 0.32), step_dy=(0.13, 0.45), max_adapt: float = 0.35, gamma: float = 0.8):
        self.z_c, self.dt, self.N, self.M, self.tail = z_c, dt, N, M, tail
        self.w = w = np.sqrt(G / z_c)
        self.a = np.exp(-w * dt * np.arange(N)) * (1 - np.exp(-w * dt))     # hệ số a_k của (*)
        self.eT = np.exp(-w * dt * N)
        self.wts = dict(zmp=w_zmp, dzmp=w_dzmp, step=w_step, move=w_move, stab=w_stab_soft)
        self.freeze, self.adapt = freeze, adapt
        self.step_dx, self.step_dy, self.max_adapt, self.gamma = step_dx, step_dy, max_adapt, gamma
        self.n = 2 * N + 2 * M                          # [px(N), py(N), fx(M), fy(M)]
        self.nrow = ROWS_PER_SUPPORT * (N + 1) + 4 * M + 2   # vùng đỡ ZMP (N khoảng + thời điểm hiện tại) + động học/giới hạn dời + (*)
        self.qp = DenseQP(self.n, self.nrow, eps=1e-6, max_iter=4000)
        D = np.eye(N) - np.eye(N, k=-1)                 # sai phân p_k − p_{k−1}
        self.DtD = D.T @ D
        self.qp_soft = DenseQP(self.n, self.nrow - 2, eps=1e-6, max_iter=4000)
        self._hull_cache = {}
        self.validator = None                           # hàm (x, y, ψ) → bool: vị trí bước có hợp lệ (vd. không đè vật cản)
        self.last = None
        ch, sh = np.cosh(w * dt), np.sinh(w * dt)       # LIPM rời rạc chính xác (chỉ để dự báo / ghi log)
        self.A = np.array([[ch, sh / w], [w * sh, ch]])
        self.B = np.array([1 - ch, -w * sh])

    # ---------------------------------------------------------------- chỉ số biến
    def ip(self, ax, k):
        return ax * self.N + k

    def iF(self, ax, m):
        return 2 * self.N + ax * self.M + m

    # ---------------------------------------------------------------- helpers
    def _split(self, coef, var_of, steps):
        """Tách tổ hợp neo thành phần hằng (2,) và hệ số các biến bước chân {m: α}."""
        c0, alpha = np.zeros(2), {}
        for i, a in coef.items():
            if i in var_of:
                alpha[var_of[i]] = alpha.get(var_of[i], 0.0) + a
            else:
                c0 += a * np.array(steps[i][:2])
        return c0, alpha

    def _xi_tail(self, schedule, t_end):
        
        w, h = self.w, self.dt
        n = int(round(self.tail / h))
        wk = np.exp(-w * h * np.arange(n)) * (1 - np.exp(-w * h))
        z = np.array([schedule.zmp_ref(t_end + (k + 0.5) * h) for k in range(n)])
        return wk @ z + np.exp(-w * h * n) * schedule.zmp_ref(t_end + self.tail)

    def predict(self, c, cd, P):
        
        N = self.N
        out = np.zeros((N + 1, 2))
        for ax in (0, 1):
            x = np.array([c[ax], cd[ax]])
            out[0, ax] = x[0]
            for k in range(N):
                x = self.A @ x + self.B * P[ax * N + k]
                out[k + 1, ax] = x[0]
        return out

    # ---------------------------------------------------------------- main
    def solve(self, c, cd, t_now: float, gs, p_prev):
        
        N, M, dt, W = self.N, self.M, self.dt, self.wts
        steps = gs.steps
        for j, tl in gs.land_time.items():               # bước đã quá thời điểm "đóng băng" → chốt, dời mục tiêu các bước sau
            if j >= 2 and tl <= t_now + self.freeze and j not in gs.committed:
                gs.commit(j, self.gamma if self.adapt else 0.0, self.validator)
        J = gs.adaptable(t_now, N * dt, self.freeze)[:M] if self.adapt else []
        var_of = {j: m for m, j in enumerate(J)}

        # ---- hàm mục tiêu ½xᵀHx + qᵀx (dựng theo khối, vector hóa)
        # (1) ZMP bám điểm neo:  p_k − Σ_m α_km f_m − c0_k
        alpha = np.zeros((N, M))
        c0 = np.zeros((N, 2))
        for k in range(N):
            coef, _ = gs.anchor(t_now + (k + 0.5) * dt)
            c0[k], al = self._split(coef, var_of, steps)
            for m, a in al.items():
                alpha[k, m] = a
        H = np.zeros((self.n, self.n))
        q = np.zeros(self.n)
        Ipp = W["zmp"] * np.eye(N) + W["dzmp"] * self.DtD          # (2) thêm ZMP mượt
        Hpf = -W["zmp"] * alpha
        Hff = W["zmp"] * alpha.T @ alpha
        for ax in (0, 1):
            sp = slice(ax * N, (ax + 1) * N)
            sf = slice(2 * N + ax * M, 2 * N + (ax + 1) * M)
            H[sp, sp] = Ipp
            H[sp, sf] = Hpf
            H[sf, sp] = Hpf.T
            H[sf, sf] = Hff
            q[sp] = -W["zmp"] * c0[:, ax]
            q[ax * N] += -W["dzmp"] * p_prev[ax]
            q[sf] = W["zmp"] * alpha.T @ c0[:, ax]
        # (3) bước chân: bám mục tiêu + không nhảy đích; biến thừa ghim về 0
        for m in range(M):
            for ax in (0, 1):
                i = self.iF(ax, m)
                if m < len(J):
                    j = J[m]
                    H[i, i] += W["step"] + W["move"]
                    q[i] += -W["step"] * gs.goal[j][ax] - W["move"] * steps[j][ax]
                else:
                    H[i, i] += 1.0

        # ---- ràng buộc
        A = np.zeros((self.nrow, self.n))
        lo = np.full(self.nrow, -1e9)
        hi = np.full(self.nrow, 1e9)
        row = 0
        # vùng đỡ ở GIỮA mỗi khoảng k; thêm vùng đỡ TẠI thời điểm hiện tại cho khoảng đầu (k = 0)
        for k, t in [(k, t_now + (k + 0.5) * dt) for k in range(N)] + [(0, t_now + 1e-4)]:
            sup = gs.support(t)
            if sup[0] == "foot":                         
                i = sup[1]
                normals, half = gs.foot_box(i)
                rs = slice(row, row + 4)
                A[rs, self.ip(0, k)], A[rs, self.ip(1, k)] = normals[:, 0], normals[:, 1]
                if i in var_of:
                    A[rs, self.iF(0, var_of[i])], A[rs, self.iF(1, var_of[i])] = -normals[:, 0], -normals[:, 1]
                    hi[rs] = half
                else:
                    hi[rs] = half + normals @ np.array(steps[i][:2])
            else:                                       # bao lồi hai bàn chân (vị trí hiện hành)
                key = (sup[1], sup[2], steps[sup[1]], steps[sup[2]])
                if key not in self._hull_cache:
                    if len(self._hull_cache) > 400:
                        self._hull_cache.clear()
                    self._hull_cache[key] = halfspaces(gs.hull_points(sup[1], sup[2]))
                Ah, bh = self._hull_cache[key]
                nr = min(len(bh), ROWS_PER_SUPPORT)
                A[row:row + nr, self.ip(0, k)] = Ah[:nr, 0]
                A[row:row + nr, self.ip(1, k)] = Ah[:nr, 1]
                hi[row:row + nr] = bh[:nr]
            row += ROWS_PER_SUPPORT
        for m in range(M):                              # động học giữa bước j và bước trước đó j−1
            if m < len(J):
                j = J[m]
                prev = j - 1
                psi = steps[prev][2]
                ex, ey = np.array([np.cos(psi), np.sin(psi)]), np.array([-np.sin(psi), np.cos(psi)])
                side = steps[j][3]
                for e, (lo_v, hi_v), sgn in ((ex, self.step_dx, 1.0), (ey, self.step_dy, side)):
                    A[row, self.iF(0, m)] += sgn * e[0]
                    A[row, self.iF(1, m)] += sgn * e[1]
                    const = 0.0
                    if prev in var_of:
                        A[row, self.iF(0, var_of[prev])] -= sgn * e[0]
                        A[row, self.iF(1, var_of[prev])] -= sgn * e[1]
                    else:
                        const = sgn * (e @ np.array(steps[prev][:2]))
                    lo[row], hi[row] = lo_v + const, hi_v + const
                    row += 1
                for ax in (0, 1):                       # không dời quá max_adapt so với mục tiêu
                    A[row, self.iF(ax, m)] = 1.0
                    lo[row] = gs.goal[j][ax] - self.max_adapt
                    hi[row] = gs.goal[j][ax] + self.max_adapt
                    row += 1
            else:
                row += 4
        # ràng buộc ổn định
        xiT = self._xi_tail(gs, t_now + N * dt)
        coefT, _ = gs.anchor(t_now + N * dt)
        ctrT_now = gs.point(coefT)
        c0T, alphaT = self._split(coefT, var_of, steps)
        off = xiT - ctrT_now                            # ξ(T) = ctr(T)(F) + off
        xi0 = c + cd / self.w
        rhs = xi0 - self.eT * (c0T + off)
        for ax in (0, 1):
            A[row, [self.ip(ax, k) for k in range(N)]] = self.a
            for m, a in alphaT.items():
                A[row, self.iF(ax, m)] += self.eT * a
            lo[row] = hi[row] = rhs[ax]
            row += 1

        # giải
        x = self.qp.solve(H, q, A, lo, hi)
        mode = "hard"
        if x is None:                                   # bị đẩy quá mạnh
            Hs, qs = H.copy(), q.copy()
            for r_ in (-2, -1):
                a_row = A[r_]
                Hs += W["stab"] * np.outer(a_row, a_row)
                qs += -W["stab"] * a_row * lo[r_]
            x = self.qp_soft.solve(Hs, qs, A[:-2], lo[:-2], hi[:-2])
            mode = "soft"
        ok = x is not None
        if not ok:
            x = np.zeros(self.n)
            x[:N], x[N:2 * N] = p_prev[0], p_prev[1]
            mode = "fail"
        p0 = np.array([x[self.ip(0, 0)], x[self.ip(1, 0)]])
        adapted = {}
        if ok:
            for m, j in enumerate(J):
                fx, fy = x[self.iF(0, m)], x[self.iF(1, m)]
                if self.validator is None or self.validator(fx, fy, steps[j][2]):
                    steps[j] = (fx, fy, steps[j][2], steps[j][3])
                adapted[j] = (steps[j][0] - gs.steps_ref[j][0], steps[j][1] - gs.steps_ref[j][1])
        P = np.r_[x[:N], x[N:2 * N]]
        self.last = dict(ok=ok, mode=mode, P=P, t=t_now + dt * np.arange(N), xi0=xi0, adapted=adapted)
        return p0, self.last

    def com_reference(self, c, cd, p0, s: float):
        
        w = self.w
        ch, sh = np.cosh(w * s), np.sinh(w * s)
        cr = p0 + (c - p0) * ch + cd * sh / w
        vr = (c - p0) * w * sh + cd * ch
        ar = w**2 * (cr - p0)
        return cr, vr, ar
