from __future__ import annotations

import heapq
import itertools
import time
from dataclasses import dataclass, field

import numpy as np

FOOT_L, FOOT_W = 0.20, 0.08          # G1-like foot footprint [m]
STANCE_W = 0.20                      # nominal lateral distance between foot centres


@dataclass
class Obstacle:
    kind: str                        # "circle" | "rect"
    p: tuple                         # circle: (cx, cy, r); rect: (xmin, ymin, xmax, ymax)

    def dist(self, pts: np.ndarray) -> np.ndarray:
        if self.kind == "circle":
            cx, cy, r = self.p
            return np.hypot(pts[:, 0] - cx, pts[:, 1] - cy) - r
        x0, y0, x1, y1 = self.p
        dx = np.maximum.reduce([x0 - pts[:, 0], np.zeros(len(pts)), pts[:, 0] - x1])
        dy = np.maximum.reduce([y0 - pts[:, 1], np.zeros(len(pts)), pts[:, 1] - y1])
        return np.hypot(dx, dy)


@dataclass
class World:
    xlim: tuple = (-0.5, 4.5)
    ylim: tuple = (-1.5, 1.5)
    obstacles: list = field(default_factory=list)
    margin: float = 0.04

    def foot_free(self, x, y, psi) -> bool:
        c, s = np.cos(psi), np.sin(psi)
        u = np.linspace(-FOOT_L / 2, FOOT_L / 2, 5)
        v = np.linspace(-FOOT_W / 2, FOOT_W / 2, 3)
        U, V = np.meshgrid(u, v)
        pts = np.c_[x + c * U.ravel() - s * V.ravel(), y + s * U.ravel() + c * V.ravel()]
        if np.any(pts[:, 0] < self.xlim[0]) or np.any(pts[:, 0] > self.xlim[1]) or \
           np.any(pts[:, 1] < self.ylim[0]) or np.any(pts[:, 1] > self.ylim[1]):
            return False
        return all(np.all(o.dist(pts) > self.margin) for o in self.obstacles)


def foot_corners(x, y, psi):
    c, s = np.cos(psi), np.sin(psi)
    loc = np.array([[FOOT_L / 2, FOOT_W / 2], [-FOOT_L / 2, FOOT_W / 2], [-FOOT_L / 2, -FOOT_W / 2], [FOOT_L / 2, -FOOT_W / 2]])
    R = np.array([[c, -s], [s, c]])
    return loc @ R.T + [x, y]


class FootstepPlanner:
    def __init__(self, world: World, eps: float = 1.5, w_turn: float = 2.0, w_side: float = 2.0,
                 max_expand: int = 60000):
        self.w, self.eps, self.w_turn, self.w_side = world, eps, w_turn, w_side
        self.max_expand = max_expand
        dxs = [-0.10, 0.0, 0.10, 0.20, 0.25]
        dys = [0.0, 0.05]
        dps = np.deg2rad([-15, 0, 15])
        self.actions = list(itertools.product(dxs, dys, dps))
        self.max_step = 0.25
        self.stats = {}

    @staticmethod
    def _key(x, y, psi, side):
        return (round(x / 0.05), round(y / 0.05), round(((psi + np.pi) % (2 * np.pi)) / np.deg2rad(10)), side)

    def _successors(self, s):
        x, y, psi, side = s                     # side = which foot moves next: +1 left, −1 right
        c, sn = np.cos(psi), np.sin(psi)
        for dx, dy, dpsi in self.actions:
            lat = side * (STANCE_W + dy)
            if side * dpsi < -1e-9 and dx < 0:  # no backward inward-twist (reachability)
                continue
            nx, ny = x + c * dx - sn * lat, y + sn * dx + c * lat
            npsi = psi + dpsi
            if self.w.foot_free(nx, ny, npsi):
                cost = 1.0 + self.w_turn * abs(dpsi) + self.w_side * dy
                yield (nx, ny, npsi, -side), cost

    def _h(self, s, goal):
        x, y, psi, side = s
        # midpoint of the stance pair ≈ stance foot shifted half a stance width inwards
        mx = x - np.sin(psi) * side * STANCE_W / 2
        my = y + np.cos(psi) * side * STANCE_W / 2
        dpsi = abs((goal[2] - psi + np.pi) % (2 * np.pi) - np.pi)
        return np.hypot(goal[0] - mx, goal[1] - my) / self.max_step + dpsi / np.deg2rad(20)

    def plan(self, start_left, start_right, goal, tol=0.15):
        """start_*: (x, y, ψ) foot poses. goal: (x, y, ψ) of the stance-pair midpoint."""
        t0 = time.time()
        s0 = (*start_right, +1)                  # right foot is stance, left moves first
        openl = [(self.eps * self._h(s0, goal), 0.0, 0, s0)]
        parent, g = {self._key(*s0): None}, {self._key(*s0): 0.0}
        nodes = {self._key(*s0): s0}
        cnt, n_exp = itertools.count(1), 0
        while openl and n_exp < self.max_expand:
            f, gc, _, s = heapq.heappop(openl)
            k = self._key(*s)
            if gc > g.get(k, np.inf) + 1e-12:
                continue
            n_exp += 1
            if self._h(s, goal) * self.max_step < tol:          # position + heading close enough
                path = [s]
                while parent[self._key(*path[-1])] is not None:
                    path.append(parent[self._key(*path[-1])])
                path = path[::-1]
                self.stats = dict(ok=True, expanded=n_exp, time=time.time() - t0, steps=len(path) - 1)
                return self._to_footsteps(start_left, start_right, path)
            for s2, c in self._successors(s):
                k2 = self._key(*s2)
                g2 = gc + c
                if g2 < g.get(k2, np.inf):
                    g[k2], parent[k2], nodes[k2] = g2, s, s2
                    heapq.heappush(openl, (g2 + self.eps * self._h(s2, goal), g2, next(cnt), s2))
        self.stats = dict(ok=False, expanded=n_exp, time=time.time() - t0)
        return None

    @staticmethod
    def _to_footsteps(start_left, start_right, path):
        """Return list of footsteps [(x, y, ψ, side)], side +1 = left, −1 = right, incl. the 2 initial feet."""
        # ordering convention used by the DCM planner / playback:
        # steps[0] = initial pose of the foot that moves FIRST (left), steps[1] = initial stance foot
        # → during interval i ≥ 1: stance = steps[i], swing foot goes steps[i−1] → steps[i+1]
        steps = [(*start_left, +1), (*start_right, -1)]
        for s in path[1:]:
            x, y, psi, next_side = s
            steps.append((x, y, psi, -next_side))   # this foot was just placed
        # final step: bring the trailing foot beside the last one (square up)
        x, y, psi, side = steps[-1]
        lat = -side * STANCE_W
        steps.append((x - np.sin(psi) * lat, y + np.cos(psi) * lat, psi, -side))
        return steps


def straight_footsteps(start_left, start_right, distance: float, step_len: float = 0.20,
                       world: World | None = None):

    psi = 0.5 * (start_left[2] + start_right[2])
    fwd = np.array([np.cos(psi), np.sin(psi)])
    mid0 = 0.5 * (np.array(start_left[:2]) + np.array(start_right[:2]))
    half_w = 0.5 * np.linalg.norm(np.array(start_left[:2]) - np.array(start_right[:2]))
    left_dir = np.array([-np.sin(psi), np.cos(psi)])
    n = max(1, int(np.ceil(distance / step_len)))
    steps = [(*start_left, +1), (*start_right, -1)]
    side = +1                                      # chân trái bước trước
    for k in range(1, n + 1):
        p = mid0 + fwd * (k * step_len) + left_dir * (side * half_w)
        steps.append((p[0], p[1], psi, side))
        side = -side
    p = mid0 + fwd * (n * step_len) + left_dir * (side * half_w)     # đứng song song
    steps.append((p[0], p[1], psi, side))
    if world is not None and not all(world.foot_free(x, y, ps) for x, y, ps, _ in steps):
        return None
    return steps
