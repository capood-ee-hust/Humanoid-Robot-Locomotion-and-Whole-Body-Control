"""Unitree G1 model helpers.

The G1 model comes from MuJoCo Menagerie (run `python setup_models.py` at repo root).
Menagerie's G1 uses *position* actuators (kp=500). For whole-body control we need
*torque* actuators, so `load_g1()` converts every actuator into a pure motor:
    force = ctrl,   ctrl in [-tau_max, tau_max]
"""
from __future__ import annotations

import os
from pathlib import Path

import mujoco
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]


def g1_dir() -> Path:
    env = os.environ.get("HMP_G1_DIR")
    p = Path(env) if env else REPO_ROOT / "models" / "unitree_g1"
    if not (p / "scene.xml").exists():
        raise FileNotFoundError(
            f"Không tìm thấy G1 tại {p}. Chạy `python setup_models.py` ở thư mục gốc repo "
            "hoặc đặt biến môi trường HMP_G1_DIR trỏ tới thư mục unitree_g1 của Menagerie.")
    return p


def to_torque_actuators(m: mujoco.MjModel) -> None:
    """Convert every actuator of a compiled model into a pure torque motor."""
    for i in range(m.nu):
        j = m.actuator_trnid[i, 0]
        lo, hi = m.jnt_actfrcrange[j]
        m.actuator_gaintype[i] = mujoco.mjtGain.mjGAIN_FIXED
        m.actuator_gainprm[i, :] = 0.0
        m.actuator_gainprm[i, 0] = 1.0
        m.actuator_biastype[i] = mujoco.mjtBias.mjBIAS_NONE
        m.actuator_biasprm[i, :] = 0.0
        m.actuator_ctrllimited[i] = 1
        m.actuator_ctrlrange[i] = [lo, hi]


def load_g1(edit_spec=None, timestep: float = 0.002) -> mujoco.MjModel:
    """Load G1 scene. `edit_spec(spec)` may add objects (tables, boxes, obstacles)."""
    spec = mujoco.MjSpec.from_file(str(g1_dir() / "scene.xml"))
    if edit_spec is not None:
        edit_spec(spec)
    m = spec.compile()
    m.opt.timestep = timestep
    to_torque_actuators(m)
    return m


class G1:
    """Convenience wrapper: indices, contact points, CoM, Jacobians, nominal pose."""

    LEG_L = ["left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint",
             "left_knee_joint", "left_ankle_pitch_joint", "left_ankle_roll_joint"]
    LEG_R = [n.replace("left", "right") for n in LEG_L]
    FEET = ("left_ankle_roll_link", "right_ankle_roll_link")

    def __init__(self, m: mujoco.MjModel, d: mujoco.MjData | None = None):
        self.m = m
        self.d = d if d is not None else mujoco.MjData(m)
        self.nv, self.nu = m.nv, m.nu
        # actuator -> joint -> dof / qpos address (actuator order == command order)
        jids = m.actuator_trnid[:, 0]
        self.act_dof = np.array([m.jnt_dofadr[j] for j in jids])
        self.act_qpos = np.array([m.jnt_qposadr[j] for j in jids])
        self.act_names = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_ACTUATOR, i) for i in range(m.nu)]
        self.tau_max = m.actuator_ctrlrange[:, 1].copy()
        self.q_lo = np.array([m.jnt_range[j, 0] for j in jids])
        self.q_hi = np.array([m.jnt_range[j, 1] for j in jids])
        self.pelvis = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "pelvis")
        self.mass = float(m.body_subtreemass[self.pelvis])   # robot only (scene may add bodies)
        # foot bodies and their 4 contact spheres (Menagerie: class "foot")
        self.foot_body = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, n) for n in self.FEET]
        self.foot_geoms = []
        for b in self.foot_body:
            gs = [g for g in range(m.ngeom)
                  if m.geom_bodyid[g] == b and m.geom_type[g] == mujoco.mjtGeom.mjGEOM_SPHERE]
            assert len(gs) == 4, "expected 4 contact spheres per foot"
            self.foot_geoms.append(gs)
        self.pelvis = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "pelvis")
        self.torso = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "torso_link")
        self._scratch = mujoco.MjData(m)

    # ------------------------------------------------------------------ indices
    def aid(self, name: str) -> int:
        return self.act_names.index(name)

    # ------------------------------------------------------------------ poses
    def nominal_qpos(self, knee: float = 0.6, height_margin: float = 0.0) -> np.ndarray:
        """Slightly crouched standing pose with flat feet on the ground (z=0)."""
        m = self.m
        q = np.zeros(m.nq)
        q[3] = 1.0
        a = {n: 0.0 for n in self.act_names}
        for side, s in (("left", 1.0), ("right", -1.0)):
            a[f"{side}_hip_pitch_joint"] = -knee / 2
            a[f"{side}_knee_joint"] = knee
            a[f"{side}_ankle_pitch_joint"] = -knee / 2
            a[f"{side}_shoulder_pitch_joint"] = 0.2
            a[f"{side}_shoulder_roll_joint"] = 0.25 * s
            a[f"{side}_elbow_joint"] = 1.0
        for n, v in a.items():
            q[self.act_qpos[self.aid(n)]] = v
        # put the lowest contact sphere exactly on the floor
        d = self._scratch
        d.qpos[:] = q
        mujoco.mj_kinematics(m, d)
        zmin = min(d.geom_xpos[g][2] - m.geom_size[g][0] for gs in self.foot_geoms for g in gs)
        q[2] -= zmin - height_margin
        return q

    # ------------------------------------------------------------------ kinematics
    def com(self, d=None) -> np.ndarray:
        """Robot CoM = subtree CoM of the pelvis (NOT body 0, which would include tables etc.)."""
        d = d or self.d
        return d.subtree_com[self.pelvis].copy()

    def com_jac(self, d=None) -> np.ndarray:
        d = d or self.d
        J = np.zeros((3, self.nv))
        mujoco.mj_jacSubtreeCom(self.m, d, J, self.pelvis)
        return J

    def point_jac(self, body: int, point: np.ndarray, d=None) -> np.ndarray:
        d = d or self.d
        jp = np.zeros((3, self.nv))
        mujoco.mj_jac(self.m, d, jp, None, point, body)
        return jp

    def body_jac(self, body: int, d=None):
        """6×nv [angular; linear-at-body-origin] Jacobian."""
        d = d or self.d
        jp = np.zeros((3, self.nv)); jr = np.zeros((3, self.nv))
        mujoco.mj_jacBody(self.m, d, jp, jr, body)
        return np.vstack([jr, jp])

    def contact_points(self, foot: int, d=None) -> list[np.ndarray]:
        d = d or self.d
        return [d.geom_xpos[g].copy() for g in self.foot_geoms[foot]]

    def support_polygon(self, feet=(0, 1), d=None) -> np.ndarray:
        """Convex hull (xy) of contact points of the given feet, CCW order."""
        pts = np.array([p[:2] for f in feet for p in self.contact_points(f, d)])
        return convex_hull(pts)

    def drift_terms(self, fns, eps: float = 1e-6):
        """Finite-difference estimate of dJ/dt · v for Jacobian functions.

        fns: list of callables J = fn(data). Returns list of vectors (dJ v).
        Uses q⁺ = q ⊕ v·eps on a scratch MjData (correct quaternion integration).
        """
        m, d, s = self.m, self.d, self._scratch
        v = d.qvel.copy()
        J0 = [fn(d) for fn in fns]
        s.qpos[:] = d.qpos
        mujoco.mj_integratePos(m, s.qpos, v, eps)
        mujoco.mj_kinematics(m, s)
        mujoco.mj_comPos(m, s)
        J1 = [fn(s) for fn in fns]
        return [((b - a) / eps) @ v for a, b in zip(J0, J1)]


def convex_hull(pts: np.ndarray) -> np.ndarray:
    """Andrew's monotone chain. pts: (N,2). Returns hull vertices CCW."""
    P = sorted(map(tuple, pts))
    if len(P) <= 2:
        return np.array(P)

    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    lower, upper = [], []
    for p in P:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)
    for p in reversed(P):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)
    return np.array(lower[:-1] + upper[:-1])


def polygon_halfspaces(hull: np.ndarray, margin: float = 0.0):
    """Return (A, b) with A x <= b describing the CCW polygon shrunk by `margin`."""
    A, b = [], []
    n = len(hull)
    for i in range(n):
        p, q = hull[i], hull[(i + 1) % n]
        e = q - p
        nrm = np.array([e[1], -e[0]])          # outward normal for CCW polygon
        nrm /= np.linalg.norm(nrm)
        A.append(nrm)
        b.append(nrm @ p - margin)
    return np.array(A), np.array(b)


def full_mass_matrix(m: mujoco.MjModel, d: mujoco.MjData, out: np.ndarray) -> np.ndarray:
    """Dense mass matrix; works with both old (qM) and new (MuJoCo ≥ 3.3) APIs."""
    try:
        mujoco.mj_fullM(m, d, out)          # MuJoCo ≥ 3.3: (m, d, dst)
    except TypeError:
        mujoco.mj_fullM(m, out, d.qM)       # older: (m, dst, qM)
    return out


def quat_to_mat(q: np.ndarray) -> np.ndarray:
    R = np.zeros(9)
    mujoco.mju_quat2Mat(R, q)
    return R.reshape(3, 3)


def rot_log(R: np.ndarray) -> np.ndarray:
    """so(3) log map → rotation vector."""
    q = np.zeros(4)
    mujoco.mju_mat2Quat(q, R.flatten())
    v = np.zeros(3)
    mujoco.mju_quat2Vel(v, q, 1.0)
    return v
