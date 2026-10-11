"""Simulation loop utilities.

Correct control pattern in MuJoCo:
    mj_step1(m, d)     # kinematics/dynamics terms for the *current* state
    d.ctrl[:] = u      # controller reads consistent xpos, subtree_com, Jacobians...
    mj_step2(m, d)     # integrate
Using plain mj_step() and reading d.xpos afterwards gives quantities that lag one
step behind d.qpos — a subtle bug that breaks finite-difference drift terms.
"""
from __future__ import annotations

from dataclasses import dataclass

import mujoco
import numpy as np


@dataclass
class Push:
    t0: float
    duration: float
    force: np.ndarray        # world-frame force [N] at body COM
    body: int

    def active(self, t: float) -> bool:
        return self.t0 <= t < self.t0 + self.duration


def apply_pushes(d: mujoco.MjData, pushes: list[Push]):
    d.xfrc_applied[:] = 0.0
    for p in pushes:
        if p.active(d.time):
            d.xfrc_applied[p.body, :3] += p.force


def horizontal_push(body: int, t0: float, magnitude: float, angle_deg: float, duration: float = 0.1) -> Push:
    a = np.deg2rad(angle_deg)
    return Push(t0, duration, magnitude * np.array([np.cos(a), np.sin(a), 0.0]), body)


def tilt_angle(d: mujoco.MjData, body: int) -> float:
    """Angle [rad] between the body's z-axis and world z."""
    z = d.xmat[body].reshape(3, 3)[:, 2]
    return float(np.arccos(np.clip(z[2], -1.0, 1.0)))
