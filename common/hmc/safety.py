"""Fail-safe layer: state machine, monitors, fault injection.

States
------
STAND     nominal whole-body control
DEGRADED  a non-critical limit is being approached → go to a lower, safer posture
SAFE_SIT  controlled lowering of the CoM (still WBC), then DAMPING
DAMPING   τ = −K_d q̇  (joints become viscous; robot folds softly instead of stiffly)
ESTOP     τ = 0 (latched; needs manual reset)

Allowed transitions (anything else is rejected and logged):
  STAND → DEGRADED | DAMPING | ESTOP
  DEGRADED → SAFE_SIT | DAMPING | ESTOP
  SAFE_SIT → DAMPING | ESTOP
  DAMPING → ESTOP
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

import numpy as np


class Mode(str, Enum):
    STAND = "STAND"
    DEGRADED = "DEGRADED"
    SAFE_SIT = "SAFE_SIT"
    DAMPING = "DAMPING"
    ESTOP = "ESTOP"


ALLOWED = {
    Mode.STAND: {Mode.DEGRADED, Mode.DAMPING, Mode.ESTOP},
    Mode.DEGRADED: {Mode.SAFE_SIT, Mode.DAMPING, Mode.ESTOP},
    Mode.SAFE_SIT: {Mode.DAMPING, Mode.ESTOP},
    Mode.DAMPING: {Mode.ESTOP},
    Mode.ESTOP: set(),
}


@dataclass
class SafetyConfig:
    max_qp_fail: int = 5             # consecutive QP failures tolerated (hold last τ)
    max_invalid_sensor: int = 5      # consecutive invalid (NaN/Inf) samples tolerated
    tilt_limit_deg: float = 25.0     # pelvis tilt → fall imminent
    watchdog_s: float = 0.010        # max age of the last command seen by the "driver"
    act_mismatch_ratio: float = 0.5  # |measured − commanded| / (|commanded| + 5 Nm)
    act_mismatch_time: float = 0.020
    thermal_limit: float = 1.0       # normalized I²t state that triggers DEGRADED
    thermal_tau: float = 20.0        # thermal time constant [s]
    damping_kd: float = 4.0


@dataclass
class SafetyEvent:
    t: float
    frm: Mode
    to: Mode
    reason: str


@dataclass
class SafetySupervisor:
    nu: int
    tau_cont: np.ndarray                     # continuous torque rating per joint
    cfg: SafetyConfig = field(default_factory=SafetyConfig)
    mode: Mode = Mode.STAND
    events: list = field(default_factory=list)
    qp_fail: int = 0
    invalid_sensor: int = 0
    mismatch_t: float = 0.0
    thermal: np.ndarray | None = None
    last_cmd_time: float = 0.0
    last_good_tau: np.ndarray | None = None

    def __post_init__(self):
        self.thermal = np.zeros(self.nu)
        self.last_good_tau = np.zeros(self.nu)

    # --------------------------------------------------------------- transitions
    def request(self, t: float, to: Mode, reason: str) -> bool:
        if to == self.mode:
            return True
        if to not in ALLOWED[self.mode]:
            return False
        self.events.append(SafetyEvent(t, self.mode, to, reason))
        self.mode = to
        return True

    # --------------------------------------------------------------- monitors
    def check_sensors(self, t, qpos, qvel) -> bool:
        ok = bool(np.all(np.isfinite(qpos)) and np.all(np.isfinite(qvel)))
        self.invalid_sensor = 0 if ok else self.invalid_sensor + 1
        if self.invalid_sensor > self.cfg.max_invalid_sensor:
            self.request(t, Mode.DAMPING, f"sensor invalid > {self.cfg.max_invalid_sensor} chu kỳ")
        return ok

    def check_qp(self, t, ok: bool, tau):
        if ok:
            self.qp_fail = 0
            self.last_good_tau = tau.copy()
        else:
            self.qp_fail += 1
            if self.qp_fail > self.cfg.max_qp_fail:
                self.request(t, Mode.DAMPING, f"QP thất bại {self.qp_fail} chu kỳ liên tiếp")

    def check_tilt(self, t, tilt_rad: float):
        if np.degrees(tilt_rad) > self.cfg.tilt_limit_deg and self.mode in (Mode.STAND, Mode.DEGRADED, Mode.SAFE_SIT):
            self.request(t, Mode.DAMPING, f"nghiêng {np.degrees(tilt_rad):.0f}° > {self.cfg.tilt_limit_deg:.0f}° (sắp ngã)")

    def check_watchdog(self, t):
        if t - self.last_cmd_time > self.cfg.watchdog_s and self.mode in (Mode.STAND, Mode.DEGRADED, Mode.SAFE_SIT):
            self.request(t, Mode.DAMPING, f"watchdog: lệnh cũ {1000 * (t - self.last_cmd_time):.0f} ms")

    def check_actuators(self, t, dt, cmd, measured):
        err = np.abs(measured - cmd) / (np.abs(cmd) + 5.0)
        if np.any(err > self.cfg.act_mismatch_ratio):
            self.mismatch_t += dt
        else:
            self.mismatch_t = 0.0
        if self.mismatch_t > self.cfg.act_mismatch_time:
            j = int(np.argmax(err))
            self.request(t, Mode.DAMPING, f"actuator #{j} lệch mô-men (lỗi khớp)")

    def update_thermal(self, t, dt, tau):
        """Normalized I²t model: dT/dt = ((τ/τ_cont)² − T) / τ_th."""
        c = self.cfg
        self.thermal += dt * ((tau / self.tau_cont) ** 2 - self.thermal) / c.thermal_tau
        if self.mode == Mode.STAND and np.any(self.thermal > c.thermal_limit):
            j = int(np.argmax(self.thermal))
            self.request(t, Mode.DEGRADED, f"quá nhiệt actuator #{j}")

    def damping_torque(self, qvel_act):
        return -self.cfg.damping_kd * qvel_act
