from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

from sim.types import VelocityCommand2D


@dataclass(frozen=True)
class ActuatorConfig:
    """
    Simple actuator model for (v, omega) commands.

    Features:
      - Saturation on v and omega
      - Acceleration (rate) limiting on v and omega (dv/dt, domega/dt)

    Ordering:
      desired -> rate limit -> saturation
    """
    max_v: float = 20.0            # m/s
    max_omega: float = 4.0         # rad/s
    max_v_accel: float = 10.0       # m/s^2
    max_omega_accel: float = 20.0   # rad/s^2


@dataclass
class ActuatorState:
    v: float = 0.0
    omega: float = 0.0


def reset_actuator_state(*, v: float = 0.0, omega: float = 0.0) -> ActuatorState:
    return ActuatorState(v=float(v), omega=float(omega))


def actuator_step(
    cmd_des: VelocityCommand2D,
    state: ActuatorState,
    dt: float,
    cfg: ActuatorConfig,
) -> Tuple[VelocityCommand2D, ActuatorState]:
    if dt <= 0.0:
        raise ValueError(f"dt must be > 0, got {dt}")

    # 0) (Optional) clamp desired into physically meaningful range
    v_des = _clamp(float(cmd_des.v), -abs(cfg.max_v), abs(cfg.max_v))
    w_des = _clamp(float(cmd_des.omega), -abs(cfg.max_omega), abs(cfg.max_omega))

    # 1) Rate limit (acceleration / angular accel)
    v_out = _rate_limit_step(x=state.v, u=v_des, max_rate=cfg.max_v_accel, dt=dt)
    w_out = _rate_limit_step(x=state.omega, u=w_des, max_rate=cfg.max_omega_accel, dt=dt)

    # 2) Output saturation (final hard clamp)
    v_out = _clamp(v_out, -abs(cfg.max_v), abs(cfg.max_v))
    w_out = _clamp(w_out, -abs(cfg.max_omega), abs(cfg.max_omega))

    new_state = ActuatorState(v=v_out, omega=w_out)
    return VelocityCommand2D(v=v_out, omega=w_out), new_state


def _clamp(x: float, lo: float, hi: float) -> float:
    return lo if x < lo else hi if x > hi else x


def _rate_limit_step(x: float, u: float, max_rate: float, dt: float) -> float:
    max_rate = abs(float(max_rate))
    if max_rate <= 0.0:
        return float(x)
    dx_max = max_rate * dt
    dx = u - x
    if dx > dx_max:
        return x + dx_max
    if dx < -dx_max:
        return x - dx_max
    return float(u)
