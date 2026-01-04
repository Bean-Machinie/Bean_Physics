"""Hohmann transfer and rocket equation helpers."""

from __future__ import annotations

from dataclasses import dataclass
from math import ceil, exp, log, pi, sqrt

import numpy as np

from ..core.impulse_events import ImpulseEvent, ImpulseSchedule
from ..core.state.system import SystemState
from ..io.units import UnitsConfig, from_si

G0 = 9.80665


@dataclass(frozen=True, slots=True)
class HohmannResult:
    mu: float
    r1: float
    r2: float
    v1: float
    v2: float
    v_p: float
    v_a: float
    dv1: float
    dv2: float
    dv_total: float
    a_transfer: float
    t_transfer: float


@dataclass(frozen=True, slots=True)
class HohmannPlanInputs:
    central_body_id: str
    spacecraft_id: str
    r1: float
    r2: float
    body_radius_m: float
    r1_mode: str
    r2_mode: str
    r1_altitude_m: float
    r2_altitude_m: float
    coast_time_s: float
    burn2_mode: str
    burn2_time_s: float
    dry_mass_kg: float
    prop_mass_kg: float
    isp_s: float
    thrust_n: float

    def to_metadata(self) -> dict[str, object]:
        return {
            "central_body_id": self.central_body_id,
            "spacecraft_id": self.spacecraft_id,
            "r1_mode": self.r1_mode,
            "r2_mode": self.r2_mode,
            "body_radius_m": self.body_radius_m,
            "r1_altitude_m": self.r1_altitude_m,
            "r1_radius_m": self.r1,
            "r2_altitude_m": self.r2_altitude_m,
            "r2_radius_m": self.r2,
            "coast_time_s": self.coast_time_s,
            "burn2_mode": self.burn2_mode,
            "burn2_time_s": self.burn2_time_s,
            "dry_mass_kg": self.dry_mass_kg,
            "prop_mass_kg": self.prop_mass_kg,
            "isp_s": self.isp_s,
            "thrust_n": self.thrust_n,
        }


def compute_hohmann(mu: float, r1: float, r2: float) -> HohmannResult:
    if mu <= 0.0:
        raise ValueError("mu must be > 0")
    if r1 <= 0.0 or r2 <= 0.0:
        raise ValueError("r1 and r2 must be > 0")
    a_transfer = 0.5 * (r1 + r2)
    v1 = sqrt(mu / r1)
    v2 = sqrt(mu / r2)
    v_p = sqrt(mu * (2.0 / r1 - 1.0 / a_transfer))
    v_a = sqrt(mu * (2.0 / r2 - 1.0 / a_transfer))
    dv1 = v_p - v1
    dv2 = v2 - v_a
    dv_total = dv1 + dv2
    t_transfer = pi * sqrt((a_transfer**3) / mu)
    return HohmannResult(
        mu=mu,
        r1=r1,
        r2=r2,
        v1=v1,
        v2=v2,
        v_p=v_p,
        v_a=v_a,
        dv1=dv1,
        dv2=dv2,
        dv_total=dv_total,
        a_transfer=a_transfer,
        t_transfer=t_transfer,
    )


def rocket_equation_delta_v(m0: float, mf: float, isp: float) -> float:
    if isp <= 0.0:
        raise ValueError("isp must be > 0")
    if m0 <= 0.0 or mf <= 0.0:
        raise ValueError("m0 and mf must be > 0")
    if m0 < mf:
        raise ValueError("m0 must be >= mf")
    ve = isp * G0
    return ve * log(m0 / mf)


def rocket_equation_propellant(m_dry: float, delta_v: float, isp: float) -> tuple[float, float, float]:
    if m_dry <= 0.0:
        raise ValueError("m_dry must be > 0")
    if isp <= 0.0:
        raise ValueError("isp must be > 0")
    ve = isp * G0
    mass_ratio = exp(delta_v / ve)
    prop_required = m_dry * (mass_ratio - 1.0)
    return prop_required, mass_ratio, ve


def burn_mass_sequence(m0: float, dv_list: list[float], ve: float) -> list[tuple[float, float]]:
    if m0 <= 0.0:
        raise ValueError("m0 must be > 0")
    if ve <= 0.0:
        raise ValueError("ve must be > 0")
    masses: list[tuple[float, float]] = []
    current = m0
    for dv in dv_list:
        if dv < 0.0:
            raise ValueError("dv values must be >= 0")
        m_after = current * exp(-dv / ve)
        masses.append((current, m_after))
        current = m_after
    return masses


def prograde_unit(velocity: np.ndarray) -> np.ndarray:
    v = np.asarray(velocity, dtype=np.float64)
    norm = float(np.linalg.norm(v))
    if norm == 0.0:
        raise ValueError("velocity norm is zero")
    return v / norm


def preview_velocity_at_time(
    state: SystemState,
    model: object,
    integrator: object,
    dt: float,
    t: float,
    target_type: str,
    target_index: int,
    impulse_events: list[ImpulseEvent] | None = None,
    max_steps: int = 20000,
) -> np.ndarray:
    if t < 0.0:
        raise ValueError("t must be >= 0")
    if max_steps <= 0:
        raise ValueError("max_steps must be > 0")
    preview = state.clone()
    schedule = None
    if impulse_events:
        schedule = ImpulseSchedule(
            events=sorted(impulse_events, key=lambda evt: evt.t)
        )
    current_t = 0.0
    step = 0
    if t > 0.0 and dt > 0.0:
        estimated_steps = int(ceil(t / dt))
        if estimated_steps > max_steps:
            dt = t / max_steps
    while current_t < t:
        next_dt = min(dt, t - current_t)
        prev_t = current_t
        current_t = current_t + next_dt
        step += 1
        if schedule is not None:
            schedule.fire_for_window(
                preview,
                prev_t=prev_t,
                new_t=current_t,
                include_start=(prev_t == 0.0),
            )
        integrator.step(preview, model, next_dt)
    if target_type == "particle":
        if preview.particles is None:
            raise ValueError("particle target missing from state")
        return preview.particles.vel[target_index].copy()
    if target_type == "rigid_body":
        if preview.rigid_bodies is None:
            raise ValueError("rigid body target missing from state")
        return preview.rigid_bodies.vel[target_index].copy()
    raise ValueError(f"unsupported target type: {target_type}")


def prograde_delta_v(velocity: np.ndarray, dv_mag: float) -> np.ndarray:
    return prograde_unit(velocity) * float(dv_mag)


def build_hohmann_impulse_events(
    mu: float,
    plan: HohmannPlanInputs,
    dv1_dir: np.ndarray,
    dv2_dir: np.ndarray,
    units_cfg: UnitsConfig,
) -> tuple[list[dict[str, object]], HohmannResult, float, float]:
    res = compute_hohmann(mu, plan.r1, plan.r2)
    t1 = float(plan.coast_time_s)
    t2 = float(plan.burn2_time_s)
    if plan.burn2_mode.lower().startswith("auto"):
        t2 = t1 + res.t_transfer
    dv1_world = prograde_delta_v(dv1_dir, res.dv1)
    dv2_world = prograde_delta_v(dv2_dir, res.dv2)
    impulse_events = [
        {
            "t": float(from_si(t1, "time", units_cfg)),
            "target": plan.spacecraft_id,
            "delta_v_world": from_si(dv1_world, "velocity", units_cfg).tolist(),
            "label": f"Burn 1: dv={res.dv1:.3f} m/s @ t={t1:.1f}s",
        },
        {
            "t": float(from_si(t2, "time", units_cfg)),
            "target": plan.spacecraft_id,
            "delta_v_world": from_si(dv2_world, "velocity", units_cfg).tolist(),
            "label": f"Burn 2: dv={res.dv2:.3f} m/s @ t={t2:.1f}s",
        },
    ]
    return impulse_events, res, t1, t2


def build_hohmann_metadata(plan: HohmannPlanInputs) -> dict[str, object]:
    return {
        "mission_analysis": {
            "kind": "hohmann",
            "hohmann": plan.to_metadata(),
        }
    }


def apply_hohmann_plan(
    defn: dict[str, object],
    mu: float,
    plan: HohmannPlanInputs,
    dv1_dir: np.ndarray,
    dv2_dir: np.ndarray,
    units_cfg: UnitsConfig,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    impulse_events, _, _, _ = build_hohmann_impulse_events(
        mu, plan, dv1_dir, dv2_dir, units_cfg
    )
    meta_update = build_hohmann_metadata(plan)
    return impulse_events, meta_update
