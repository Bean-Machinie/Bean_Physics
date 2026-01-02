"""Simulation run loop with optional sampling."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

from .forces.base import Model
from .integrators import Integrator
from .state.system import SystemState


@dataclass(slots=True)
class RunResult:
    final_state: SystemState
    time: np.ndarray | None = None
    particles_pos: np.ndarray | None = None
    particles_vel: np.ndarray | None = None
    rigid_pos: np.ndarray | None = None
    rigid_quat: np.ndarray | None = None
    rigid_omega_body: np.ndarray | None = None


def run(
    state: SystemState,
    model: Model,
    integrator: Integrator,
    dt: float,
    steps: int,
    sample_every: int | None = None,
    callback: Callable[[int, SystemState], None] | None = None,
) -> RunResult:
    if sample_every is not None and sample_every <= 0:
        raise ValueError("sample_every must be > 0")

    times: list[float] = []
    p_pos: list[np.ndarray] = []
    p_vel: list[np.ndarray] = []
    r_pos: list[np.ndarray] = []
    r_quat: list[np.ndarray] = []
    r_omega: list[np.ndarray] = []

    def sample(step: int) -> None:
        t = step * dt
        times.append(t)
        if state.particles is not None:
            p_pos.append(state.particles.pos.copy())
            p_vel.append(state.particles.vel.copy())
        if state.rigid_bodies is not None:
            r_pos.append(state.rigid_bodies.pos.copy())
            r_quat.append(state.rigid_bodies.quat.copy())
            r_omega.append(state.rigid_bodies.omega.copy())

    if sample_every is not None:
        sample(0)

    for step in range(1, steps + 1):
        integrator.step(state, model, dt)
        if callback is not None:
            callback(step, state)
        if sample_every is not None and step % sample_every == 0:
            sample(step)

    if sample_every is None:
        return RunResult(final_state=state)

    return RunResult(
        final_state=state,
        time=np.asarray(times, dtype=np.float64),
        particles_pos=np.asarray(p_pos, dtype=np.float64) if p_pos else None,
        particles_vel=np.asarray(p_vel, dtype=np.float64) if p_vel else None,
        rigid_pos=np.asarray(r_pos, dtype=np.float64) if r_pos else None,
        rigid_quat=np.asarray(r_quat, dtype=np.float64) if r_quat else None,
        rigid_omega_body=np.asarray(r_omega, dtype=np.float64) if r_omega else None,
    )
