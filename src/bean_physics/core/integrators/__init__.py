"""Integrator interfaces and implementations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from ..forces.base import Model
from ..math.quat import quat_integrate_expmap
from ..state.system import SystemState


class Integrator(Protocol):
    def step(self, state: SystemState, model: Model, dt: float) -> None:
        """Advance state by one fixed step (mutating)."""


@dataclass(slots=True)
class SymplecticEuler:
    def step(self, state: SystemState, model: Model, dt: float) -> None:
        if state.particles is not None:
            a = model.acc_particles(state)
            state.particles.vel += a * dt
            state.particles.pos += state.particles.vel * dt

        if state.rigid_bodies is not None:
            a = model.acc_rigid(state)
            state.rigid_bodies.vel += a * dt
            state.rigid_bodies.pos += state.rigid_bodies.vel * dt

            alpha = model.alpha_rigid(state)
            state.rigid_bodies.omega += alpha * dt
            state.rigid_bodies.quat = quat_integrate_expmap(
                state.rigid_bodies.quat, state.rigid_bodies.omega, dt
            )


@dataclass(slots=True)
class VelocityVerlet:
    def step(self, state: SystemState, model: Model, dt: float) -> None:
        if state.particles is not None:
            a = model.acc_particles(state)
            state.particles.pos += state.particles.vel * dt + 0.5 * a * dt * dt
            a_next = model.acc_particles(state)
            state.particles.vel += 0.5 * (a + a_next) * dt

        if state.rigid_bodies is not None:
            a = model.acc_rigid(state)
            state.rigid_bodies.pos += state.rigid_bodies.vel * dt + 0.5 * a * dt * dt
            a_next = model.acc_rigid(state)
            state.rigid_bodies.vel += 0.5 * (a + a_next) * dt

            alpha = model.alpha_rigid(state)
            state.rigid_bodies.omega += alpha * dt
            state.rigid_bodies.quat = quat_integrate_expmap(
                state.rigid_bodies.quat, state.rigid_bodies.omega, dt
            )
