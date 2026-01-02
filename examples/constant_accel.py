"""Constant-acceleration example (toy model)."""

from __future__ import annotations

import numpy as np

from bean_physics.core.integrators import SymplecticEuler
from bean_physics.core.state import ParticlesState, SystemState


class ConstantAccelerationModel:
    def __init__(self, accel: np.ndarray) -> None:
        self.accel = np.asarray(accel, dtype=np.float64)

    def acc_particles(self, state: SystemState) -> np.ndarray:
        if state.particles is None:
            return np.zeros((0, 3), dtype=np.float64)
        n = state.particles.pos.shape[0]
        return np.broadcast_to(self.accel, (n, 3)).copy()

    def acc_rigid(self, state: SystemState) -> np.ndarray:
        if state.rigid_bodies is None:
            return np.zeros((0, 3), dtype=np.float64)
        m = state.rigid_bodies.pos.shape[0]
        return np.zeros((m, 3), dtype=np.float64)

    def alpha_rigid(self, state: SystemState) -> np.ndarray:
        if state.rigid_bodies is None:
            return np.zeros((0, 3), dtype=np.float64)
        m = state.rigid_bodies.pos.shape[0]
        return np.zeros((m, 3), dtype=np.float64)


if __name__ == "__main__":
    p0 = np.array([[0.0, 0.0, 0.0]], dtype=np.float64)
    v0 = np.array([[1.0, 0.0, 0.0]], dtype=np.float64)
    m = np.array([1.0], dtype=np.float64)

    state = SystemState(particles=ParticlesState(pos=p0, vel=v0, mass=m))
    model = ConstantAccelerationModel(accel=np.array([0.0, 1.0, 0.0]))
    integrator = SymplecticEuler()

    dt = 0.001
    steps = 1000
    for _ in range(steps):
        integrator.step(state, model, dt)

    t = steps * dt
    # Analytic expectation:
    # v(t) = v0 + a * t
    # p(t) = p0 + v0 * t + 0.5 * a * t^2
    print("final pos:", state.particles.pos[0])
    print("final vel:", state.particles.vel[0])