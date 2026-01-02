from __future__ import annotations

import numpy as np

from bean_physics.core.integrators import SymplecticEuler, VelocityVerlet
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


def _run_integrator(integrator, dt: float, steps: int) -> SystemState:
    p0 = np.array([[0.0, 0.0, 0.0]], dtype=np.float64)
    v0 = np.array([[1.0, -2.0, 0.5]], dtype=np.float64)
    m = np.array([1.0], dtype=np.float64)
    state = SystemState(particles=ParticlesState(pos=p0, vel=v0, mass=m))
    model = ConstantAccelerationModel(accel=np.array([0.1, 0.2, -0.3]))

    for _ in range(steps):
        integrator.step(state, model, dt)
    return state


def test_integrators_constant_accel_accuracy() -> None:
    dt = 0.01
    steps = 1000
    t = dt * steps

    p0 = np.array([0.0, 0.0, 0.0])
    v0 = np.array([1.0, -2.0, 0.5])
    a = np.array([0.1, 0.2, -0.3])

    expected_v = v0 + a * t
    expected_p = p0 + v0 * t + 0.5 * a * t * t
    expected_p_se = expected_p + 0.5 * a * dt * t

    state_se = _run_integrator(SymplecticEuler(), dt, steps)
    state_vv = _run_integrator(VelocityVerlet(), dt, steps)

    assert np.allclose(state_se.particles.vel[0], expected_v, atol=1e-3)
    assert np.allclose(state_se.particles.pos[0], expected_p_se, atol=1e-8)

    assert np.allclose(state_vv.particles.vel[0], expected_v, atol=1e-8)
    assert np.allclose(state_vv.particles.pos[0], expected_p, atol=1e-6)


def test_determinism_no_randomness() -> None:
    dt = 0.02
    steps = 50
    integrator = VelocityVerlet()

    state_a = _run_integrator(integrator, dt, steps)
    state_b = _run_integrator(integrator, dt, steps)

    assert np.array_equal(state_a.particles.pos, state_b.particles.pos)
    assert np.array_equal(state_a.particles.vel, state_b.particles.vel)
