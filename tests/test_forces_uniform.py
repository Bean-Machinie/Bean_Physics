from __future__ import annotations

import numpy as np
from bean_physics.core.forces import CompositeModel, UniformGravity
from bean_physics.core.integrators import VelocityVerlet
from bean_physics.core.state import ParticlesState, SystemState


def _make_state(n: int) -> SystemState:
    pos = np.zeros((n, 3), dtype=np.float64)
    vel = np.zeros((n, 3), dtype=np.float64)
    mass = np.ones(n, dtype=np.float64)
    return SystemState(particles=ParticlesState(pos=pos, vel=vel, mass=mass))


def test_uniform_gravity_shape_and_values() -> None:
    model = UniformGravity(g=np.array([0.0, -9.81, 0.0], dtype=np.float64))

    state0 = _make_state(0)
    acc0 = model.acc_particles(state0)
    assert acc0.shape == (0, 3)

    state3 = _make_state(3)
    acc3 = model.acc_particles(state3)
    assert acc3.shape == (3, 3)
    assert np.allclose(acc3, np.array([0.0, -9.81, 0.0]))


def test_composite_model_sums_acceleration() -> None:
    g1 = UniformGravity(g=np.array([0.0, -9.81, 0.0], dtype=np.float64))
    g2 = UniformGravity(g=np.array([1.0, 0.0, 0.0], dtype=np.float64))
    comp = CompositeModel(models=[g1, g2])

    state = _make_state(2)
    acc = comp.acc_particles(state)
    expected = np.array([1.0, -9.81, 0.0], dtype=np.float64)
    assert np.allclose(acc, expected)


def test_uniform_gravity_integrator_matches_analytic() -> None:
    dt = 0.01
    steps = 1000
    t = dt * steps

    p0 = np.array([[0.0, 0.0, 0.0]], dtype=np.float64)
    v0 = np.array([[1.0, 2.0, -1.0]], dtype=np.float64)
    p0_init = p0.copy()
    v0_init = v0.copy()
    m = np.array([1.0], dtype=np.float64)
    state = SystemState(particles=ParticlesState(pos=p0, vel=v0, mass=m))

    g = np.array([0.0, -9.81, 0.0], dtype=np.float64)
    model = UniformGravity(g=g)
    integrator = VelocityVerlet()

    for _ in range(steps):
        integrator.step(state, model, dt)

    expected_v = v0_init[0] + g * t
    expected_p = p0_init[0] + v0_init[0] * t + 0.5 * g * t * t

    assert np.allclose(state.particles.vel[0], expected_v, atol=1e-8)
    assert np.allclose(state.particles.pos[0], expected_p, atol=1e-6)
