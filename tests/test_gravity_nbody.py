from __future__ import annotations

import numpy as np

from bean_physics.core.diagnostics import total_energy_gravity
from bean_physics.core.forces import CompositeModel, NBodyGravity, UniformGravity
from bean_physics.core.integrators import SymplecticEuler, VelocityVerlet
from bean_physics.core.state import ParticlesState, SystemState


def _make_state(pos: np.ndarray, vel: np.ndarray, mass: np.ndarray) -> SystemState:
    return SystemState(particles=ParticlesState(pos=pos, vel=vel, mass=mass))


def test_nbody_shapes_and_self_interaction() -> None:
    model = NBodyGravity(G=1.0)
    state1 = _make_state(
        pos=np.array([[0.0, 0.0, 0.0]], dtype=np.float64),
        vel=np.zeros((1, 3), dtype=np.float64),
        mass=np.array([2.0], dtype=np.float64),
    )
    acc1 = model.acc_particles(state1)
    assert acc1.shape == (1, 3)
    assert np.allclose(acc1, 0.0)

    state2 = _make_state(
        pos=np.array([[-1.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=np.float64),
        vel=np.zeros((2, 3), dtype=np.float64),
        mass=np.array([2.0, 3.0], dtype=np.float64),
    )
    acc2 = model.acc_particles(state2)
    mom_dot = state2.particles.mass[:, None] * acc2
    assert np.allclose(np.sum(mom_dot, axis=0), 0.0, atol=1e-12)


def test_momentum_conservation_gravity_only() -> None:
    pos = np.array(
        [
            [-1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, -1.0, 0.0],
            [0.5, 0.5, 0.0],
        ],
        dtype=np.float64,
    )
    vel = np.array(
        [
            [0.0, 0.1, 0.0],
            [0.0, -0.1, 0.0],
            [-0.1, 0.0, 0.0],
            [0.1, 0.0, 0.0],
            [0.0, 0.0, 0.0],
        ],
        dtype=np.float64,
    )
    mass = np.array([1.0, 1.5, 0.8, 1.2, 0.6], dtype=np.float64)
    state = _make_state(pos=pos, vel=vel, mass=mass)
    model = NBodyGravity(G=1.0, softening=0.01)
    integrator = VelocityVerlet()

    p0 = np.sum(state.particles.vel * state.particles.mass[:, None], axis=0)
    dt = 0.001
    for _ in range(2000):
        integrator.step(state, model, dt)
    p1 = np.sum(state.particles.vel * state.particles.mass[:, None], axis=0)

    tol = 1e-8 * (np.linalg.norm(p0) + 1.0)
    assert np.linalg.norm(p1 - p0) < tol


def test_two_body_orbit_separation_band() -> None:
    G = 1.0
    mass = np.array([1.0, 1.0], dtype=np.float64)
    pos = np.array([[-0.5, 0.0, 0.0], [0.5, 0.0, 0.0]], dtype=np.float64)
    v = np.sqrt(0.5)
    vel = np.array([[0.0, v, 0.0], [0.0, -v, 0.0]], dtype=np.float64)
    state = _make_state(pos=pos, vel=vel, mass=mass)

    model = NBodyGravity(G=G)
    integrator = VelocityVerlet()
    dt = 0.001
    steps = 5000

    sep0 = np.linalg.norm(state.particles.pos[1] - state.particles.pos[0])
    sep_min = sep0
    sep_max = sep0
    for _ in range(steps):
        integrator.step(state, model, dt)
        sep = np.linalg.norm(state.particles.pos[1] - state.particles.pos[0])
        sep_min = min(sep_min, sep)
        sep_max = max(sep_max, sep)

    assert sep_min > 0.9 * sep0
    assert sep_max < 1.1 * sep0


def test_energy_drift_verlet_better_than_symplectic() -> None:
    G = 1.0
    mass = np.array([1.0, 1.0], dtype=np.float64)
    pos = np.array([[-0.5, 0.0, 0.0], [0.5, 0.0, 0.0]], dtype=np.float64)
    v = np.sqrt(0.5)
    vel = np.array([[0.0, v, 0.0], [0.0, -v, 0.0]], dtype=np.float64)

    dt = 0.01
    steps = 2000
    model = NBodyGravity(G=G)

    state_se = _make_state(pos=pos.copy(), vel=vel.copy(), mass=mass)
    state_vv = _make_state(pos=pos.copy(), vel=vel.copy(), mass=mass)

    e0 = total_energy_gravity(state_se, G=G)

    integ_se = SymplecticEuler()
    for _ in range(steps):
        integ_se.step(state_se, model, dt)
    e_se = total_energy_gravity(state_se, G=G)

    integ_vv = VelocityVerlet()
    for _ in range(steps):
        integ_vv.step(state_vv, model, dt)
    e_vv = total_energy_gravity(state_vv, G=G)

    drift_se = abs(e_se - e0)
    drift_vv = abs(e_vv - e0)
    assert drift_vv < drift_se


def test_composite_with_uniform_gravity() -> None:
    g = UniformGravity(g=np.array([0.0, -1.0, 0.0], dtype=np.float64))
    nbody = NBodyGravity(G=1.0)
    comp = CompositeModel(models=[g, nbody])

    pos = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=np.float64)
    vel = np.zeros((2, 3), dtype=np.float64)
    mass = np.array([1.0, 2.0], dtype=np.float64)
    state = _make_state(pos=pos, vel=vel, mass=mass)

    acc_comp = comp.acc_particles(state)
    acc_sum = g.acc_particles(state) + nbody.acc_particles(state)
    assert np.allclose(acc_comp, acc_sum)
