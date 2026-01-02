"""Small N-body cluster example (deterministic)."""

from __future__ import annotations

import numpy as np

from bean_physics.core.diagnostics import linear_momentum, total_energy_gravity
from bean_physics.core.forces import NBodyGravity
from bean_physics.core.integrators import VelocityVerlet
from bean_physics.core.state import ParticlesState, SystemState


if __name__ == "__main__":
    rng = np.random.default_rng(123)
    n = 50
    pos = rng.normal(scale=1.0, size=(n, 3))
    vel = rng.normal(scale=0.1, size=(n, 3))
    mass = rng.uniform(low=0.5, high=2.0, size=(n,))

    state = SystemState(
        particles=ParticlesState(
            pos=pos.astype(np.float64),
            vel=vel.astype(np.float64),
            mass=mass.astype(np.float64),
        )
    )
    model = NBodyGravity(G=1.0, softening=0.05)
    integrator = VelocityVerlet()

    dt = 0.001
    steps = 2000
    e0 = total_energy_gravity(state, G=1.0, softening=0.05)

    for _ in range(steps):
        integrator.step(state, model, dt)

    any_nan = np.isnan(state.particles.pos).any() or np.isnan(state.particles.vel).any()
    p = linear_momentum(state)
    e = total_energy_gravity(state, G=1.0, softening=0.05)

    print("any NaN:", any_nan)
    print("total momentum:", p)
    print("energy drift:", e - e0)
