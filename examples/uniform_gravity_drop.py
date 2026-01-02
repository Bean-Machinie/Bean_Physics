"""Uniform gravity drop example (toy model)."""

from __future__ import annotations

import numpy as np

from bean_physics.core.diagnostics import (
    center_of_mass,
    kinetic_energy,
    linear_momentum,
    total_mass,
)
from bean_physics.core.forces import UniformGravity
from bean_physics.core.integrators import VelocityVerlet
from bean_physics.core.state import ParticlesState, SystemState


if __name__ == "__main__":
    p0 = np.array([[0.0, 0.0, 0.0]], dtype=np.float64)
    v0 = np.array([[1.0, 5.0, 0.0]], dtype=np.float64)
    m = np.array([2.0], dtype=np.float64)

    state = SystemState(particles=ParticlesState(pos=p0, vel=v0, mass=m))
    model = UniformGravity(g=np.array([0.0, -9.81, 0.0], dtype=np.float64))
    integrator = VelocityVerlet()

    dt = 0.001
    steps = 1000
    for _ in range(steps):
        integrator.step(state, model, dt)

    print("final pos:", state.particles.pos[0])
    print("final vel:", state.particles.vel[0])
    print("total mass:", total_mass(state))
    print("center of mass:", center_of_mass(state))
    print("linear momentum:", linear_momentum(state))
    print("kinetic energy:", kinetic_energy(state))