"""Two-body orbit example with diagnostics."""

from __future__ import annotations

import numpy as np

from bean_physics.core.diagnostics import linear_momentum, total_energy_gravity
from bean_physics.core.forces import NBodyGravity
from bean_physics.core.integrators import VelocityVerlet
from bean_physics.core.state import ParticlesState, SystemState


if __name__ == "__main__":
    G = 1.0
    m = np.array([1.0, 1.0], dtype=np.float64)
    pos = np.array([[-0.5, 0.0, 0.0], [0.5, 0.0, 0.0]], dtype=np.float64)
    v = np.sqrt(0.5)
    vel = np.array([[0.0, v, 0.0], [0.0, -v, 0.0]], dtype=np.float64)

    state = SystemState(particles=ParticlesState(pos=pos, vel=vel, mass=m))
    model = NBodyGravity(G=G)
    integrator = VelocityVerlet()

    dt = 0.001
    steps = 10_000
    report_every = 500

    delta = state.particles.pos[1] - state.particles.pos[0]
    r = np.linalg.norm(delta)
    r_min = r
    r_max = r
    e0 = total_energy_gravity(state, G=G)

    for step in range(1, steps + 1):
        integrator.step(state, model, dt)
        delta = state.particles.pos[1] - state.particles.pos[0]
        r = np.linalg.norm(delta)
        r_min = min(r_min, r)
        r_max = max(r_max, r)

        if step % report_every == 0:
            p = linear_momentum(state)
            e = total_energy_gravity(state, G=G)
            print(
                f"step {step:5d} | r_min={r_min:.6f} r_max={r_max:.6f} | "
                f"|p|={np.linalg.norm(p):.6e} | dE={e - e0:.6e}"
            )
