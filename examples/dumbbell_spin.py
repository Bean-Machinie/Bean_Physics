"""Torque-free dumbbell rotation example."""

from __future__ import annotations

import numpy as np

from bean_physics.core.diagnostics import angular_momentum_body, rotational_ke_body
from bean_physics.core.forces import RigidBodyForces
from bean_physics.core.integrators import VelocityVerlet
from bean_physics.core.math.quat import quat_to_rotmat
from bean_physics.core.rigid_body.mass_properties import mass_properties
from bean_physics.core.state import RigidBodiesState, SystemState


if __name__ == "__main__":
    points_body = np.array([[-1.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=np.float64)
    masses = np.array([1.0, 1.0], dtype=np.float64)
    total_mass, com_body, inertia_body = mass_properties(points_body, masses)

    pos = np.array([[0.0, 0.0, 0.0]], dtype=np.float64)
    vel = np.array([[0.0, 0.0, 0.0]], dtype=np.float64)
    quat = np.array([[1.0, 0.0, 0.0, 0.0]], dtype=np.float64)
    omega_body = np.array([[0.3, 0.4, 0.5]], dtype=np.float64)
    rb = RigidBodiesState(
        pos=pos,
        vel=vel,
        quat=quat,
        omega=omega_body,
        mass=np.array([total_mass]),
        inertia_body=inertia_body,
    )
    state = SystemState(rigid_bodies=rb)

    model = RigidBodyForces(mass=np.array([total_mass]), inertia_body=inertia_body)
    model.set_applied_forces(
        body_index=np.array([], dtype=np.int64),
        forces_body=np.zeros((0, 3), dtype=np.float64),
        points_body=np.zeros((0, 3), dtype=np.float64),
    )

    rot0 = quat_to_rotmat(state.rigid_bodies.quat)
    l_body0 = angular_momentum_body(inertia_body, omega_body[0])
    l_world0 = rot0[0] @ l_body0
    l0 = np.linalg.norm(l_world0)
    ke0 = rotational_ke_body(inertia_body, omega_body[0])

    integrator = VelocityVerlet()
    dt = 1e-3
    steps = 5000
    max_l_drift = 0.0
    max_ke_drift = 0.0
    for _ in range(steps):
        integrator.step(state, model, dt)
        rot = quat_to_rotmat(state.rigid_bodies.quat)
        l_body = angular_momentum_body(inertia_body, state.rigid_bodies.omega[0])
        l_world = rot[0] @ l_body
        l_mag = np.linalg.norm(l_world)
        max_l_drift = max(max_l_drift, abs(l_mag - l0))

        ke = rotational_ke_body(inertia_body, state.rigid_bodies.omega[0])
        max_ke_drift = max(max_ke_drift, abs(ke - ke0))

    q_norm = np.linalg.norm(state.rigid_bodies.quat[0])
    sim_time = steps * dt
    print("steps:", steps)
    print("dt:", dt)
    print("sim time:", sim_time)
    print("initial |L0|:", l0)
    print("max |L| drift (abs):", max_l_drift)
    print("max |L| drift (rel):", max_l_drift / l0)
    print("max KE_rot drift (abs):", max_ke_drift)
    print("max KE_rot drift (rel):", max_ke_drift / ke0)
    print("quat norm:", q_norm)
