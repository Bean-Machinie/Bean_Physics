from __future__ import annotations

import numpy as np

from bean_physics.core.forces import RigidBodyForces
from bean_physics.core.integrators import VelocityVerlet
from bean_physics.core.math.quat import quat_to_rotmat
from bean_physics.core.rigid_body.mass_properties import box_inertia_body
from bean_physics.core.state import RigidBodiesState, SystemState
from bean_physics.io.scenario import scenario_to_runtime


def _make_state(omega_body: np.ndarray) -> SystemState:
    pos = np.zeros((1, 3), dtype=np.float64)
    vel = np.zeros((1, 3), dtype=np.float64)
    quat = np.array([[1.0, 0.0, 0.0, 0.0]], dtype=np.float64)
    mass = np.array([1.0], dtype=np.float64)
    inertia = np.eye(3, dtype=np.float64)
    rb = RigidBodiesState(
        pos=pos,
        vel=vel,
        quat=quat,
        omega=omega_body,
        mass=mass,
        inertia_body=inertia,
    )
    return SystemState(rigid_bodies=rb)


def test_force_at_com_no_torque() -> None:
    state = _make_state(omega_body=np.zeros((1, 3), dtype=np.float64))
    model = RigidBodyForces(mass=np.array([2.0]), inertia_body=np.eye(3))
    model.set_applied_forces(
        body_index=np.array([0]),
        forces_world=np.array([[2.0, 4.0, 6.0]], dtype=np.float64),
        points_body=np.array([[0.0, 0.0, 0.0]], dtype=np.float64),
    )

    acc = model.acc_rigid(state)
    alpha = model.alpha_rigid(state)
    assert np.allclose(acc[0], np.array([1.0, 2.0, 3.0]))
    assert np.allclose(alpha[0], np.zeros(3))


def test_equal_opposite_forces_pure_torque() -> None:
    state = _make_state(omega_body=np.zeros((1, 3), dtype=np.float64))
    model = RigidBodyForces(mass=np.array([1.0]), inertia_body=np.eye(3))
    model.set_applied_forces(
        body_index=np.array([0, 0]),
        forces_world=np.array([[0.0, 1.0, 0.0], [0.0, -1.0, 0.0]], dtype=np.float64),
        points_body=np.array([[1.0, 0.0, 0.0], [-1.0, 0.0, 0.0]], dtype=np.float64),
    )

    acc = model.acc_rigid(state)
    alpha = model.alpha_rigid(state)
    assert np.allclose(acc[0], np.zeros(3))
    assert np.allclose(alpha[0], np.array([0.0, 0.0, 2.0]))


def test_torque_free_angular_momentum_magnitude() -> None:
    omega0 = np.array([[0.3, 0.4, 0.5]], dtype=np.float64)
    state = _make_state(omega_body=omega0.copy())
    inertia = np.array([[[2.0, 0.0, 0.0], [0.0, 3.0, 0.0], [0.0, 0.0, 4.0]]])
    model = RigidBodyForces(mass=np.array([1.0]), inertia_body=inertia)
    model.set_applied_forces(
        body_index=np.array([], dtype=np.int64),
        forces_world=np.zeros((0, 3), dtype=np.float64),
        points_body=np.zeros((0, 3), dtype=np.float64),
    )

    rot0 = quat_to_rotmat(state.rigid_bodies.quat)
    l_body0 = np.einsum("bij,bj->bi", inertia, state.rigid_bodies.omega)
    l_world0 = np.einsum("bij,bj->bi", rot0, l_body0)
    l0 = float(np.linalg.norm(l_world0[0]))

    integrator = VelocityVerlet()
    dt = 1e-3
    for _ in range(5000):
        integrator.step(state, model, dt)

    rot1 = quat_to_rotmat(state.rigid_bodies.quat)
    l_body1 = np.einsum("bij,bj->bi", inertia, state.rigid_bodies.omega)
    l_world1 = np.einsum("bij,bj->bi", rot1, l_body1)
    l1 = float(np.linalg.norm(l_world1[0]))

    assert abs(l1 - l0) / l0 < 1e-3


def test_torque_free_spin_stable_quat_norm() -> None:
    inertia = box_inertia_body(2.0, np.array([1.0, 2.0, 3.0]))
    defn = {
        "schema_version": 1,
        "simulation": {"dt": 0.002, "steps": 1000, "integrator": "velocity_verlet"},
        "entities": {
            "rigid_bodies": {
                "pos": [[0.0, 0.0, 0.0]],
                "vel": [[0.0, 0.0, 0.0]],
                "quat": [[1.0, 0.0, 0.0, 0.0]],
                "omega_body": [[0.2, 0.3, 0.4]],
                "mass": [2.0],
                "mass_distribution": {
                    "points_body": [[0.0, 0.0, 0.0]],
                    "point_masses": [2.0],
                    "inertia_body": inertia.tolist(),
                },
                "source": [
                    {
                        "kind": "box",
                        "params": {"size": [1.0, 2.0, 3.0]},
                        "mass": 2.0,
                    }
                ],
            }
        },
        "models": [],
    }
    state, model, integrator, dt, steps, _ = scenario_to_runtime(defn)
    for _ in range(steps):
        integrator.step(state, model, dt)
    q_norm = float(np.linalg.norm(state.rigid_bodies.quat[0]))
    assert abs(q_norm - 1.0) < 1e-4
