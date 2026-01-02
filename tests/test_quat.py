from __future__ import annotations

import numpy as np

from bean_physics.core.math.quat import (
    quat_from_axis_angle,
    quat_integrate_expmap,
    quat_mul,
    quat_rotate,
)


def test_quat_rotate_preserves_norm() -> None:
    rng = np.random.default_rng(123)
    v = rng.normal(size=(100, 3))
    axis = rng.normal(size=(100, 3))
    angle = rng.uniform(low=-np.pi, high=np.pi, size=(100,))
    q = quat_from_axis_angle(axis, angle)

    v_rot = quat_rotate(q, v)
    n0 = np.linalg.norm(v, axis=-1)
    n1 = np.linalg.norm(v_rot, axis=-1)
    assert np.allclose(n0, n1, rtol=1e-12, atol=1e-12)


def test_quat_composition() -> None:
    rng = np.random.default_rng(456)
    v = rng.normal(size=(10, 3))
    q1 = quat_from_axis_angle(rng.normal(size=(10, 3)), rng.normal(size=(10,)))
    q2 = quat_from_axis_angle(rng.normal(size=(10, 3)), rng.normal(size=(10,)))

    v_seq = quat_rotate(q2, quat_rotate(q1, v))
    q_comp = quat_mul(q2, q1)
    v_comp = quat_rotate(q_comp, v)

    assert np.allclose(v_seq, v_comp, rtol=1e-12, atol=1e-12)


def test_quat_integrate_expmap_norm_stays_1() -> None:
    q = np.array([[1.0, 0.0, 0.0, 0.0]], dtype=np.float64)
    omega_body = np.array([[0.1, -0.2, 0.3]], dtype=np.float64)
    dt = 0.01

    for _ in range(1000):
        q = quat_integrate_expmap(q, omega_body, dt)

    n = np.linalg.norm(q, axis=-1)
    assert np.allclose(n, 1.0, atol=1e-10)
