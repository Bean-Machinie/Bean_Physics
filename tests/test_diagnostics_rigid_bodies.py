from __future__ import annotations

import numpy as np

from bean_physics.core.diagnostics import angular_momentum_body, rotational_ke_body


def test_angular_momentum_body_single() -> None:
    I = np.diag([2.0, 3.0, 4.0])
    omega = np.array([0.5, -1.0, 2.0], dtype=np.float64)
    L = angular_momentum_body(I, omega)
    assert np.allclose(L, np.array([1.0, -3.0, 8.0]))


def test_rotational_ke_body_single() -> None:
    I = np.diag([2.0, 3.0, 4.0])
    omega = np.array([0.5, -1.0, 2.0], dtype=np.float64)
    ke = rotational_ke_body(I, omega)
    assert ke == 0.5 * (0.5 * 1.0 + (-1.0) * -3.0 + 2.0 * 8.0)

