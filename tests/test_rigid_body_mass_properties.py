from __future__ import annotations

import numpy as np

from bean_physics.core.rigid_body.mass_properties import mass_properties


def test_mass_properties_com_and_inertia() -> None:
    points = np.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]], dtype=np.float64)
    masses = np.array([1.0, 3.0], dtype=np.float64)

    total_mass, com, inertia = mass_properties(points, masses)
    assert total_mass == 4.0
    assert np.allclose(com, np.array([1.5, 0.0, 0.0]))
    expected_inertia = np.diag([0.0, 3.0, 3.0])
    assert np.allclose(inertia, expected_inertia)


def test_mass_properties_dumbbell() -> None:
    a = 2.0
    m = 1.5
    points = np.array([[-a, 0.0, 0.0], [a, 0.0, 0.0]], dtype=np.float64)
    masses = np.array([m, m], dtype=np.float64)

    total_mass, com, inertia = mass_properties(points, masses)
    assert total_mass == 2.0 * m
    assert np.allclose(com, np.zeros(3))
    expected = np.diag([0.0, 2.0 * m * a * a, 2.0 * m * a * a])
    assert np.allclose(inertia, expected)
