from __future__ import annotations

import numpy as np

from bean_physics.core.rigid_body.mass_properties import (
    box_inertia_body,
    mass_properties,
    rigid_body_from_points,
    shift_points_to_com,
    sphere_inertia_body,
)


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


def test_box_inertia_body() -> None:
    inertia = box_inertia_body(2.0, np.array([2.0, 4.0, 6.0]))
    expected = np.diag(
        [
            2.0 * (4.0**2 + 6.0**2) / 12.0,
            2.0 * (2.0**2 + 6.0**2) / 12.0,
            2.0 * (2.0**2 + 4.0**2) / 12.0,
        ]
    )
    assert np.allclose(inertia, expected)


def test_sphere_inertia_body() -> None:
    inertia = sphere_inertia_body(3.0, 2.0)
    expected = np.diag([2.0 / 5.0 * 3.0 * 4.0] * 3)
    assert np.allclose(inertia, expected)


def test_rigid_body_from_points_dumbbell() -> None:
    a = 1.5
    m = 2.0
    points = np.array([[-a, 0.0, 0.0], [a, 0.0, 0.0]], dtype=np.float64)
    masses = np.array([m, m], dtype=np.float64)
    total_mass, com, inertia = rigid_body_from_points(points, masses)
    assert total_mass == 2.0 * m
    assert np.allclose(com, np.zeros(3))
    expected = np.diag([0.0, 2.0 * m * a * a, 2.0 * m * a * a])
    assert np.allclose(inertia, expected)


def test_shift_points_to_com_keeps_inertia() -> None:
    points = np.array([[0.0, 0.0, 0.0], [3.0, 0.0, 0.0]], dtype=np.float64)
    masses = np.array([1.0, 2.0], dtype=np.float64)
    _, com, inertia = rigid_body_from_points(points, masses)
    shifted, com_old = shift_points_to_com(points, masses)
    _, com_shifted, inertia_shifted = rigid_body_from_points(shifted, masses)
    assert np.allclose(com_old, com)
    assert np.allclose(com_shifted, np.zeros(3))
    assert np.allclose(inertia_shifted, inertia)
