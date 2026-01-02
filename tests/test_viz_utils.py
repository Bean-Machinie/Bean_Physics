from __future__ import annotations

import numpy as np

from bean_physics.app.viz_utils import compute_bounds, compute_selection_bounds


def test_compute_bounds_empty() -> None:
    center, radius = compute_bounds(np.zeros((0, 3), dtype=np.float32))
    assert np.allclose(center, np.zeros(3))
    assert radius == 0.0


def test_compute_bounds_single() -> None:
    points = np.array([[1.0, 2.0, 3.0]], dtype=np.float32)
    center, radius = compute_bounds(points)
    assert np.allclose(center, [1.0, 2.0, 3.0])
    assert radius == 0.0


def test_compute_bounds_many() -> None:
    points = np.array(
        [
            [-1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 2.0, 0.0],
        ],
        dtype=np.float32,
    )
    center, radius = compute_bounds(points)
    assert np.allclose(center, [0.0, 1.0, 0.0])
    assert radius > 0.0


def test_selection_bounds() -> None:
    points = np.array(
        [
            [-1.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            [0.0, 2.0, 0.0],
        ],
        dtype=np.float32,
    )
    center, radius = compute_selection_bounds(points, [1])
    assert np.allclose(center, [2.0, 0.0, 0.0])
    assert radius == 0.0
