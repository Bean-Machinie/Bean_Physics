"""Pure helpers for viewport math."""

from __future__ import annotations

import numpy as np


def compute_bounds(points: np.ndarray) -> tuple[np.ndarray, float]:
    if points.size == 0:
        return np.zeros(3, dtype=np.float32), 0.0
    pts = np.asarray(points, dtype=np.float32)
    mins = np.min(pts, axis=0)
    maxs = np.max(pts, axis=0)
    center = (mins + maxs) * 0.5
    radius = float(np.max(np.linalg.norm(pts - center, axis=1)))
    return center, radius


def compute_selection_bounds(
    points: np.ndarray, indices: list[int]
) -> tuple[np.ndarray, float]:
    if points.size == 0 or not indices:
        return np.zeros(3, dtype=np.float32), 0.0
    valid = [i for i in indices if 0 <= i < points.shape[0]]
    if not valid:
        return np.zeros(3, dtype=np.float32), 0.0
    subset = points[valid]
    return compute_bounds(subset)
