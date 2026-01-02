"""Rigid body mass properties from discrete point masses."""

from __future__ import annotations

import numpy as np


def mass_properties(points_body: np.ndarray, masses: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
    """Return total mass, center of mass, and inertia tensor about CoM.

    Parameters
    ----------
    points_body : (K, 3)
        Point positions in body coordinates.
    masses : (K,)
        Point masses.
    """
    points = np.asarray(points_body, dtype=np.float64)
    masses = np.asarray(masses, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("points_body must have shape (K, 3)")
    if masses.ndim != 1 or masses.shape[0] != points.shape[0]:
        raise ValueError("masses must have shape (K,)")

    total_mass = float(np.sum(masses))
    if total_mass <= 0.0:
        raise ValueError("total mass must be positive")

    com = np.sum(points * masses[:, np.newaxis], axis=0) / total_mass
    r = points - com
    r2 = np.sum(r * r, axis=1)
    eye = np.eye(3, dtype=np.float64)
    inertia = np.sum(
        masses[:, np.newaxis, np.newaxis] * (r2[:, None, None] * eye - r[:, :, None] * r[:, None, :]),
        axis=0,
    )
    return total_mass, com, inertia
