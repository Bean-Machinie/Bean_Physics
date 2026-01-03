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


def box_inertia_body(mass: float, size: np.ndarray) -> np.ndarray:
    """Return inertia tensor for a box about CoM in body frame."""
    mass_val = float(mass)
    if mass_val <= 0.0:
        raise ValueError("mass must be > 0")
    dims = np.asarray(size, dtype=np.float64)
    if dims.shape != (3,):
        raise ValueError("size must have shape (3,)")
    if np.any(dims <= 0.0):
        raise ValueError("size values must be > 0")
    sx, sy, sz = dims
    ixx = (mass_val / 12.0) * (sy * sy + sz * sz)
    iyy = (mass_val / 12.0) * (sx * sx + sz * sz)
    izz = (mass_val / 12.0) * (sx * sx + sy * sy)
    return np.diag([ixx, iyy, izz])


def sphere_inertia_body(mass: float, radius: float) -> np.ndarray:
    """Return inertia tensor for a solid sphere about CoM in body frame."""
    mass_val = float(mass)
    if mass_val <= 0.0:
        raise ValueError("mass must be > 0")
    radius_val = float(radius)
    if radius_val <= 0.0:
        raise ValueError("radius must be > 0")
    i = (2.0 / 5.0) * mass_val * radius_val * radius_val
    return np.diag([i, i, i])
