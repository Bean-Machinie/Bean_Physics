"""Rigid body diagnostics."""

from __future__ import annotations

import numpy as np


def angular_momentum_body(I_body: np.ndarray, omega_body: np.ndarray) -> np.ndarray:
    """Return angular momentum in body frame."""
    I = np.asarray(I_body, dtype=np.float64)
    omega = np.asarray(omega_body, dtype=np.float64)
    if I.ndim == 2:
        return I @ omega
    return np.einsum("bij,bj->bi", I, omega)


def rotational_ke_body(I_body: np.ndarray, omega_body: np.ndarray) -> np.ndarray:
    """Return rotational kinetic energy in body frame."""
    L = angular_momentum_body(I_body, omega_body)
    omega = np.asarray(omega_body, dtype=np.float64)
    if L.ndim == 1:
        return 0.5 * float(np.dot(omega, L))
    return 0.5 * np.sum(omega * L, axis=1)
