"""Vector utilities for NumPy arrays.

All vectors are expected to be shaped (..., 3).
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


ArrayF = NDArray[np.float64]


def norm(v: ArrayF, axis: int = -1) -> ArrayF:
    """Return the L2 norm along an axis."""
    return np.linalg.norm(v, axis=axis)


def unit(v: ArrayF, axis: int = -1) -> ArrayF:
    """Return unit vectors with safe handling of zero vectors."""
    v = np.asarray(v, dtype=np.float64)
    n = np.linalg.norm(v, axis=axis, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        u = np.where(n > 0.0, v / n, 0.0)
    return u


def cross(a: ArrayF, b: ArrayF) -> ArrayF:
    """Return the cross product of two vectors."""
    return np.cross(a, b)