"""Quaternion utilities.

Conventions:
- Storage order: [w, x, y, z]
- Quaternion represents body->world rotation.
- Vector rotation: v_world = R(q) * v_body
- Composition: applying q1 then q2 is q = quat_mul(q2, q1)
- Angular velocity is expressed in WORLD frame.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from .vector import norm, unit


ArrayF = NDArray[np.float64]


def quat_normalize(q: ArrayF) -> ArrayF:
    """Normalize quaternion(s) to unit length."""
    q = np.asarray(q, dtype=np.float64)
    n = norm(q, axis=-1)[..., np.newaxis]
    with np.errstate(divide="ignore", invalid="ignore"):
        qn = np.where(n > 0.0, q / n, q)
    return qn


def quat_conj(q: ArrayF) -> ArrayF:
    """Return the quaternion conjugate."""
    q = np.asarray(q, dtype=np.float64)
    w = q[..., :1]
    xyz = -q[..., 1:]
    return np.concatenate([w, xyz], axis=-1)


def quat_mul(q1: ArrayF, q2: ArrayF) -> ArrayF:
    """Hamilton product of two quaternions (supports broadcasting)."""
    q1 = np.asarray(q1, dtype=np.float64)
    q2 = np.asarray(q2, dtype=np.float64)
    w1, x1, y1, z1 = np.split(q1, 4, axis=-1)
    w2, x2, y2, z2 = np.split(q2, 4, axis=-1)
    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
    return np.concatenate([w, x, y, z], axis=-1)


def quat_to_rotmat(q: ArrayF) -> ArrayF:
    """Convert quaternion(s) to rotation matrix/matrices."""
    q = quat_normalize(q)
    w, x, y, z = np.split(q, 4, axis=-1)

    ww = w * w
    xx = x * x
    yy = y * y
    zz = z * z

    wx = w * x
    wy = w * y
    wz = w * z
    xy = x * y
    xz = x * z
    yz = y * z

    m00 = ww + xx - yy - zz
    m01 = 2.0 * (xy - wz)
    m02 = 2.0 * (xz + wy)

    m10 = 2.0 * (xy + wz)
    m11 = ww - xx + yy - zz
    m12 = 2.0 * (yz - wx)

    m20 = 2.0 * (xz - wy)
    m21 = 2.0 * (yz + wx)
    m22 = ww - xx - yy + zz

    row0 = np.concatenate([m00, m01, m02], axis=-1)
    row1 = np.concatenate([m10, m11, m12], axis=-1)
    row2 = np.concatenate([m20, m21, m22], axis=-1)

    return np.stack([row0, row1, row2], axis=-2)


def quat_rotate(q: ArrayF, v: ArrayF) -> ArrayF:
    """Rotate vector(s) using quaternion(s) with body->world convention."""
    q = quat_normalize(q)
    v = np.asarray(v, dtype=np.float64)
    zeros = np.zeros(v[..., :1].shape, dtype=np.float64)
    vq = np.concatenate([zeros, v], axis=-1)
    return quat_mul(quat_mul(q, vq), quat_conj(q))[..., 1:]


def quat_from_axis_angle(axis: ArrayF, angle_rad: ArrayF) -> ArrayF:
    """Create quaternion(s) from axis-angle."""
    axis = np.asarray(axis, dtype=np.float64)
    angle_rad = np.asarray(angle_rad, dtype=np.float64)
    axis_unit = unit(axis, axis=-1)
    half = 0.5 * angle_rad
    sin_half = np.sin(half)[..., np.newaxis]
    w = np.cos(half)[..., np.newaxis]
    xyz = axis_unit * sin_half
    return np.concatenate([w, xyz], axis=-1)


def quat_integrate_expmap(q: ArrayF, omega_world: ArrayF, dt: float) -> ArrayF:
    """Integrate orientation using exponential map.

    For body->world quaternions with world-frame angular velocity, the update is:
        q_next = quat_mul(delta_q, q)
    """
    q = np.asarray(q, dtype=np.float64)
    omega_world = np.asarray(omega_world, dtype=np.float64)
    angle = norm(omega_world, axis=-1) * dt
    delta_q = quat_from_axis_angle(omega_world, angle)
    q_next = quat_mul(delta_q, q)
    return quat_normalize(q_next)