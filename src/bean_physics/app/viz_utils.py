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


def rigid_transform_matrix(rot: np.ndarray, pos: np.ndarray) -> np.ndarray:
    r = np.asarray(rot, dtype=np.float32)
    t = np.asarray(pos, dtype=np.float32)
    if r.shape != (3, 3):
        raise ValueError("rot must have shape (3, 3)")
    if t.shape != (3,):
        raise ValueError("pos must have shape (3,)")
    mat = np.eye(4, dtype=np.float32)
    mat[:3, :3] = r
    mat[3, :3] = t
    return mat


def compose_visual_transform(
    pos: np.ndarray,
    rot_world: np.ndarray,
    offset_body: np.ndarray,
    rot_local: np.ndarray,
    scale: np.ndarray,
) -> np.ndarray:
    pos = np.asarray(pos, dtype=np.float32)
    rot_world = np.asarray(rot_world, dtype=np.float32)
    offset_body = np.asarray(offset_body, dtype=np.float32)
    rot_local = np.asarray(rot_local, dtype=np.float32)
    scale = np.asarray(scale, dtype=np.float32)
    if pos.shape != (3,) or offset_body.shape != (3,) or scale.shape != (3,):
        raise ValueError("pos, offset_body, and scale must have shape (3,)")
    if rot_world.shape != (3, 3) or rot_local.shape != (3, 3):
        raise ValueError("rotations must have shape (3, 3)")

    t_world = np.eye(4, dtype=np.float32)
    t_world[3, :3] = pos
    r_world = np.eye(4, dtype=np.float32)
    r_world[:3, :3] = rot_world
    t_offset = np.eye(4, dtype=np.float32)
    t_offset[3, :3] = offset_body
    r_local = np.eye(4, dtype=np.float32)
    r_local[:3, :3] = rot_local
    s_local = np.eye(4, dtype=np.float32)
    s_local[0, 0] = scale[0]
    s_local[1, 1] = scale[1]
    s_local[2, 2] = scale[2]
    return s_local @ r_local @ t_offset @ r_world @ t_world
