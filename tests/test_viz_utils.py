from __future__ import annotations

import numpy as np

from bean_physics.app.viz_utils import compose_visual_transform, rigid_transform_matrix


def test_rigid_transform_matrix_translation_only() -> None:
    rot = np.eye(3, dtype=np.float32)
    pos = np.array([5.0, -2.0, 1.5], dtype=np.float32)
    mat = rigid_transform_matrix(rot, pos)
    assert np.allclose(mat[:3, :3], rot)
    assert np.allclose(mat[3, :3], pos)


def test_compose_visual_transform_translation() -> None:
    pos = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    rot_world = np.eye(3, dtype=np.float32)
    offset = np.array([0.5, 0.0, 0.0], dtype=np.float32)
    rot_local = np.eye(3, dtype=np.float32)
    scale = np.array([2.0, 2.0, 2.0], dtype=np.float32)
    mat = compose_visual_transform(pos, rot_world, offset, rot_local, scale)
    assert np.allclose(mat[3, :3], pos + offset)
