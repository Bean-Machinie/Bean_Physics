from __future__ import annotations

import numpy as np

from bean_physics.app.viz_utils import rigid_transform_matrix


def test_rigid_transform_matrix_translation_only() -> None:
    rot = np.eye(3, dtype=np.float32)
    pos = np.array([5.0, -2.0, 1.5], dtype=np.float32)
    mat = rigid_transform_matrix(rot, pos)
    assert np.allclose(mat[:3, :3], rot)
    assert np.allclose(mat[3, :3], pos)
