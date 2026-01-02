"""Math utilities namespace."""

from .quat import (  # noqa: F401
    quat_conj,
    quat_from_axis_angle,
    quat_integrate_expmap,
    quat_mul,
    quat_normalize,
    quat_rotate,
    quat_to_rotmat,
)
from .vector import cross, norm, unit  # noqa: F401