"""Rigid body state containers (kinematics only)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from ..math.quat import quat_normalize


ArrayF = NDArray[np.float64]


@dataclass(slots=True)
class RigidBodiesState:
    """Rigid body kinematic state.

    omega is stored in the BODY frame (omega_body).
    """
    pos: ArrayF
    vel: ArrayF
    quat: ArrayF
    omega: ArrayF
    mass: ArrayF
    inertia_body: ArrayF

    def __post_init__(self) -> None:
        self.pos = np.ascontiguousarray(self.pos, dtype=np.float64)
        self.vel = np.ascontiguousarray(self.vel, dtype=np.float64)
        self.quat = np.ascontiguousarray(self.quat, dtype=np.float64)
        self.omega = np.ascontiguousarray(self.omega, dtype=np.float64)
        self.mass = np.ascontiguousarray(self.mass, dtype=np.float64)
        self.inertia_body = np.ascontiguousarray(self.inertia_body, dtype=np.float64)
        self.validate(strict_quat=False)

    def validate(self, strict_quat: bool = False) -> None:
        if self.pos.ndim != 2 or self.pos.shape[1] != 3:
            raise ValueError("pos must have shape (M, 3)")
        if self.vel.shape != self.pos.shape:
            raise ValueError("vel must have shape (M, 3)")
        if self.omega.shape != self.pos.shape:
            raise ValueError("omega must have shape (M, 3)")
        if self.quat.ndim != 2 or self.quat.shape[1] != 4:
            raise ValueError("quat must have shape (M, 4)")
        if self.quat.shape[0] != self.pos.shape[0]:
            raise ValueError("quat must match number of bodies")
        if self.mass.ndim != 1 or self.mass.shape[0] != self.pos.shape[0]:
            raise ValueError("mass must have shape (M,)")
        if self.inertia_body.ndim == 2:
            self.inertia_body = np.broadcast_to(
                self.inertia_body, (self.mass.shape[0], 3, 3)
            ).copy()
        if self.inertia_body.shape != (self.mass.shape[0], 3, 3):
            raise ValueError("inertia_body must have shape (M, 3, 3)")

        if strict_quat:
            norms = np.linalg.norm(self.quat, axis=-1)
            if not np.allclose(norms, 1.0, atol=1e-6):
                raise ValueError("quat must be unit length")
        else:
            self.quat = quat_normalize(self.quat)

    def copy(self) -> "RigidBodiesState":
        return RigidBodiesState(
            pos=self.pos.copy(),
            vel=self.vel.copy(),
            quat=self.quat.copy(),
            omega=self.omega.copy(),
            mass=self.mass.copy(),
            inertia_body=self.inertia_body.copy(),
        )
