"""Force/model interfaces."""

from __future__ import annotations

from typing import Protocol

import numpy as np
from numpy.typing import NDArray

from ..state.system import SystemState


ArrayF = NDArray[np.float64]


class Model(Protocol):
    def acc_particles(self, state: SystemState) -> ArrayF:
        """Return particle accelerations as (N, 3)."""

    def acc_rigid(self, state: SystemState) -> ArrayF:
        """Return rigid body CoM accelerations as (M, 3)."""

    def alpha_rigid(self, state: SystemState) -> ArrayF:
        """Return rigid body angular accelerations (body frame) as (M, 3)."""


class ParticleModel(Protocol):
    def acc_particles(self, state: SystemState) -> ArrayF:
        """Return particle accelerations as (N, 3)."""


class ParticleOnlyModel:
    """Base class for particle-only models with rigid-body defaults."""

    def acc_rigid(self, state: SystemState) -> ArrayF:
        if state.rigid_bodies is None:
            return np.zeros((0, 3), dtype=np.float64)
        m = state.rigid_bodies.pos.shape[0]
        return np.zeros((m, 3), dtype=np.float64)

    def alpha_rigid(self, state: SystemState) -> ArrayF:
        if state.rigid_bodies is None:
            return np.zeros((0, 3), dtype=np.float64)
        m = state.rigid_bodies.pos.shape[0]
        return np.zeros((m, 3), dtype=np.float64)
