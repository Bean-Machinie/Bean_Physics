"""Uniform gravity model for particles."""

from __future__ import annotations

import numpy as np

from .base import ParticleOnlyModel
from ..state.system import SystemState


class UniformGravity(ParticleOnlyModel):
    def __init__(self, g: np.ndarray) -> None:
        self.g = np.asarray(g, dtype=np.float64)
        if self.g.shape != (3,):
            raise ValueError("g must have shape (3,)")

    def acc_particles(self, state: SystemState) -> np.ndarray:
        if state.particles is None:
            return np.zeros((0, 3), dtype=np.float64)
        n = state.particles.pos.shape[0]
        if n == 0:
            return np.zeros((0, 3), dtype=np.float64)
        return np.broadcast_to(self.g, (n, 3)).copy()

    def acc_rigid(self, state: SystemState) -> np.ndarray:
        if state.rigid_bodies is None:
            return np.zeros((0, 3), dtype=np.float64)
        m = state.rigid_bodies.pos.shape[0]
        if m == 0:
            return np.zeros((0, 3), dtype=np.float64)
        return np.broadcast_to(self.g, (m, 3)).copy()
