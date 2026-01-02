"""Composite particle model utilities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from .base import Model, ParticleOnlyModel
from ..state.system import SystemState


@dataclass(slots=True)
class CompositeModel(ParticleOnlyModel):
    models: Sequence[Model]

    def acc_particles(self, state: SystemState) -> np.ndarray:
        if state.particles is None:
            return np.zeros((0, 3), dtype=np.float64)
        n = state.particles.pos.shape[0]
        if n == 0:
            return np.zeros((0, 3), dtype=np.float64)
        if not self.models:
            return np.zeros((n, 3), dtype=np.float64)

        acc = np.zeros((n, 3), dtype=np.float64)
        for model in self.models:
            acc += model.acc_particles(state)
        return acc

    def acc_rigid(self, state: SystemState) -> np.ndarray:
        if state.rigid_bodies is None:
            return np.zeros((0, 3), dtype=np.float64)
        m = state.rigid_bodies.pos.shape[0]
        if m == 0 or not self.models:
            return np.zeros((m, 3), dtype=np.float64)
        acc = np.zeros((m, 3), dtype=np.float64)
        for model in self.models:
            acc += model.acc_rigid(state)
        return acc

    def alpha_rigid(self, state: SystemState) -> np.ndarray:
        if state.rigid_bodies is None:
            return np.zeros((0, 3), dtype=np.float64)
        m = state.rigid_bodies.pos.shape[0]
        if m == 0 or not self.models:
            return np.zeros((m, 3), dtype=np.float64)
        alpha = np.zeros((m, 3), dtype=np.float64)
        for model in self.models:
            alpha += model.alpha_rigid(state)
        return alpha
