"""Composite particle model utilities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from .base import ParticleModel, ParticleOnlyModel
from ..state.system import SystemState


@dataclass(slots=True)
class CompositeModel(ParticleOnlyModel):
    models: Sequence[ParticleModel]

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