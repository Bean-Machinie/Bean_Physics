"""Newtonian N-body gravity for particles."""

from __future__ import annotations

import numpy as np

from .base import ParticleOnlyModel
from ..state.system import SystemState


class NBodyGravity(ParticleOnlyModel):
    def __init__(
        self,
        G: float = 6.67430e-11,
        softening: float = 0.0,
        chunk_size: int | None = None,
    ) -> None:
        self.G = float(G)
        self.softening = float(softening)
        self.chunk_size = chunk_size

    def acc_particles(self, state: SystemState) -> np.ndarray:
        if state.particles is None:
            return np.zeros((0, 3), dtype=np.float64)
        pos = state.particles.pos
        mass = state.particles.mass
        n = pos.shape[0]
        if n == 0:
            return np.zeros((0, 3), dtype=np.float64)

        eps2 = self.softening * self.softening
        if self.chunk_size is None or self.chunk_size >= n:
            delta = pos[None, :, :] - pos[:, None, :]
            dist2 = np.sum(delta * delta, axis=-1) + eps2
            np.fill_diagonal(dist2, np.inf)
            inv_dist3 = dist2 ** -1.5
            acc = np.sum(
                delta * inv_dist3[..., np.newaxis] * mass[None, :, None],
                axis=1,
            )
            return self.G * acc

        acc = np.zeros((n, 3), dtype=np.float64)
        for i0 in range(0, n, self.chunk_size):
            i1 = min(i0 + self.chunk_size, n)
            block = pos[i0:i1]
            delta = pos[None, :, :] - block[:, None, :]
            dist2 = np.sum(delta * delta, axis=-1) + eps2
            diag_idx = np.arange(i0, i1)
            dist2[np.arange(i1 - i0), diag_idx] = np.inf
            inv_dist3 = dist2 ** -1.5
            acc[i0:i1] = np.sum(
                delta * inv_dist3[..., np.newaxis] * mass[None, :, None],
                axis=1,
            )
        return self.G * acc
