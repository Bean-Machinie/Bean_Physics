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
        pos_all, mass_all, n_particles = _collect_nbody_masses(state)
        if n_particles == 0:
            return np.zeros((0, 3), dtype=np.float64)
        acc_all = _nbody_accel(
            pos_all, mass_all, self.G, self.softening, self.chunk_size
        )
        return acc_all[:n_particles]

    def acc_rigid(self, state: SystemState) -> np.ndarray:
        pos_all, mass_all, n_particles = _collect_nbody_masses(state)
        if state.rigid_bodies is None:
            return np.zeros((0, 3), dtype=np.float64)
        n_rigid = state.rigid_bodies.pos.shape[0]
        if n_rigid == 0:
            return np.zeros((0, 3), dtype=np.float64)
        acc_all = _nbody_accel(
            pos_all, mass_all, self.G, self.softening, self.chunk_size
        )
        return acc_all[n_particles:]


def _collect_nbody_masses(
    state: SystemState,
) -> tuple[np.ndarray, np.ndarray, int]:
    pos_parts = (
        state.particles.pos if state.particles is not None else np.zeros((0, 3))
    )
    mass_parts = (
        state.particles.mass if state.particles is not None else np.zeros(0)
    )
    pos_rigid = (
        state.rigid_bodies.pos if state.rigid_bodies is not None else np.zeros((0, 3))
    )
    mass_rigid = (
        state.rigid_bodies.mass if state.rigid_bodies is not None else np.zeros(0)
    )
    pos_all = np.concatenate([pos_parts, pos_rigid], axis=0).astype(np.float64, copy=False)
    mass_all = np.concatenate([mass_parts, mass_rigid], axis=0).astype(np.float64, copy=False)
    return pos_all, mass_all, pos_parts.shape[0]


def _nbody_accel(
    pos: np.ndarray,
    mass: np.ndarray,
    G: float,
    softening: float,
    chunk_size: int | None,
) -> np.ndarray:
    n = pos.shape[0]
    if n == 0:
        return np.zeros((0, 3), dtype=np.float64)
    eps2 = softening * softening
    if chunk_size is None or chunk_size >= n:
        delta = pos[None, :, :] - pos[:, None, :]
        dist2 = np.sum(delta * delta, axis=-1) + eps2
        np.fill_diagonal(dist2, np.inf)
        inv_dist3 = dist2 ** -1.5
        acc = np.sum(
            delta * inv_dist3[..., np.newaxis] * mass[None, :, None],
            axis=1,
        )
        return G * acc

    acc = np.zeros((n, 3), dtype=np.float64)
    for i0 in range(0, n, chunk_size):
        i1 = min(i0 + chunk_size, n)
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
    return G * acc
