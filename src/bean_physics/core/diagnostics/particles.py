"""Particle diagnostics."""

from __future__ import annotations

import numpy as np

from ..state.system import SystemState


def total_mass(state: SystemState) -> float:
    if state.particles is None or state.particles.mass.size == 0:
        return 0.0
    return float(np.sum(state.particles.mass))


def center_of_mass(state: SystemState) -> np.ndarray:
    if state.particles is None or state.particles.mass.size == 0:
        raise ValueError("cannot compute center of mass for empty particle set")
    m = state.particles.mass
    total = np.sum(m)
    if total == 0.0:
        raise ValueError("cannot compute center of mass with zero total mass")
    return np.sum(state.particles.pos * m[:, np.newaxis], axis=0) / total


def linear_momentum(state: SystemState) -> np.ndarray:
    if state.particles is None or state.particles.mass.size == 0:
        return np.zeros(3, dtype=np.float64)
    m = state.particles.mass
    return np.sum(state.particles.vel * m[:, np.newaxis], axis=0)


def kinetic_energy(state: SystemState) -> float:
    if state.particles is None or state.particles.mass.size == 0:
        return 0.0
    v2 = np.sum(state.particles.vel**2, axis=1)
    return float(0.5 * np.sum(state.particles.mass * v2))


def potential_energy_gravity(
    state: SystemState, G: float, softening: float = 0.0
) -> float:
    if state.particles is None:
        return 0.0
    pos = state.particles.pos
    mass = state.particles.mass
    n = pos.shape[0]
    if n < 2:
        return 0.0

    eps2 = softening * softening
    delta = pos[None, :, :] - pos[:, None, :]
    dist2 = np.sum(delta * delta, axis=-1) + eps2
    iu = np.triu_indices(n, k=1)
    dist = np.sqrt(dist2[iu])
    mprod = mass[iu[0]] * mass[iu[1]]
    return float(-G * np.sum(mprod / dist))


def total_energy_gravity(state: SystemState, G: float, softening: float = 0.0) -> float:
    return kinetic_energy(state) + potential_energy_gravity(state, G, softening)
