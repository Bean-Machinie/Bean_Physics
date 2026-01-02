from __future__ import annotations

import numpy as np
import pytest

from bean_physics.core.diagnostics import (
    center_of_mass,
    kinetic_energy,
    linear_momentum,
    potential_energy_gravity,
    total_energy_gravity,
    total_mass,
)
from bean_physics.core.state import ParticlesState, SystemState


def test_diagnostics_values() -> None:
    pos = np.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]], dtype=np.float64)
    vel = np.array([[1.0, 0.0, 0.0], [0.0, 2.0, 0.0]], dtype=np.float64)
    mass = np.array([1.0, 3.0], dtype=np.float64)
    state = SystemState(particles=ParticlesState(pos=pos, vel=vel, mass=mass))

    assert total_mass(state) == 4.0
    assert np.allclose(center_of_mass(state), np.array([1.5, 0.0, 0.0]))
    assert np.allclose(linear_momentum(state), np.array([1.0, 6.0, 0.0]))
    assert kinetic_energy(state) == 6.5


def test_diagnostics_empty_set_behavior() -> None:
    pos = np.zeros((0, 3), dtype=np.float64)
    vel = np.zeros((0, 3), dtype=np.float64)
    mass = np.zeros(0, dtype=np.float64)
    state = SystemState(particles=ParticlesState(pos=pos, vel=vel, mass=mass))

    assert total_mass(state) == 0.0
    assert np.allclose(linear_momentum(state), np.zeros(3, dtype=np.float64))
    assert kinetic_energy(state) == 0.0
    with pytest.raises(ValueError, match="empty particle set"):
        _ = center_of_mass(state)


def test_gravity_potential_energy() -> None:
    pos = np.array([[0.0, 0.0, 0.0], [4.0, 0.0, 0.0]], dtype=np.float64)
    vel = np.zeros((2, 3), dtype=np.float64)
    mass = np.array([2.0, 3.0], dtype=np.float64)
    state = SystemState(particles=ParticlesState(pos=pos, vel=vel, mass=mass))

    pe = potential_energy_gravity(state, G=1.0)
    assert pe == -1.5
    assert total_energy_gravity(state, G=1.0) == pe
