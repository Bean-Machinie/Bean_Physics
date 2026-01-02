"""Particle state containers."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


ArrayF = NDArray[np.float64]


@dataclass(slots=True)
class ParticlesState:
    pos: ArrayF
    vel: ArrayF
    mass: ArrayF

    def __post_init__(self) -> None:
        self.pos = np.ascontiguousarray(self.pos, dtype=np.float64)
        self.vel = np.ascontiguousarray(self.vel, dtype=np.float64)
        self.mass = np.ascontiguousarray(self.mass, dtype=np.float64)
        self.validate()

    def validate(self) -> None:
        if self.pos.ndim != 2 or self.pos.shape[1] != 3:
            raise ValueError("pos must have shape (N, 3)")
        if self.vel.shape != self.pos.shape:
            raise ValueError("vel must have shape (N, 3)")
        if self.mass.ndim != 1 or self.mass.shape[0] != self.pos.shape[0]:
            raise ValueError("mass must have shape (N,)")

    def copy(self) -> "ParticlesState":
        return ParticlesState(
            pos=self.pos.copy(),
            vel=self.vel.copy(),
            mass=self.mass.copy(),
        )