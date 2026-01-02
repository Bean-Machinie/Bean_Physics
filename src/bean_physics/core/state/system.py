"""System state container for all simulation entities."""

from __future__ import annotations

from dataclasses import dataclass

from .particles import ParticlesState
from .rigid_bodies import RigidBodiesState


@dataclass(slots=True)
class SystemState:
    particles: ParticlesState | None = None
    rigid_bodies: RigidBodiesState | None = None

    def validate(self) -> None:
        if self.particles is not None:
            self.particles.validate()
        if self.rigid_bodies is not None:
            self.rigid_bodies.validate()

    def clone(self) -> "SystemState":
        return SystemState(
            particles=self.particles.copy() if self.particles is not None else None,
            rigid_bodies=(
                self.rigid_bodies.copy() if self.rigid_bodies is not None else None
            ),
        )