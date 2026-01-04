"""Impulse event scheduling for instantaneous velocity changes."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .state.system import SystemState


@dataclass(slots=True, frozen=True)
class ImpulseEvent:
    t: float
    target_type: str
    target_index: int
    target_id: str
    delta_v: np.ndarray
    label: str | None = None


@dataclass(slots=True)
class ImpulseSchedule:
    events: list[ImpulseEvent]
    next_index: int = 0

    def reset(self) -> None:
        self.next_index = 0

    def fire_for_window(
        self,
        state: SystemState,
        prev_t: float,
        new_t: float,
        include_start: bool = False,
    ) -> list[ImpulseEvent]:
        fired: list[ImpulseEvent] = []
        while self.next_index < len(self.events):
            event = self.events[self.next_index]
            if event.t < prev_t or (event.t == prev_t and not include_start):
                self.next_index += 1
                continue
            if event.t <= new_t and (event.t > prev_t or include_start):
                _apply_event(state, event)
                fired.append(event)
                self.next_index += 1
                continue
            break
        return fired


def _apply_event(state: SystemState, event: ImpulseEvent) -> None:
    if event.target_type == "particle":
        if state.particles is None:
            raise ValueError("impulse target particle missing from state")
        state.particles.vel[event.target_index] += event.delta_v
        return
    if event.target_type == "rigid_body":
        if state.rigid_bodies is None:
            raise ValueError("impulse target rigid body missing from state")
        state.rigid_bodies.vel[event.target_index] += event.delta_v
        return
    raise ValueError(f"unsupported impulse target type: {event.target_type}")
