"""Pure helpers for object list and particle edits."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

from ...io.scenario import ScenarioDefinition


@dataclass(frozen=True, slots=True)
class ObjectRef:
    type: str
    index: int


def list_particles(defn: ScenarioDefinition) -> list[ObjectRef]:
    entities = defn.get("entities", {})
    particles = entities.get("particles")
    if not particles:
        return []
    count = len(particles.get("mass", []))
    return [ObjectRef(type="particle", index=i) for i in range(count)]


def particle_summary(defn: ScenarioDefinition, index: int) -> dict[str, float]:
    particles = defn.get("entities", {}).get("particles")
    if not particles:
        raise ValueError("no particles in scenario")
    pos = particles["pos"][index]
    vel = particles["vel"][index]
    mass = particles["mass"][index]
    return {
        "x": float(pos[0]),
        "y": float(pos[1]),
        "z": float(pos[2]),
        "vx": float(vel[0]),
        "vy": float(vel[1]),
        "vz": float(vel[2]),
        "mass": float(mass),
    }


def add_particle(defn: ScenarioDefinition) -> int:
    entities = defn.setdefault("entities", {})
    particles = entities.setdefault("particles", {"pos": [], "vel": [], "mass": []})
    particles["pos"].append([0.0, 0.0, 0.0])
    particles["vel"].append([0.0, 0.0, 0.0])
    particles["mass"].append(1.0)
    return len(particles["mass"]) - 1


def remove_particle(defn: ScenarioDefinition, index: int) -> None:
    particles = defn.get("entities", {}).get("particles")
    if not particles:
        raise ValueError("no particles to remove")
    for key in ("pos", "vel", "mass"):
        items = particles[key]
        if index < 0 or index >= len(items):
            raise IndexError("particle index out of range")
        del items[index]
    if not particles["mass"]:
        entities = defn.get("entities", {})
        entities.pop("particles", None)
        if not entities:
            defn.pop("entities", None)


def apply_particle_edit(
    defn: ScenarioDefinition, index: int, values: Sequence[object]
) -> None:
    if len(values) != 7:
        raise ValueError("expected 7 values")
    cleaned = []
    for i, value in enumerate(values):
        try:
            fval = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError("invalid number") from exc
        if not math.isfinite(fval):
            raise ValueError("non-finite value")
        if i == 6 and fval <= 0.0:
            raise ValueError("mass must be > 0")
        cleaned.append(fval)
    particles = defn.get("entities", {}).get("particles")
    if not particles:
        raise ValueError("no particles in scenario")
    if index < 0 or index >= len(particles["mass"]):
        raise IndexError("particle index out of range")
    particles["pos"][index] = cleaned[0:3]
    particles["vel"][index] = cleaned[3:6]
    particles["mass"][index] = cleaned[6]
