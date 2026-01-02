"""Pure helpers for editing particle definitions."""

from __future__ import annotations

import math
from typing import Sequence

from ...io.scenario import ScenarioDefinition


ParticleRow = Sequence[object]


def particles_to_rows(defn: ScenarioDefinition) -> list[list[float]]:
    entities = defn.get("entities", {})
    particles = entities.get("particles")
    if not particles:
        return []
    pos = particles.get("pos", [])
    vel = particles.get("vel", [])
    mass = particles.get("mass", [])
    rows: list[list[float]] = []
    for i, m in enumerate(mass):
        px, py, pz = pos[i]
        vx, vy, vz = vel[i]
        rows.append([px, py, pz, vx, vy, vz, m])
    return rows


def rows_to_particles(defn: ScenarioDefinition, rows: Sequence[ParticleRow]) -> None:
    cleaned: list[list[float]] = []
    for r_index, row in enumerate(rows):
        if len(row) != 7:
            raise ValueError("each row must have 7 values")
        values = []
        for c_index, value in enumerate(row):
            try:
                fval = float(value)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"row {r_index + 1} has invalid number") from exc
            if not math.isfinite(fval):
                raise ValueError(f"row {r_index + 1} has non-finite value")
            if c_index == 6 and fval <= 0.0:
                raise ValueError("mass must be > 0")
            values.append(fval)
        cleaned.append(values)

    entities = defn.setdefault("entities", {})
    if not cleaned:
        entities.pop("particles", None)
        if not entities:
            defn.pop("entities", None)
        return

    pos = [[row[0], row[1], row[2]] for row in cleaned]
    vel = [[row[3], row[4], row[5]] for row in cleaned]
    mass = [row[6] for row in cleaned]
    entities["particles"] = {"pos": pos, "vel": vel, "mass": mass}
