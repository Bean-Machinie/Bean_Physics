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
    subtype: str | None = None


def list_particles(defn: ScenarioDefinition) -> list[ObjectRef]:
    entities = defn.get("entities", {})
    particles = entities.get("particles")
    if not particles:
        return []
    count = len(particles.get("mass", []))
    return [ObjectRef(type="particle", index=i) for i in range(count)]


def list_forces(defn: ScenarioDefinition) -> list[ObjectRef]:
    models = defn.get("models", [])
    refs: list[ObjectRef] = []
    for idx, entry in enumerate(models):
        if not isinstance(entry, dict) or len(entry) != 1:
            continue
        key = next(iter(entry))
        if key in {"uniform_gravity", "nbody_gravity"}:
            refs.append(ObjectRef(type="force", index=idx, subtype=key))
    return refs


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


def force_summary(defn: ScenarioDefinition, model_index: int) -> dict[str, object]:
    models = defn.get("models", [])
    if model_index < 0 or model_index >= len(models):
        raise IndexError("force index out of range")
    entry = models[model_index]
    if "uniform_gravity" in entry:
        g = entry["uniform_gravity"]["g"]
        return {"type": "uniform_gravity", "g": [float(v) for v in g]}
    if "nbody_gravity" in entry:
        cfg = entry["nbody_gravity"]
        return {
            "type": "nbody_gravity",
            "G": float(cfg["G"]),
            "softening": float(cfg.get("softening", 0.0)),
            "chunk_size": cfg.get("chunk_size"),
        }
    raise ValueError("unsupported force type")


def add_particle(defn: ScenarioDefinition) -> int:
    entities = defn.setdefault("entities", {})
    particles = entities.setdefault("particles", {"pos": [], "vel": [], "mass": []})
    particles["pos"].append([0.0, 0.0, 0.0])
    particles["vel"].append([0.0, 0.0, 0.0])
    particles["mass"].append(1.0)
    return len(particles["mass"]) - 1


def add_uniform_gravity(defn: ScenarioDefinition, g: Sequence[object]) -> int:
    g_vals = _validate_vec3(g, "g")
    models = defn.setdefault("models", [])
    models.append({"uniform_gravity": {"g": g_vals}})
    return len(models) - 1


def add_nbody_gravity(
    defn: ScenarioDefinition,
    G: object,
    softening: object = 0.0,
    chunk_size: object | None = None,
) -> int:
    G_val, softening_val, chunk_val = validate_nbody_fields(G, softening, chunk_size)
    models = defn.setdefault("models", [])
    models.append(
        {
            "nbody_gravity": {
                "G": G_val,
                "softening": softening_val,
                "chunk_size": chunk_val,
            }
        }
    )
    return len(models) - 1


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


def remove_force(defn: ScenarioDefinition, model_index: int) -> None:
    models = defn.get("models", [])
    if model_index < 0 or model_index >= len(models):
        raise IndexError("force index out of range")
    del models[model_index]


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


def apply_uniform_gravity(
    defn: ScenarioDefinition, model_index: int, g: Sequence[object]
) -> None:
    g_vals = _validate_vec3(g, "g")
    _set_force(defn, model_index, "uniform_gravity", {"g": g_vals})


def apply_nbody_gravity(
    defn: ScenarioDefinition,
    model_index: int,
    G: object,
    softening: object,
    chunk_size: object | None,
) -> None:
    G_val, softening_val, chunk_val = validate_nbody_fields(G, softening, chunk_size)
    _set_force(
        defn,
        model_index,
        "nbody_gravity",
        {"G": G_val, "softening": softening_val, "chunk_size": chunk_val},
    )


def validate_nbody_fields(
    G: object, softening: object, chunk_size: object | None
) -> tuple[float, float, int | None]:
    G_val = _validate_float(G, "G", positive=True, allow_zero=False)
    softening_val = _validate_float(softening, "softening", positive=True, allow_zero=True)
    chunk_val = _parse_chunk_size(chunk_size)
    return G_val, softening_val, chunk_val


def _parse_chunk_size(value: object | None) -> int | None:
    if value is None or value == "":
        return None
    try:
        intval = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("chunk_size must be an integer or blank") from exc
    if intval < 1:
        raise ValueError("chunk_size must be >= 1")
    return intval


def _set_force(
    defn: ScenarioDefinition, model_index: int, key: str, payload: dict[str, object]
) -> None:
    models = defn.get("models", [])
    if model_index < 0 or model_index >= len(models):
        raise IndexError("force index out of range")
    models[model_index] = {key: payload}


def _validate_vec3(values: Sequence[object], name: str) -> list[float]:
    if len(values) != 3:
        raise ValueError(f"{name} must have 3 values")
    return [_validate_float(v, name) for v in values]


def _validate_float(
    value: object,
    name: str,
    positive: bool = False,
    allow_zero: bool = True,
) -> float:
    try:
        fval = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a number") from exc
    if not math.isfinite(fval):
        raise ValueError(f"{name} must be finite")
    if positive:
        if allow_zero and fval < 0:
            raise ValueError(f"{name} must be >= 0")
        if not allow_zero and fval <= 0:
            raise ValueError(f"{name} must be > 0")
    return fval
