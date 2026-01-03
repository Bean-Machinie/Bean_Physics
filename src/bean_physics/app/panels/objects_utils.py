"""Pure helpers for object list and particle edits."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

import numpy as np

from ...core.rigid_body.mass_properties import (
    box_inertia_body,
    mass_properties,
    rigid_body_from_points,
    shift_points_to_com,
    sphere_inertia_body,
)
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


def list_rigid_bodies(defn: ScenarioDefinition) -> list[ObjectRef]:
    entities = defn.get("entities", {})
    rigid = entities.get("rigid_bodies")
    if not rigid:
        return []
    count = len(rigid.get("mass", []))
    return [ObjectRef(type="rigid_body", index=i) for i in range(count)]


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


def rigid_body_summary(defn: ScenarioDefinition, index: int) -> dict[str, object]:
    rigid = _require_rigid(defn)
    _validate_rigid_index(rigid, index)
    pos = rigid["pos"][index]
    vel = rigid["vel"][index]
    quat = rigid["quat"][index]
    omega = rigid["omega_body"][index]
    mass = rigid["mass"][index]
    source = _rigid_source_entry(rigid, index)
    inertia = _rigid_inertia_body(rigid, index)
    com = np.zeros(3, dtype=np.float64)
    if source and source.get("kind") == "points":
        points, masses = _points_from_source(source)
        total_mass, com, inertia = rigid_body_from_points(points, masses)
        mass = total_mass
    return {
        "x": float(pos[0]),
        "y": float(pos[1]),
        "z": float(pos[2]),
        "vx": float(vel[0]),
        "vy": float(vel[1]),
        "vz": float(vel[2]),
        "qw": float(quat[0]),
        "qx": float(quat[1]),
        "qy": float(quat[2]),
        "qz": float(quat[3]),
        "wx": float(omega[0]),
        "wy": float(omega[1]),
        "wz": float(omega[2]),
        "mass": float(mass),
        "com": [float(v) for v in com],
        "source": source,
        "inertia_body": inertia,
    }


def rigid_body_shapes(defn: ScenarioDefinition | None) -> list[dict[str, object]]:
    if defn is None:
        return []
    rigid = defn.get("entities", {}).get("rigid_bodies")
    if not rigid:
        return []
    count = len(rigid.get("mass", []))
    shapes: list[dict[str, object]] = []
    for idx in range(count):
        source = _rigid_source_entry(rigid, idx)
        if source is None:
            shapes.append({"kind": "sphere", "radius": 0.5})
            continue
        kind = source.get("kind")
        params = source.get("params", {})
        if kind == "box":
            shapes.append({"kind": "box", "size": params.get("size", [1.0, 1.0, 1.0])})
        elif kind == "sphere":
            shapes.append({"kind": "sphere", "radius": params.get("radius", 0.5)})
        elif kind == "points":
            shapes.append({"kind": "sphere", "radius": 0.2})
        else:
            shapes.append({"kind": "sphere", "radius": 0.5})
    return shapes


def rigid_body_points_body(defn: ScenarioDefinition | None, index: int) -> np.ndarray:
    if defn is None:
        return np.zeros((0, 3), dtype=np.float64)
    rigid = defn.get("entities", {}).get("rigid_bodies")
    if not rigid:
        return np.zeros((0, 3), dtype=np.float64)
    sources = rigid.get("source")
    if isinstance(sources, list) and index < len(sources):
        source = sources[index]
        if source.get("kind") == "points":
            points, _ = _points_from_source(source)
            return points
    return np.zeros((0, 3), dtype=np.float64)


def rigid_body_force_points(
    defn: ScenarioDefinition, body_index: int
) -> list[dict[str, object]]:
    models = defn.get("models", [])
    for entry in models:
        if "rigid_body_forces" in entry:
            forces = entry["rigid_body_forces"].get("forces", [])
            return [
                f
                for f in forces
                if int(f.get("body_index", -1)) == body_index
            ]
    return []


def set_rigid_body_force_points(
    defn: ScenarioDefinition,
    body_index: int,
    forces: list[dict[str, object]],
) -> None:
    models = defn.setdefault("models", [])
    entry = None
    for model in models:
        if "rigid_body_forces" in model:
            entry = model
            break
    if entry is None:
        entry = {"rigid_body_forces": {"forces": []}}
        models.append(entry)
    existing = entry["rigid_body_forces"].get("forces", [])
    kept = [f for f in existing if int(f.get("body_index", -1)) != body_index]
    entry["rigid_body_forces"]["forces"] = kept + forces


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


def add_rigid_body_template(
    defn: ScenarioDefinition, kind: str, params: dict[str, object] | None = None
) -> int:
    kind = kind.lower().strip()
    if kind not in {"box", "sphere", "points"}:
        raise ValueError("unsupported rigid body template")
    params = params or {}
    if kind == "box":
        size = params.get("size", [1.0, 1.0, 1.0])
        inertia = box_inertia_body(1.0, np.asarray(size, dtype=np.float64))
        source_params = {"size": [float(v) for v in size]}
    elif kind == "sphere":
        radius = float(params.get("radius", 0.5))
        inertia = sphere_inertia_body(1.0, radius)
        source_params = {"radius": radius}
    else:
        points = params.get("points")
        if points is None:
            points = [{"mass": 1.0, "pos": [0.0, 0.0, 0.0]}]
        points_body, masses = _points_from_list(points)
        total_mass, _, inertia = rigid_body_from_points(points_body, masses)
        points_body, _ = shift_points_to_com(points_body, masses)
        source_params = {"points": _points_to_list(points_body, masses)}

    entities = defn.setdefault("entities", {})
    rigid = entities.setdefault(
        "rigid_bodies",
        {
            "pos": [],
            "vel": [],
            "quat": [],
            "omega_body": [],
            "mass": [],
            "mass_distribution": {"points_body": [[0.0, 0.0, 0.0]], "point_masses": [1.0]},
        },
    )
    for key in ("pos", "vel", "quat", "omega_body", "mass"):
        rigid.setdefault(key, [])
    rigid.setdefault(
        "mass_distribution",
        {"points_body": [[0.0, 0.0, 0.0]], "point_masses": [1.0]},
    )

    rigid["pos"].append([0.0, 0.0, 0.0])
    rigid["vel"].append([0.0, 0.0, 0.0])
    rigid["quat"].append([1.0, 0.0, 0.0, 0.0])
    rigid["omega_body"].append([0.0, 0.0, 0.0])
    rigid["mass"].append(1.0 if kind != "points" else float(total_mass))

    sources = rigid.setdefault("source", [])
    if kind == "points":
        sources.append(
            {
                "kind": kind,
                "points": source_params["points"],
                "mass": rigid["mass"][-1],
            }
        )
    else:
        sources.append(
            {
                "kind": kind,
                "params": source_params,
                "mass": rigid["mass"][-1],
            }
        )

    inertia_list = _rigid_inertia_list(rigid)
    inertia_list.append(inertia)
    _set_rigid_inertia_list(rigid, inertia_list)
    if kind == "points":
        mass_dist = rigid["mass_distribution"]
        mass_dist["points_body"] = [p for p in points_body.tolist()]
        mass_dist["point_masses"] = [float(m) for m in masses.tolist()]
    return len(rigid["mass"]) - 1


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


def remove_rigid_body(defn: ScenarioDefinition, index: int) -> None:
    rigid = defn.get("entities", {}).get("rigid_bodies")
    if not rigid:
        raise ValueError("no rigid bodies to remove")
    _validate_rigid_index(rigid, index)
    for key in ("pos", "vel", "quat", "omega_body", "mass"):
        del rigid[key][index]
    sources = rigid.get("source")
    if isinstance(sources, list) and len(sources) > index:
        del sources[index]
    inertia_list = _rigid_inertia_list(rigid)
    if inertia_list:
        del inertia_list[index]
        _set_rigid_inertia_list(rigid, inertia_list)
    if not rigid["mass"]:
        entities = defn.get("entities", {})
        entities.pop("rigid_bodies", None)
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


def apply_rigid_body_edit(
    defn: ScenarioDefinition,
    index: int,
    kind: str,
    params: dict[str, object],
    mass: object,
    pos: Sequence[object],
    vel: Sequence[object],
    quat: Sequence[object],
    omega_body: Sequence[object],
) -> None:
    rigid = _require_rigid(defn)
    _validate_rigid_index(rigid, index)

    mass_val = _validate_float(mass, "mass", positive=True, allow_zero=False)
    pos_vals = _validate_vec3(pos, "pos")
    vel_vals = _validate_vec3(vel, "vel")
    omega_vals = _validate_vec3(omega_body, "omega_body")
    quat_vals = _validate_quat(quat)

    kind = kind.lower().strip()
    if kind not in {"box", "sphere", "points"}:
        raise ValueError("unsupported rigid body template")

    if kind == "box":
        size = params.get("size")
        if size is None:
            raise ValueError("size is required for box")
        inertia = box_inertia_body(mass_val, np.asarray(size, dtype=np.float64))
        source_params = {"size": [float(v) for v in size]}
    elif kind == "sphere":
        radius = params.get("radius")
        if radius is None:
            raise ValueError("radius is required for sphere")
        inertia = sphere_inertia_body(mass_val, float(radius))
        source_params = {"radius": float(radius)}
    else:
        points = params.get("points")
        if points is None:
            raise ValueError("points are required for points source")
        points_body, masses = _points_from_list(points)
        total_mass, _, inertia = rigid_body_from_points(points_body, masses)
        points_body, _ = shift_points_to_com(points_body, masses)
        source_params = {"points": _points_to_list(points_body, masses)}
        mass_val = total_mass

    rigid["pos"][index] = pos_vals
    rigid["vel"][index] = vel_vals
    rigid["quat"][index] = quat_vals
    rigid["omega_body"][index] = omega_vals
    rigid["mass"][index] = mass_val

    sources = rigid.setdefault("source", [])
    while len(sources) < len(rigid["mass"]):
        sources.append(
            {"kind": "box", "params": {"size": [1.0, 1.0, 1.0]}, "mass": 1.0}
        )
    if kind == "points":
        sources[index] = {"kind": kind, "points": source_params["points"], "mass": mass_val}
    else:
        sources[index] = {"kind": kind, "params": source_params, "mass": mass_val}

    inertia_list = _rigid_inertia_list(rigid)
    while len(inertia_list) < len(rigid["mass"]):
        inertia_list.append(np.eye(3, dtype=np.float64))
    inertia_list[index] = inertia
    _set_rigid_inertia_list(rigid, inertia_list)
    if kind == "points":
        mass_dist = rigid.setdefault(
            "mass_distribution",
            {"points_body": [[0.0, 0.0, 0.0]], "point_masses": [1.0]},
        )
        mass_dist["points_body"] = [p for p in points_body.tolist()]
        mass_dist["point_masses"] = [float(m) for m in masses.tolist()]


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


def _require_rigid(defn: ScenarioDefinition) -> dict[str, object]:
    rigid = defn.get("entities", {}).get("rigid_bodies")
    if not rigid:
        raise ValueError("no rigid bodies in scenario")
    return rigid


def _validate_rigid_index(rigid: dict[str, object], index: int) -> None:
    if index < 0 or index >= len(rigid.get("mass", [])):
        raise IndexError("rigid body index out of range")


def _rigid_source_entry(
    rigid: dict[str, object], index: int
) -> dict[str, object] | None:
    sources = rigid.get("source")
    if not isinstance(sources, list) or index >= len(sources):
        return None
    return sources[index]


def _rigid_inertia_body(rigid: dict[str, object], index: int) -> np.ndarray:
    inertia_list = _rigid_inertia_list(rigid)
    if inertia_list:
        if len(inertia_list) == 1:
            return inertia_list[0]
        return inertia_list[index]
    mass_dist = rigid.get("mass_distribution", {})
    points_body = np.asarray(mass_dist.get("points_body", []), dtype=np.float64)
    point_masses = np.asarray(mass_dist.get("point_masses", []), dtype=np.float64)
    if points_body.size == 0 or point_masses.size == 0:
        return np.eye(3, dtype=np.float64)
    _, _, inertia = mass_properties(points_body, point_masses)
    return inertia


def _rigid_inertia_list(rigid: dict[str, object]) -> list[np.ndarray]:
    mass_dist = rigid.get("mass_distribution", {})
    inertia = mass_dist.get("inertia_body")
    if inertia is None:
        return []
    arr = np.asarray(inertia, dtype=np.float64)
    if arr.ndim == 2:
        return [arr]
    if arr.ndim == 3:
        return [arr[i] for i in range(arr.shape[0])]
    return []


def _set_rigid_inertia_list(
    rigid: dict[str, object], inertia_list: Sequence[np.ndarray]
) -> None:
    mass_dist = rigid.setdefault(
        "mass_distribution",
        {"points_body": [[0.0, 0.0, 0.0]], "point_masses": [1.0]},
    )
    if len(inertia_list) == 1:
        mass_dist["inertia_body"] = inertia_list[0].tolist()
    else:
        mass_dist["inertia_body"] = [arr.tolist() for arr in inertia_list]


def _points_from_source(source: dict[str, object]) -> tuple[np.ndarray, np.ndarray]:
    points = source.get("points")
    if points is None:
        params = source.get("params", {})
        points = params.get("points", [])
    return _points_from_list(points)


def _points_from_list(
    points: Sequence[dict[str, object]],
) -> tuple[np.ndarray, np.ndarray]:
    if not points:
        raise ValueError("points list must not be empty")
    positions = []
    masses = []
    for idx, entry in enumerate(points):
        if not isinstance(entry, dict):
            raise ValueError(f"point {idx} must be an object")
        mass = entry.get("mass")
        pos = entry.get("pos")
        if mass is None or pos is None:
            raise ValueError(f"point {idx} must have mass and pos")
        pos_vals = _validate_vec3(pos, f"points[{idx}].pos")
        mass_val = _validate_float(mass, f"points[{idx}].mass", positive=True, allow_zero=False)
        positions.append(pos_vals)
        masses.append(mass_val)
    return np.asarray(positions, dtype=np.float64), np.asarray(masses, dtype=np.float64)


def _points_to_list(points_body: np.ndarray, masses: np.ndarray) -> list[dict[str, object]]:
    return [
        {"mass": float(masses[i]), "pos": [float(v) for v in points_body[i]]}
        for i in range(points_body.shape[0])
    ]


def _validate_quat(values: Sequence[object]) -> list[float]:
    if len(values) != 4:
        raise ValueError("quat must have 4 values")
    cleaned = []
    for value in values:
        fval = _validate_float(value, "quat", positive=False, allow_zero=True)
        cleaned.append(fval)
    q = np.asarray(cleaned, dtype=np.float64)
    norm = np.linalg.norm(q)
    if norm == 0.0:
        raise ValueError("quat must be non-zero")
    q = q / norm
    return [float(v) for v in q]


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
