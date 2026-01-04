"""Scenario I/O and adapters."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from ..core.forces import CompositeModel, NBodyGravity, RigidBodyForces, UniformGravity
from ..core.integrators import SymplecticEuler, VelocityVerlet
from ..core.impulse_events import ImpulseEvent
from ..core.rigid_body.mass_properties import (
    box_inertia_body,
    mass_properties,
    rigid_body_from_points,
    sphere_inertia_body,
)
from ..core.math.quat import quat_to_rotmat
from ..core.state import ParticlesState, RigidBodiesState, SystemState
from .units import PRESETS, UnitsConfig, config_from_defn, to_si


ScenarioDefinition = dict[str, Any]


def load_scenario(path: str | Path) -> ScenarioDefinition:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return _validate_scenario_v1(data)


def save_scenario(path: str | Path, defn: ScenarioDefinition) -> None:
    Path(path).write_text(
        json.dumps(defn, indent=2, sort_keys=True),
        encoding="utf-8",
    )


def scenario_to_runtime(
    defn: ScenarioDefinition,
) -> tuple[SystemState, CompositeModel, object, float, int, dict[str, Any]]:
    sim = defn["simulation"]
    units_cfg = config_from_defn(defn)
    dt = float(to_si(sim["dt"], "time", units_cfg))
    steps = int(sim["steps"])
    integrator_name = sim["integrator"]
    if integrator_name == "symplectic_euler":
        integrator = SymplecticEuler()
    elif integrator_name == "velocity_verlet":
        integrator = VelocityVerlet()
    else:
        raise ValueError(f"unsupported integrator: {integrator_name}")

    state = SystemState()
    inertia_body = None
    if "entities" in defn:
        entities = defn["entities"]
        if "particles" in entities:
            p = entities["particles"]
            state.particles = ParticlesState(
                pos=np.asarray(to_si(p["pos"], "length", units_cfg), dtype=np.float64),
                vel=np.asarray(to_si(p["vel"], "velocity", units_cfg), dtype=np.float64),
                mass=np.asarray(to_si(p["mass"], "mass", units_cfg), dtype=np.float64),
            )
        if "rigid_bodies" in entities:
            r = entities["rigid_bodies"]
            mass = np.asarray(to_si(r["mass"], "mass", units_cfg), dtype=np.float64)
            inertia_body = _rigid_body_inertia(r, mass, units_cfg)
            state.rigid_bodies = RigidBodiesState(
                pos=np.asarray(to_si(r["pos"], "length", units_cfg), dtype=np.float64),
                vel=np.asarray(to_si(r["vel"], "velocity", units_cfg), dtype=np.float64),
                quat=np.asarray(r["quat"], dtype=np.float64),
                omega=np.asarray(to_si(r["omega_body"], "omega", units_cfg), dtype=np.float64),
                mass=mass,
                inertia_body=inertia_body,
            )

    models = []
    for entry in defn.get("models", []):
        if "uniform_gravity" in entry:
            g = np.asarray(to_si(entry["uniform_gravity"]["g"], "accel", units_cfg), dtype=np.float64)
            models.append(UniformGravity(g=g))
        elif "nbody_gravity" in entry:
            cfg = entry["nbody_gravity"]
            models.append(
                NBodyGravity(
                    G=float(to_si(cfg["G"], "G", units_cfg)),
                    softening=float(cfg.get("softening", 0.0)),
                    chunk_size=cfg.get("chunk_size"),
                )
            )
        elif "rigid_body_forces" in entry:
            if state.rigid_bodies is None or inertia_body is None:
                raise ValueError("rigid_body_forces requires rigid bodies and inertia")
            forces = entry["rigid_body_forces"].get("forces", [])
            forces = [f for f in forces if f.get("enabled", True)]
            rb_model = RigidBodyForces(
                mass=state.rigid_bodies.mass,
                inertia_body=inertia_body,
            )
            if forces:
                body_index = np.array([f["body_index"] for f in forces], dtype=np.int64)
                points_body = np.array(
                    to_si([f["point_body"] for f in forces], "length", units_cfg),
                    dtype=np.float64,
                )
                forces_body = _resolve_force_body(forces, state, units_cfg)
            else:
                body_index = np.zeros(0, dtype=np.int64)
                points_body = np.zeros((0, 3), dtype=np.float64)
                forces_body = np.zeros((0, 3), dtype=np.float64)
            rb_model.set_applied_forces(body_index, forces_body, points_body)
            models.append(rb_model)
        else:
            raise ValueError(f"unknown model entry: {entry}")

    model = CompositeModel(models=models)
    impulse_events = _parse_impulse_events(defn, units_cfg)
    aux = {"inertia_body": inertia_body, "impulse_events": impulse_events}
    return state, model, integrator, dt, steps, aux


def _require(obj: dict[str, Any], key: str, ctx: str) -> Any:
    if key not in obj:
        raise ValueError(f"missing required field: {ctx}.{key}")
    return obj[key]


def _validate_array(arr: Any, shape_suffix: tuple[int, ...], ctx: str) -> None:
    a = np.asarray(arr, dtype=np.float64)
    if a.ndim != len(shape_suffix) + 1:
        raise ValueError(f"{ctx} must be an array with shape (*, {', '.join(map(str, shape_suffix))})")
    if tuple(a.shape[1:]) != shape_suffix:
        raise ValueError(f"{ctx} must have shape (*, {', '.join(map(str, shape_suffix))})")


def _validate_ids(
    values: Any,
    expected_len: int,
    ctx: str,
    id_map: dict[str, tuple[str, int]],
    kind: str,
) -> None:
    if not isinstance(values, list) or len(values) != expected_len:
        raise ValueError(f"{ctx} must be a list of length {expected_len}")
    for idx, entry in enumerate(values):
        if not isinstance(entry, str) or not entry:
            raise ValueError(f"{ctx}[{idx}] must be a non-empty string")
        if entry in id_map:
            raise ValueError(f"duplicate object id: {entry}")
        id_map[entry] = (kind, idx)


def _validate_scenario_v1(data: dict[str, Any]) -> ScenarioDefinition:
    if not isinstance(data, dict):
        raise ValueError("scenario must be a JSON object")
    if data.get("schema_version") != 1:
        raise ValueError("schema_version must be 1")

    sim = _require(data, "simulation", "scenario")
    _require(sim, "dt", "simulation")
    _require(sim, "steps", "simulation")
    _require(sim, "integrator", "simulation")
    if sim["dt"] <= 0:
        raise ValueError("simulation.dt must be > 0")
    if sim["steps"] < 0:
        raise ValueError("simulation.steps must be >= 0")
    if sim["integrator"] not in {"symplectic_euler", "velocity_verlet"}:
        raise ValueError("simulation.integrator invalid")

    if "sampling" in data:
        every = data["sampling"].get("every")
        if every is not None and every <= 0:
            raise ValueError("sampling.every must be > 0")

    if "units" in data:
        units = data["units"]
        if not isinstance(units, dict):
            raise ValueError("units must be an object")
        preset = str(units.get("preset", "SI"))
        preset_upper = preset.upper()
        if preset_upper not in PRESETS:
            raise ValueError("units.preset is not supported")
        if "enabled" in units and not isinstance(units["enabled"], bool):
            raise ValueError("units.enabled must be boolean")

    entities = data.get("entities", {})
    if "particles" in entities:
        p = entities["particles"]
        _require(p, "pos", "particles")
        _require(p, "vel", "particles")
        _require(p, "mass", "particles")
        _validate_array(p["pos"], (3,), "particles.pos")
        _validate_array(p["vel"], (3,), "particles.vel")
        if len(p["mass"]) != len(p["pos"]):
            raise ValueError("particles.mass must have length N")
        if "visual" in p:
            _validate_visual_list(p["visual"], len(p["mass"]), "particles.visual")

    id_map: dict[str, tuple[str, int]] = {}
    if "particles" in entities:
        p = entities["particles"]
        if "ids" in p:
            _validate_ids(p["ids"], len(p["mass"]), "particles.ids", id_map, "particle")

    if "rigid_bodies" in entities:
        r = entities["rigid_bodies"]
        for key in ("pos", "vel", "quat", "omega_body", "mass", "mass_distribution"):
            _require(r, key, "rigid_bodies")
        _validate_array(r["pos"], (3,), "rigid_bodies.pos")
        _validate_array(r["vel"], (3,), "rigid_bodies.vel")
        _validate_array(r["omega_body"], (3,), "rigid_bodies.omega_body")
        _validate_array(r["quat"], (4,), "rigid_bodies.quat")
        if len(r["mass"]) != len(r["pos"]):
            raise ValueError("rigid_bodies.mass must have length M")
        if "visual" in r:
            _validate_visual_list(r["visual"], len(r["mass"]), "rigid_bodies.visual")
        if "ids" in r:
            _validate_ids(r["ids"], len(r["mass"]), "rigid_bodies.ids", id_map, "rigid_body")

        md = r["mass_distribution"]
        _require(md, "points_body", "rigid_bodies.mass_distribution")
        _require(md, "point_masses", "rigid_bodies.mass_distribution")
        _validate_array(md["points_body"], (3,), "rigid_bodies.mass_distribution.points_body")
        if len(md["point_masses"]) != len(md["points_body"]):
            raise ValueError("rigid_bodies.mass_distribution.point_masses must have length K")
        if "inertia_body" in md and md["inertia_body"] is not None:
            I = np.asarray(md["inertia_body"], dtype=np.float64)
            if I.ndim not in (2, 3) or I.shape[-2:] != (3, 3):
                raise ValueError("rigid_bodies.mass_distribution.inertia_body must be (3,3) or (M,3,3)")
            if I.ndim == 3 and I.shape[0] != len(r["pos"]):
                raise ValueError("rigid_bodies.mass_distribution.inertia_body must have length M")

        if "source" in r:
            sources = r["source"]
            if not isinstance(sources, list) or len(sources) != len(r["pos"]):
                raise ValueError("rigid_bodies.source must be a list of length M")
            for idx, src in enumerate(sources):
                if not isinstance(src, dict):
                    raise ValueError("rigid_bodies.source entries must be objects")
                kind = _require(src, "kind", f"rigid_bodies.source[{idx}]")
                params = src.get("params", {})
                mass = _require(src, "mass", f"rigid_bodies.source[{idx}]")
                if kind not in {"box", "sphere", "points"}:
                    raise ValueError("rigid_bodies.source.kind must be 'box', 'sphere', or 'points'")
                if kind == "box":
                    params = _require(src, "params", f"rigid_bodies.source[{idx}]")
                    size = params.get("size")
                    if size is None or len(size) != 3:
                        raise ValueError("rigid_bodies.source.params.size must have length 3")
                if kind == "sphere":
                    params = _require(src, "params", f"rigid_bodies.source[{idx}]")
                    radius = params.get("radius")
                    if radius is None:
                        raise ValueError("rigid_bodies.source.params.radius is required")
                if kind == "points":
                    points = src.get("points")
                    if points is None:
                        points = params.get("points")
                    if not isinstance(points, list) or not points:
                        raise ValueError("rigid_bodies.source.points must be a non-empty list")
                    total_mass = 0.0
                    for p_idx, point in enumerate(points):
                        if not isinstance(point, dict):
                            raise ValueError("rigid_bodies.source.points entries must be objects")
                        if "mass" not in point or "pos" not in point:
                            raise ValueError("rigid_bodies.source.points entries require mass and pos")
                        if len(point["pos"]) != 3:
                            raise ValueError("rigid_bodies.source.points.pos must have length 3")
                        if float(point["mass"]) <= 0.0:
                            raise ValueError("rigid_bodies.source.points.mass must be > 0")
                        total_mass += float(point["mass"])
                    if not np.isclose(float(r["mass"][idx]), total_mass):
                        raise ValueError("rigid_bodies.mass must match sum of source points mass")
                if float(mass) <= 0.0:
                    raise ValueError("rigid_bodies.source.mass must be > 0")
                if not np.isclose(float(mass), float(r["mass"][idx])):
                    raise ValueError("rigid_bodies.source.mass must match rigid_bodies.mass")

    for entry in data.get("models", []):
        if not isinstance(entry, dict) or len(entry) != 1:
            raise ValueError("each model entry must be an object with one key")
        key = next(iter(entry))
        cfg = entry[key]
        if key == "uniform_gravity":
            _require(cfg, "g", "models.uniform_gravity")
            if len(cfg["g"]) != 3:
                raise ValueError("uniform_gravity.g must have length 3")
        elif key == "nbody_gravity":
            _require(cfg, "G", "models.nbody_gravity")
        elif key == "rigid_body_forces":
            forces = cfg.get("forces", [])
            for f in forces:
                for req in ("body_index", "point_body"):
                    _require(f, req, "models.rigid_body_forces.forces")
                if "force_body" not in f and "force_world" not in f:
                    raise ValueError("rigid_body_forces entries must include force_body or force_world")
                if len(f["point_body"]) != 3:
                    raise ValueError("rigid_body_forces entries must use length-3 vectors")
                if "force_body" in f and len(f["force_body"]) != 3:
                    raise ValueError("rigid_body_forces.force_body must have length 3")
                if "force_world" in f and len(f["force_world"]) != 3:
                    raise ValueError("rigid_body_forces.force_world must have length 3")
                if "enabled" in f and not isinstance(f["enabled"], bool):
                    raise ValueError("rigid_body_forces.enabled must be boolean")
                if "throttle" in f:
                    throttle = float(f["throttle"])
                    if throttle < 0.0 or throttle > 1.0:
                        raise ValueError("rigid_body_forces.throttle must be in [0, 1]")
                if "name" in f and not isinstance(f["name"], str):
                    raise ValueError("rigid_body_forces.name must be a string")
                if "group" in f and not isinstance(f["group"], str):
                    raise ValueError("rigid_body_forces.group must be a string")
        else:
            raise ValueError(f"unknown model type: {key}")

    if "impulse_events" in data:
        _validate_impulse_events(data["impulse_events"], id_map)

    return data


def _validate_impulse_events(
    events: Any, id_map: dict[str, tuple[str, int]]
) -> None:
    if not isinstance(events, list):
        raise ValueError("impulse_events must be a list")
    if not id_map:
        raise ValueError("impulse_events require entities.*.ids to resolve targets")
    for idx, event in enumerate(events):
        if not isinstance(event, dict):
            raise ValueError("impulse_events entries must be objects")
        ctx = f"impulse_events[{idx}]"
        t = _require(event, "t", ctx)
        if float(t) < 0.0:
            raise ValueError(f"{ctx}.t must be >= 0")
        target = _require(event, "target", ctx)
        if not isinstance(target, str):
            raise ValueError(f"{ctx}.target must be a string")
        if target not in id_map:
            raise ValueError(f"{ctx}.target not found in scenario ids")
        delta = _require(event, "delta_v_world", ctx)
        if len(delta) != 3:
            raise ValueError(f"{ctx}.delta_v_world must have length 3")
        label = event.get("label")
        if label is not None and not isinstance(label, str):
            raise ValueError(f"{ctx}.label must be a string")


def _validate_visual_list(values: Any, expected_len: int, ctx: str) -> None:
    if not isinstance(values, list) or len(values) != expected_len:
        raise ValueError(f"{ctx} must be a list of length {expected_len}")
    for idx, entry in enumerate(values):
        if entry is None:
            continue
        if not isinstance(entry, dict):
            raise ValueError(f"{ctx}[{idx}] must be an object")
        _validate_visual_block(entry, f"{ctx}[{idx}]")


def _validate_visual_block(entry: dict[str, Any], ctx: str) -> None:
    kind = entry.get("kind")
    if kind not in {"mesh", "primitive"}:
        raise ValueError(f"{ctx}.kind must be 'mesh' or 'primitive'")
    if kind == "mesh":
        if "mesh_path" not in entry:
            raise ValueError(f"{ctx}.mesh_path is required for mesh visuals")
    if "scale" in entry:
        scale = entry["scale"]
        if isinstance(scale, (int, float)):
            return
        if len(scale) != 3:
            raise ValueError(f"{ctx}.scale must have length 3")
    if "offset_body" in entry and len(entry["offset_body"]) != 3:
        raise ValueError(f"{ctx}.offset_body must have length 3")
    if "rotation_body_quat" in entry and len(entry["rotation_body_quat"]) != 4:
        raise ValueError(f"{ctx}.rotation_body_quat must have length 4")
    if "color_tint" in entry and len(entry["color_tint"]) != 3:
        raise ValueError(f"{ctx}.color_tint must have length 3")


def _parse_impulse_events(
    defn: dict[str, Any], units_cfg: UnitsConfig
) -> list[ImpulseEvent]:
    events = defn.get("impulse_events", [])
    if not events:
        return []
    entities = defn.get("entities", {})
    id_map: dict[str, tuple[str, int]] = {}
    particles = entities.get("particles")
    if isinstance(particles, dict):
        ids = particles.get("ids", [])
        for idx, entry in enumerate(ids):
            id_map[str(entry)] = ("particle", idx)
    rigid = entities.get("rigid_bodies")
    if isinstance(rigid, dict):
        ids = rigid.get("ids", [])
        for idx, entry in enumerate(ids):
            id_map[str(entry)] = ("rigid_body", idx)

    if not id_map:
        raise ValueError("impulse_events require entities.*.ids to resolve targets")

    parsed: list[ImpulseEvent] = []
    for event in events:
        target = str(event["target"])
        if target not in id_map:
            raise ValueError(f"impulse_events target not found: {target}")
        target_type, target_index = id_map[target]
        delta_v = np.asarray(
            to_si(event["delta_v_world"], "velocity", units_cfg),
            dtype=np.float64,
        )
        parsed.append(
            ImpulseEvent(
                t=float(to_si(event["t"], "time", units_cfg)),
                target_type=target_type,
                target_index=target_index,
                target_id=target,
                delta_v=delta_v,
                label=event.get("label"),
            )
        )
    return sorted(parsed, key=lambda evt: evt.t)


def _resolve_force_body(
    forces: list[dict[str, Any]], state: SystemState, units_cfg: UnitsConfig
) -> np.ndarray:
    forces_body = []
    if state.rigid_bodies is None:
        return np.zeros((0, 3), dtype=np.float64)
    rot = quat_to_rotmat(state.rigid_bodies.quat)
    for force in forces:
        throttle = float(force.get("throttle", 1.0))
        if "force_body" in force:
            base = np.asarray(to_si(force["force_body"], "force", units_cfg), dtype=np.float64)
        else:
            body_idx = int(force["body_index"])
            f_world = np.asarray(to_si(force["force_world"], "force", units_cfg), dtype=np.float64)
            r = rot[body_idx]
            base = r.T @ f_world
        forces_body.append((throttle * base).tolist())
    return np.asarray(forces_body, dtype=np.float64)


def _rigid_body_inertia(
    rigid: dict[str, Any], mass: np.ndarray, units_cfg: UnitsConfig
) -> np.ndarray:
    if mass.shape[0] == 0:
        return np.zeros((0, 3, 3), dtype=np.float64)
    mass_dist = rigid["mass_distribution"]
    inertia = mass_dist.get("inertia_body")
    if inertia is not None:
        inertia_arr = np.asarray(inertia, dtype=np.float64)
        inertia_arr = np.asarray(to_si(inertia_arr, "inertia", units_cfg), dtype=np.float64)
        if inertia_arr.ndim == 2:
            return np.broadcast_to(inertia_arr, (mass.shape[0], 3, 3)).copy()
        if inertia_arr.ndim == 3:
            return inertia_arr
        raise ValueError("inertia_body must be (3,3) or (M,3,3)")

    sources = rigid.get("source")
    if sources is not None:
        inertia_list = []
        for idx, src in enumerate(sources):
            kind = src.get("kind")
            params = src.get("params", {})
            if kind == "box":
                inertia_list.append(
                    box_inertia_body(
                        mass[idx],
                        np.asarray(
                            to_si(params.get("size"), "length", units_cfg),
                            dtype=np.float64,
                        ),
                    )
                )
            elif kind == "sphere":
                inertia_list.append(
                    sphere_inertia_body(
                        mass[idx],
                        float(to_si(params.get("radius"), "length", units_cfg)),
                    )
                )
            elif kind == "points":
                points = src.get("points")
                if points is None:
                    points = params.get("points", [])
                positions = np.asarray(
                    to_si([p["pos"] for p in points], "length", units_cfg), dtype=np.float64
                )
                masses = np.asarray(
                    to_si([p["mass"] for p in points], "mass", units_cfg), dtype=np.float64
                )
                total_mass, _, inertia = rigid_body_from_points(positions, masses)
                if not np.isclose(total_mass, float(mass[idx])):
                    raise ValueError("rigid_bodies.mass must match sum of source points mass")
                inertia_list.append(inertia)
            else:
                raise ValueError("unsupported rigid body source kind")
        return np.stack(inertia_list, axis=0)

    points_body = np.asarray(to_si(mass_dist["points_body"], "length", units_cfg), dtype=np.float64)
    point_masses = np.asarray(to_si(mass_dist["point_masses"], "mass", units_cfg), dtype=np.float64)
    total_mass, _, inertia = mass_properties(points_body, point_masses)
    mass_sum = float(np.sum(mass))
    if mass.shape[0] > 0 and not np.isclose(mass_sum / mass.shape[0], total_mass):
        raise ValueError("rigid_bodies mass does not match mass_distribution")
    return np.broadcast_to(inertia, (mass.shape[0], 3, 3)).copy()
