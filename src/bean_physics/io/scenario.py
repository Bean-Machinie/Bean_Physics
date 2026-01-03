"""Scenario I/O and adapters."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from ..core.forces import CompositeModel, NBodyGravity, RigidBodyForces, UniformGravity
from ..core.integrators import SymplecticEuler, VelocityVerlet
from ..core.rigid_body.mass_properties import (
    box_inertia_body,
    mass_properties,
    rigid_body_from_points,
    sphere_inertia_body,
)
from ..core.state import ParticlesState, RigidBodiesState, SystemState


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
    dt = float(sim["dt"])
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
                pos=np.asarray(p["pos"], dtype=np.float64),
                vel=np.asarray(p["vel"], dtype=np.float64),
                mass=np.asarray(p["mass"], dtype=np.float64),
            )
        if "rigid_bodies" in entities:
            r = entities["rigid_bodies"]
            mass = np.asarray(r["mass"], dtype=np.float64)
            inertia_body = _rigid_body_inertia(r, mass)
            state.rigid_bodies = RigidBodiesState(
                pos=np.asarray(r["pos"], dtype=np.float64),
                vel=np.asarray(r["vel"], dtype=np.float64),
                quat=np.asarray(r["quat"], dtype=np.float64),
                omega=np.asarray(r["omega_body"], dtype=np.float64),
                mass=mass,
                inertia_body=inertia_body,
            )

    models = []
    for entry in defn.get("models", []):
        if "uniform_gravity" in entry:
            g = np.asarray(entry["uniform_gravity"]["g"], dtype=np.float64)
            models.append(UniformGravity(g=g))
        elif "nbody_gravity" in entry:
            cfg = entry["nbody_gravity"]
            models.append(
                NBodyGravity(
                    G=float(cfg["G"]),
                    softening=float(cfg.get("softening", 0.0)),
                    chunk_size=cfg.get("chunk_size"),
                )
            )
        elif "rigid_body_forces" in entry:
            if state.rigid_bodies is None or inertia_body is None:
                raise ValueError("rigid_body_forces requires rigid bodies and inertia")
            forces = entry["rigid_body_forces"].get("forces", [])
            rb_model = RigidBodyForces(
                mass=state.rigid_bodies.mass,
                inertia_body=inertia_body,
            )
            body_index = np.array([f["body_index"] for f in forces], dtype=np.int64)
            forces_world = np.array([f["force_world"] for f in forces], dtype=np.float64)
            points_body = np.array([f["point_body"] for f in forces], dtype=np.float64)
            rb_model.set_applied_forces(body_index, forces_world, points_body)
            models.append(rb_model)
        else:
            raise ValueError(f"unknown model entry: {entry}")

    model = CompositeModel(models=models)
    aux = {"inertia_body": inertia_body}
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
                for req in ("body_index", "point_body", "force_world"):
                    _require(f, req, "models.rigid_body_forces.forces")
                if len(f["point_body"]) != 3 or len(f["force_world"]) != 3:
                    raise ValueError("rigid_body_forces entries must use length-3 vectors")
        else:
            raise ValueError(f"unknown model type: {key}")

    return data


def _rigid_body_inertia(rigid: dict[str, Any], mass: np.ndarray) -> np.ndarray:
    if mass.shape[0] == 0:
        return np.zeros((0, 3, 3), dtype=np.float64)
    mass_dist = rigid["mass_distribution"]
    inertia = mass_dist.get("inertia_body")
    if inertia is not None:
        inertia_arr = np.asarray(inertia, dtype=np.float64)
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
                    box_inertia_body(mass[idx], np.asarray(params.get("size"), dtype=np.float64))
                )
            elif kind == "sphere":
                inertia_list.append(
                    sphere_inertia_body(mass[idx], float(params.get("radius")))
                )
            elif kind == "points":
                points = src.get("points")
                if points is None:
                    points = params.get("points", [])
                positions = np.asarray([p["pos"] for p in points], dtype=np.float64)
                masses = np.asarray([p["mass"] for p in points], dtype=np.float64)
                total_mass, _, inertia = rigid_body_from_points(positions, masses)
                if not np.isclose(total_mass, float(mass[idx])):
                    raise ValueError("rigid_bodies.mass must match sum of source points mass")
                inertia_list.append(inertia)
            else:
                raise ValueError("unsupported rigid body source kind")
        return np.stack(inertia_list, axis=0)

    points_body = np.asarray(mass_dist["points_body"], dtype=np.float64)
    point_masses = np.asarray(mass_dist["point_masses"], dtype=np.float64)
    total_mass, _, inertia = mass_properties(points_body, point_masses)
    mass_sum = float(np.sum(mass))
    if mass.shape[0] > 0 and not np.isclose(mass_sum / mass.shape[0], total_mass):
        raise ValueError("rigid_bodies mass does not match mass_distribution")
    return np.broadcast_to(inertia, (mass.shape[0], 3, 3)).copy()
