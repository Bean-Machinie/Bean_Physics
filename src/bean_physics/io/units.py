"""Units presets and conversions for scenario I/O and GUI."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True, slots=True)
class UnitPreset:
    name: str
    L: float
    M: float
    T: float
    length_label: str
    mass_label: str
    time_label: str


@dataclass(frozen=True, slots=True)
class UnitsConfig:
    preset: str
    enabled: bool = True


PRESETS: dict[str, UnitPreset] = {
    "SI": UnitPreset("SI", 1.0, 1.0, 1.0, "m", "kg", "s"),
    "KM": UnitPreset("KM", 1000.0, 1.0, 1.0, "km", "kg", "s"),
    "ASTRO": UnitPreset(
        "ASTRO",
        149_597_870_700.0,
        1.98847e30,
        86_400.0,
        "AU",
        "Msun",
        "day",
    ),
}


def preset_names() -> list[str]:
    return list(PRESETS.keys())


def get_preset(name: str) -> UnitPreset:
    if name not in PRESETS:
        raise ValueError(f"unknown units preset: {name}")
    return PRESETS[name]


def default_config() -> UnitsConfig:
    return UnitsConfig(preset="SI", enabled=True)


def config_from_defn(defn: dict[str, Any]) -> UnitsConfig:
    units = defn.get("units", {})
    if not isinstance(units, dict):
        return default_config()
    preset = str(units.get("preset", "SI")).upper()
    enabled = bool(units.get("enabled", True))
    if preset not in PRESETS:
        preset = "SI"
    return UnitsConfig(preset=preset, enabled=enabled)


def ensure_units_block(defn: dict[str, Any]) -> UnitsConfig:
    cfg = config_from_defn(defn)
    defn["units"] = {"preset": cfg.preset, "enabled": cfg.enabled}
    return cfg


def label_for(kind: str, cfg: UnitsConfig) -> str:
    preset = _effective_preset(cfg)
    l = preset.length_label
    m = preset.mass_label
    t = preset.time_label
    if kind == "length":
        return l
    if kind == "mass":
        return m
    if kind == "time":
        return t
    if kind == "velocity":
        return f"{l}/{t}"
    if kind == "accel":
        return f"{l}/{t}^2"
    if kind == "force":
        return f"{m}*{l}/{t}^2"
    if kind == "omega":
        return f"1/{t}"
    if kind == "inertia":
        return f"{m}*{l}^2"
    if kind == "torque":
        return f"{m}*{l}^2/{t}^2"
    if kind == "G":
        return f"{l}^3/({m}*{t}^2)"
    return ""


def to_si(value: Any, kind: str, cfg: UnitsConfig) -> Any:
    scale = _scale_for_kind(kind, _effective_preset(cfg))
    return _apply_scale(value, scale)


def from_si(value: Any, kind: str, cfg: UnitsConfig) -> Any:
    scale = _scale_for_kind(kind, _effective_preset(cfg))
    return _apply_scale(value, 1.0 / scale)


def convert_value(
    value: Any, kind: str, from_cfg: UnitsConfig, to_cfg: UnitsConfig
) -> Any:
    if from_cfg == to_cfg:
        return value
    scale_from = _scale_for_kind(kind, _effective_preset(from_cfg))
    scale_to = _scale_for_kind(kind, _effective_preset(to_cfg))
    return _apply_scale(value, scale_from / scale_to)


def convert_definition_units(
    defn: dict[str, Any], from_cfg: UnitsConfig, to_cfg: UnitsConfig
) -> None:
    if from_cfg == to_cfg:
        return

    def _convert_list(values: Iterable[Any], kind: str) -> list:
        arr = np.asarray(list(values), dtype=np.float64)
        converted = convert_value(arr, kind, from_cfg, to_cfg)
        return converted.tolist()

    sim = defn.get("simulation", {})
    if "dt" in sim:
        sim["dt"] = float(convert_value(sim["dt"], "time", from_cfg, to_cfg))

    entities = defn.get("entities", {})
    particles = entities.get("particles")
    if isinstance(particles, dict):
        if "pos" in particles:
            particles["pos"] = _convert_list(particles["pos"], "length")
        if "vel" in particles:
            particles["vel"] = _convert_list(particles["vel"], "velocity")
        if "mass" in particles:
            particles["mass"] = _convert_list(particles["mass"], "mass")
        visuals = particles.get("visual")
        if isinstance(visuals, list):
            for visual in visuals:
                if isinstance(visual, dict) and "offset_body" in visual:
                    visual["offset_body"] = convert_value(
                        visual["offset_body"], "length", from_cfg, to_cfg
                    ).tolist()

    rigid = entities.get("rigid_bodies")
    if isinstance(rigid, dict):
        if "pos" in rigid:
            rigid["pos"] = _convert_list(rigid["pos"], "length")
        if "vel" in rigid:
            rigid["vel"] = _convert_list(rigid["vel"], "velocity")
        if "omega_body" in rigid:
            rigid["omega_body"] = _convert_list(rigid["omega_body"], "omega")
        if "mass" in rigid:
            rigid["mass"] = _convert_list(rigid["mass"], "mass")

        mass_dist = rigid.get("mass_distribution", {})
        if "points_body" in mass_dist:
            mass_dist["points_body"] = _convert_list(
                mass_dist["points_body"], "length"
            )
        if "point_masses" in mass_dist:
            mass_dist["point_masses"] = _convert_list(
                mass_dist["point_masses"], "mass"
            )
        inertia = mass_dist.get("inertia_body")
        if inertia is not None:
            inertia_arr = np.asarray(inertia, dtype=np.float64)
            inertia_arr = convert_value(inertia_arr, "inertia", from_cfg, to_cfg)
            mass_dist["inertia_body"] = inertia_arr.tolist()

        sources = rigid.get("source")
        if isinstance(sources, list):
            for src in sources:
                if not isinstance(src, dict):
                    continue
                kind = src.get("kind")
                if kind == "box":
                    params = src.get("params", {})
                    if "size" in params:
                        params["size"] = convert_value(
                            params["size"], "length", from_cfg, to_cfg
                        ).tolist()
                elif kind == "sphere":
                    params = src.get("params", {})
                    if "radius" in params:
                        params["radius"] = float(
                            convert_value(
                                params["radius"], "length", from_cfg, to_cfg
                            )
                        )
                elif kind == "points":
                    points = src.get("points")
                    if isinstance(points, list):
                        for p in points:
                            if "pos" in p:
                                p["pos"] = convert_value(
                                    p["pos"], "length", from_cfg, to_cfg
                                ).tolist()
                            if "mass" in p:
                                p["mass"] = float(
                                    convert_value(
                                        p["mass"], "mass", from_cfg, to_cfg
                                    )
                                )
                if "mass" in src:
                    src["mass"] = float(
                        convert_value(src["mass"], "mass", from_cfg, to_cfg)
                    )

        visuals = rigid.get("visual")
        if isinstance(visuals, list):
            for visual in visuals:
                if isinstance(visual, dict) and "offset_body" in visual:
                    visual["offset_body"] = convert_value(
                        visual["offset_body"], "length", from_cfg, to_cfg
                    ).tolist()

    models = defn.get("models", [])
    for entry in models:
        if "uniform_gravity" in entry:
            g = entry["uniform_gravity"].get("g")
            if g is not None:
                entry["uniform_gravity"]["g"] = convert_value(
                    g, "accel", from_cfg, to_cfg
                )
        elif "nbody_gravity" in entry:
            cfg = entry["nbody_gravity"]
            if "G" in cfg:
                cfg["G"] = float(
                    convert_value(cfg["G"], "G", from_cfg, to_cfg)
                )
        elif "rigid_body_forces" in entry:
            forces = entry["rigid_body_forces"].get("forces", [])
            for force in forces:
                if "point_body" in force:
                    force["point_body"] = convert_value(
                        force["point_body"], "length", from_cfg, to_cfg
                    ).tolist()
                if "force_body" in force:
                    force["force_body"] = convert_value(
                        force["force_body"], "force", from_cfg, to_cfg
                    ).tolist()
                if "force_world" in force:
                    force["force_world"] = convert_value(
                        force["force_world"], "force", from_cfg, to_cfg
                    ).tolist()

    defn["units"] = {"preset": to_cfg.preset, "enabled": to_cfg.enabled}


def _effective_preset(cfg: UnitsConfig) -> UnitPreset:
    return get_preset(cfg.preset if cfg.enabled else "SI")


def _scale_for_kind(kind: str, preset: UnitPreset) -> float:
    if kind == "length":
        return preset.L
    if kind == "mass":
        return preset.M
    if kind == "time":
        return preset.T
    if kind == "velocity":
        return preset.L / preset.T
    if kind == "accel":
        return preset.L / (preset.T**2)
    if kind == "force":
        return preset.M * preset.L / (preset.T**2)
    if kind == "omega":
        return 1.0 / preset.T
    if kind == "inertia":
        return preset.M * (preset.L**2)
    if kind == "torque":
        return preset.M * (preset.L**2) / (preset.T**2)
    if kind == "G":
        return (preset.L**3) / (preset.M * (preset.T**2))
    raise ValueError(f"unsupported unit kind: {kind}")


def _apply_scale(value: Any, scale: float) -> Any:
    if isinstance(value, (list, tuple, np.ndarray)):
        arr = np.asarray(value, dtype=np.float64)
        return (arr * scale).astype(np.float64)
    return float(value) * scale
