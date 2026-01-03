from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from bean_physics.core.diagnostics import linear_momentum
from bean_physics.core.forces import RigidBodyForces
from bean_physics.core.math.quat import quat_to_rotmat
from bean_physics.core.run import run
from bean_physics.io import load_scenario, save_scenario, scenario_to_runtime


def _scenario_equal(a: dict, b: dict) -> bool:
    if type(a) is not type(b):
        return False
    if isinstance(a, dict):
        if set(a.keys()) != set(b.keys()):
            return False
        return all(_scenario_equal(a[k], b[k]) for k in a)
    if isinstance(a, list):
        if len(a) != len(b):
            return False
        if len(a) == 0:
            return True
        if isinstance(a[0], (int, float)):
            return np.allclose(a, b)
        return all(_scenario_equal(x, y) for x, y in zip(a, b))
    return a == b


def test_round_trip(tmp_path: Path) -> None:
    src = Path("examples/scenarios/two_body_orbit_v1.json")
    defn = load_scenario(src)
    out = tmp_path / "roundtrip.json"
    save_scenario(out, defn)
    defn2 = load_scenario(out)
    assert _scenario_equal(defn, defn2)


def test_round_trip_rigid_body(tmp_path: Path) -> None:
    src = Path("examples/scenarios/rigid_body_spin_box_v1.json")
    defn = load_scenario(src)
    out = tmp_path / "roundtrip_rigid.json"
    save_scenario(out, defn)
    defn2 = load_scenario(out)
    assert _scenario_equal(defn, defn2)


def test_round_trip_rigid_body_points(tmp_path: Path) -> None:
    defn = {
        "schema_version": 1,
        "simulation": {"dt": 0.01, "steps": 10, "integrator": "velocity_verlet"},
        "entities": {
            "rigid_bodies": {
                "pos": [[0.0, 0.0, 0.0]],
                "vel": [[0.0, 0.0, 0.0]],
                "quat": [[1.0, 0.0, 0.0, 0.0]],
                "omega_body": [[0.0, 0.0, 0.0]],
                "mass": [3.0],
                "mass_distribution": {
                    "points_body": [[-1.0, 0.0, 0.0], [2.0, 0.0, 0.0]],
                    "point_masses": [1.0, 2.0],
                },
                "source": [
                    {
                        "kind": "points",
                        "points": [
                            {"mass": 1.0, "pos": [-1.0, 0.0, 0.0]},
                            {"mass": 2.0, "pos": [2.0, 0.0, 0.0]},
                        ],
                        "mass": 3.0,
                    }
                ],
            }
        },
        "models": [],
    }
    out = tmp_path / "roundtrip_points.json"
    save_scenario(out, defn)
    defn2 = load_scenario(out)
    assert _scenario_equal(defn, defn2)


def test_force_world_migrates_to_body() -> None:
    angle = np.pi / 2.0
    q = [np.cos(angle / 2.0), 0.0, 0.0, np.sin(angle / 2.0)]
    defn = {
        "schema_version": 1,
        "simulation": {"dt": 0.01, "steps": 1, "integrator": "velocity_verlet"},
        "entities": {
            "rigid_bodies": {
                "pos": [[0.0, 0.0, 0.0]],
                "vel": [[0.0, 0.0, 0.0]],
                "quat": [q],
                "omega_body": [[0.0, 0.0, 0.0]],
                "mass": [1.0],
                "mass_distribution": {
                    "points_body": [[0.0, 0.0, 0.0]],
                    "point_masses": [1.0],
                    "inertia_body": [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
                },
            }
        },
        "models": [
            {
                "rigid_body_forces": {
                    "forces": [
                        {
                            "body_index": 0,
                            "point_body": [1.0, 0.0, 0.0],
                            "force_world": [0.0, 1.0, 0.0],
                        }
                    ]
                }
            }
        ],
    }
    state, model, _, _, _, _ = scenario_to_runtime(defn)
    rb_models = [m for m in getattr(model, "models", []) if isinstance(m, RigidBodyForces)]
    assert rb_models
    rb_model = rb_models[0]
    rot = quat_to_rotmat(np.asarray(state.rigid_bodies.quat))
    expected = rot[0].T @ np.asarray([0.0, 1.0, 0.0])
    assert np.allclose(rb_model.forces_body[0], expected)


def test_visual_round_trip(tmp_path: Path) -> None:
    defn = {
        "schema_version": 1,
        "simulation": {"dt": 0.01, "steps": 10, "integrator": "velocity_verlet"},
        "entities": {
            "particles": {
                "pos": [[0.0, 0.0, 0.0]],
                "vel": [[0.0, 0.0, 0.0]],
                "mass": [1.0],
                "visual": [
                    {
                        "kind": "mesh",
                        "mesh_path": "models/planet.glb",
                        "scale": [1.0, 2.0, 3.0],
                        "offset_body": [0.0, 0.5, 0.0],
                        "rotation_body_quat": [1.0, 0.0, 0.0, 0.0],
                        "color_tint": [0.2, 0.4, 0.6],
                    }
                ],
            }
        },
        "models": [],
    }
    out = tmp_path / "visual.json"
    save_scenario(out, defn)
    loaded = load_scenario(out)
    assert _scenario_equal(defn, loaded)


def test_determinism_two_body() -> None:
    defn = load_scenario("examples/scenarios/two_body_orbit_v1.json")
    state1, model1, integrator1, dt1, steps1, _ = scenario_to_runtime(defn)
    state2, model2, integrator2, dt2, steps2, _ = scenario_to_runtime(defn)

    res1 = run(state1, model1, integrator1, dt1, steps1, sample_every=10)
    res2 = run(state2, model2, integrator2, dt2, steps2, sample_every=10)

    assert np.array_equal(res1.particles_pos, res2.particles_pos)
    assert np.array_equal(res1.particles_vel, res2.particles_vel)


def test_validation_missing_fields(tmp_path: Path) -> None:
    bad = {"schema_version": 1, "simulation": {"dt": 0.01, "steps": 10}}
    path = tmp_path / "bad.json"
    path.write_text(json.dumps(bad), encoding="utf-8")
    try:
        load_scenario(path)
        assert False, "expected ValueError"
    except ValueError as exc:
        assert "simulation.integrator" in str(exc)


def test_integration_two_body_no_nans() -> None:
    defn = load_scenario("examples/scenarios/two_body_orbit_v1.json")
    state, model, integrator, dt, steps, _ = scenario_to_runtime(defn)
    res = run(state, model, integrator, dt, steps, sample_every=100)
    assert not np.isnan(res.particles_pos).any()
    p = linear_momentum(res.final_state)
    assert np.linalg.norm(p) < 1e-2
