from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from bean_physics.core.diagnostics import linear_momentum
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
