from __future__ import annotations

from pathlib import Path

from bean_physics.app.session import ScenarioSession
from bean_physics.io.scenario import load_scenario, scenario_to_runtime


def test_new_default_is_valid_scenario() -> None:
    session = ScenarioSession()
    defn = session.new_default()
    state, model, integrator, dt, steps, _ = scenario_to_runtime(defn)
    assert state is not None
    assert model is not None
    assert integrator is not None
    assert dt > 0.0
    assert steps > 0


def test_save_as_and_load_round_trip(tmp_path: Path) -> None:
    session = ScenarioSession()
    session.scenario_def = session.new_default()
    path = tmp_path / "scenario.json"
    session.save_as(path)

    loaded = load_scenario(path)
    assert loaded["schema_version"] == session.scenario_def["schema_version"]
    assert loaded["simulation"] == session.scenario_def["simulation"]
    assert loaded.get("models") == session.scenario_def.get("models")


def test_dirty_flag_transitions(tmp_path: Path) -> None:
    session = ScenarioSession()
    session.scenario_def = session.new_default()
    session.mark_dirty()
    assert session.is_dirty

    path = tmp_path / "scenario.json"
    session.save_as(path)
    assert session.is_dirty is False

    session.load(path)
    assert session.is_dirty is False
