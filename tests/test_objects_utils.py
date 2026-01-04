from __future__ import annotations

import math
from pathlib import Path

import pytest

from bean_physics.app.panels.objects_utils import (
    add_nbody_gravity,
    add_particle,
    add_rigid_body_template,
    add_uniform_gravity,
    apply_nbody_gravity,
    apply_particle_edit,
    apply_rigid_body_edit,
    apply_uniform_gravity,
    list_forces,
    list_particles,
    list_rigid_bodies,
    particle_summary,
    rigid_body_summary,
    remove_force,
    remove_particle,
    remove_rigid_body,
)
from bean_physics.app.session import ScenarioSession
from bean_physics.io.scenario import load_scenario, save_scenario, scenario_to_runtime


def test_add_particle_updates_scenario() -> None:
    session = ScenarioSession()
    defn = session.new_default()
    index = add_particle(defn)
    assert index == 0
    assert list_particles(defn)[0].index == 0
    summary = particle_summary(defn, 0)
    assert summary["mass"] == 1.0


def test_remove_particle_reindexes() -> None:
    session = ScenarioSession()
    defn = session.new_default()
    add_particle(defn)
    add_particle(defn)
    remove_particle(defn, 0)
    particles = defn["entities"]["particles"]
    assert len(particles["mass"]) == 1
    assert list_particles(defn)[0].index == 0


def test_apply_particle_edit_validation() -> None:
    session = ScenarioSession()
    defn = session.new_default()
    add_particle(defn)
    with pytest.raises(ValueError, match="mass must be > 0"):
        apply_particle_edit(defn, 0, [0, 0, 0, 0, 0, 0, 0])
    with pytest.raises(ValueError, match="non-finite value"):
        apply_particle_edit(defn, 0, [0, 0, 0, math.nan, 0, 0, 1])


def test_round_trip_save_load(tmp_path: Path) -> None:
    session = ScenarioSession()
    defn = session.new_default()
    add_particle(defn)
    apply_particle_edit(defn, 0, [1, 2, 3, 4, 5, 6, 2.5])
    path = tmp_path / "scenario.json"
    save_scenario(path, defn)
    loaded = load_scenario(path)
    assert loaded["entities"]["particles"] == defn["entities"]["particles"]


def test_add_forces_updates_models() -> None:
    session = ScenarioSession()
    defn = session.new_default()
    idx_u = add_uniform_gravity(defn, [0, -9.81, 0])
    idx_n = add_nbody_gravity(defn, 1.0, 0.0, None)
    forces = list_forces(defn)
    assert any(obj.index == idx_u for obj in forces)
    assert any(obj.index == idx_n for obj in forces)
    assert defn["models"][idx_u]["uniform_gravity"]["g"] == [0.0, -9.81, 0.0]
    assert defn["models"][idx_n]["nbody_gravity"]["G"] == 1.0


def test_remove_force_reindexes() -> None:
    session = ScenarioSession()
    defn = session.new_default()
    add_uniform_gravity(defn, [0, -9.81, 0])
    add_nbody_gravity(defn, 1.0, 0.0, None)
    remove_force(defn, 0)
    assert "uniform_gravity" not in defn["models"][0]


def test_apply_force_validation() -> None:
    session = ScenarioSession()
    defn = session.new_default()
    idx = add_uniform_gravity(defn, [0, -9.81, 0])
    with pytest.raises(ValueError, match="g must have 3 values"):
        apply_uniform_gravity(defn, idx, [0, 1])

    idx_n = add_nbody_gravity(defn, 1.0, 0.0, None)
    with pytest.raises(ValueError, match="G must be > 0"):
        apply_nbody_gravity(defn, idx_n, 0.0, 0.0, None)
    with pytest.raises(ValueError, match="chunk_size must be >= 1"):
        apply_nbody_gravity(defn, idx_n, 1.0, 0.0, 0)


def test_uniform_gravity_changes_velocity() -> None:
    session = ScenarioSession()
    defn = session.new_default()
    add_particle(defn)
    apply_particle_edit(defn, 0, [0, 0, 0, 0, 0, 0, 1])
    add_uniform_gravity(defn, [0, -9.81, 0])
    state, model, integrator, dt, _, _ = scenario_to_runtime(defn)
    v0 = state.particles.vel[0, 1]
    for _ in range(5):
        integrator.step(state, model, dt)
    assert state.particles.vel[0, 1] < v0


def test_add_remove_rigid_body_template() -> None:
    session = ScenarioSession()
    defn = session.new_default()
    idx = add_rigid_body_template(defn, "box", {"size": [1.0, 2.0, 3.0]})
    bodies = list_rigid_bodies(defn)
    assert bodies[0].index == idx
    summary = rigid_body_summary(defn, idx)
    assert summary["mass"] == 1.0
    apply_rigid_body_edit(
        defn,
        idx,
        "sphere",
        {"radius": 2.0},
        3.0,
        [0.0, 1.0, 2.0],
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0, 0.0],
        [0.1, 0.2, 0.3],
    )
    summary = rigid_body_summary(defn, idx)
    assert summary["mass"] == 3.0
    remove_rigid_body(defn, idx)
    assert list_rigid_bodies(defn) == []


def test_physical_radius_helpers() -> None:
    session = ScenarioSession()
    defn = session.new_default()
    idx = add_particle(defn)
    from bean_physics.app.panels.objects_utils import (
        particle_radius_m,
        set_particle_radius_m,
        rigid_body_radius_m,
        set_rigid_body_radius_m,
    )

    assert particle_radius_m(defn, idx) is None
    set_particle_radius_m(defn, idx, 6_378_137.0)
    assert particle_radius_m(defn, idx) == 6_378_137.0
    set_particle_radius_m(defn, idx, 0.0)
    assert particle_radius_m(defn, idx) is None

    rb_idx = add_rigid_body_template(defn, "sphere")
    assert rigid_body_radius_m(defn, rb_idx) is None
    set_rigid_body_radius_m(defn, rb_idx, 1737_400.0)
    assert rigid_body_radius_m(defn, rb_idx) == 1_737_400.0
