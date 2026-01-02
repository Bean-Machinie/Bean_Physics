from __future__ import annotations

import math
from pathlib import Path

import pytest

from bean_physics.app.panels.objects_utils import (
    add_particle,
    apply_particle_edit,
    list_particles,
    particle_summary,
    remove_particle,
)
from bean_physics.app.session import ScenarioSession
from bean_physics.io.scenario import load_scenario, save_scenario


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
