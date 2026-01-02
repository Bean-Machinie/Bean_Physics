from __future__ import annotations

import pytest

from bean_physics.app.panels.particles_utils import particles_to_rows, rows_to_particles
from bean_physics.app.session import ScenarioSession
from bean_physics.io.scenario import scenario_to_runtime


def test_rows_to_particles_adds_particle() -> None:
    session = ScenarioSession()
    defn = session.new_default()
    rows_to_particles(defn, [[1, 2, 3, 0.1, 0.2, 0.3, 2.5]])

    particles = defn["entities"]["particles"]
    assert particles["pos"] == [[1.0, 2.0, 3.0]]
    assert particles["vel"] == [[0.1, 0.2, 0.3]]
    assert particles["mass"] == [2.5]

    state, model, integrator, dt, steps, _ = scenario_to_runtime(defn)
    assert state.particles is not None
    assert model is not None
    assert integrator is not None
    assert dt > 0.0
    assert steps > 0


def test_invalid_mass_rejected() -> None:
    session = ScenarioSession()
    defn = session.new_default()
    with pytest.raises(ValueError, match="mass must be > 0"):
        rows_to_particles(defn, [[0, 0, 0, 0, 0, 0, 0]])


def test_rows_round_trip() -> None:
    session = ScenarioSession()
    defn = session.new_default()
    rows_to_particles(defn, [[1, 2, 3, 4, 5, 6, 1]])

    rows = particles_to_rows(defn)
    defn2 = session.new_default()
    rows_to_particles(defn2, rows)
    assert defn2["entities"]["particles"] == defn["entities"]["particles"]
