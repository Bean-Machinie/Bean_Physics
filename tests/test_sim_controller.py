from __future__ import annotations

from pathlib import Path

import numpy as np

from bean_physics.app.sim_controller import SimulationController


def _scenario_path() -> Path:
    return Path("examples/scenarios/two_body_orbit_v1.json")


def test_load_scenario_builds_runtime() -> None:
    controller = SimulationController()
    controller.load_scenario(_scenario_path())

    assert controller.runtime is not None
    assert controller.runtime.state.particles is not None
    assert controller.runtime.model is not None
    assert controller.runtime.integrator is not None
    assert controller.runtime.dt > 0.0
    assert controller.runtime.steps > 0


def test_step_once_is_deterministic() -> None:
    controller = SimulationController()
    controller.load_scenario(_scenario_path())

    initial = controller.particle_positions().copy()
    assert controller.step_once()
    after_first = controller.particle_positions().copy()
    assert not np.array_equal(after_first, initial)

    controller.reset()
    assert controller.step_once()
    after_second = controller.particle_positions().copy()
    assert np.array_equal(after_first, after_second)


def test_reset_restores_initial_positions() -> None:
    controller = SimulationController()
    controller.load_scenario(_scenario_path())

    initial = controller.particle_positions().copy()
    controller.step_once()
    controller.reset()
    reset_pos = controller.particle_positions().copy()
    assert np.array_equal(initial, reset_pos)
