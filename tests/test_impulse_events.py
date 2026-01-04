from __future__ import annotations

import numpy as np

from bean_physics.core.run import run
from bean_physics.io.scenario import scenario_to_runtime


def _base_defn(dt: float, steps: int) -> dict:
    return {
        "schema_version": 1,
        "simulation": {"dt": dt, "steps": steps, "integrator": "symplectic_euler"},
        "entities": {
            "particles": {
                "pos": [[0.0, 0.0, 0.0]],
                "vel": [[0.0, 0.0, 0.0]],
                "mass": [1.0],
                "ids": ["p0"],
            }
        },
        "models": [],
    }


def test_impulse_event_applies_once() -> None:
    defn = _base_defn(dt=0.5, steps=4)
    defn["impulse_events"] = [
        {"t": 1.0, "target": "p0", "delta_v_world": [2.0, 0.0, 0.0]}
    ]
    state, model, integrator, dt, steps, aux = scenario_to_runtime(defn)
    result = run(
        state,
        model,
        integrator,
        dt,
        steps,
        sample_every=1,
        impulse_events=aux.get("impulse_events", []),
    )
    vel = result.particles_vel[:, 0, 0]
    assert np.allclose(vel[0:2], 0.0)
    assert np.allclose(vel[2:], 2.0)


def test_impulse_event_crossing_logic() -> None:
    defn = _base_defn(dt=0.3, steps=4)
    defn["impulse_events"] = [
        {"t": 1.0, "target": "p0", "delta_v_world": [0.0, 3.0, 0.0]}
    ]
    state, model, integrator, dt, steps, aux = scenario_to_runtime(defn)
    result = run(
        state,
        model,
        integrator,
        dt,
        steps,
        sample_every=1,
        impulse_events=aux.get("impulse_events", []),
    )
    vel = result.particles_vel[:, 0, 1]
    assert np.allclose(vel[0:4], 0.0)
    assert np.allclose(vel[4], 3.0)
