import numpy as np

from bean_physics.io.scenario import scenario_to_runtime
from bean_physics.io.units import UnitsConfig, convert_value, from_si, label_for, to_si


def test_units_round_trip_scalar() -> None:
    cfg = UnitsConfig(preset="KM", enabled=True)
    kinds = [
        "length",
        "mass",
        "time",
        "velocity",
        "accel",
        "force",
        "omega",
        "inertia",
        "torque",
        "G",
    ]
    for kind in kinds:
        original = 1.2345
        si_val = to_si(original, kind, cfg)
        round_trip = from_si(si_val, kind, cfg)
        assert np.isclose(round_trip, original)


def test_units_round_trip_between_presets() -> None:
    cfg_from = UnitsConfig(preset="KM", enabled=True)
    cfg_to = UnitsConfig(preset="SI", enabled=True)
    value = 2.5
    km_to_si = convert_value(value, "length", cfg_from, cfg_to)
    si_to_km = convert_value(km_to_si, "length", cfg_to, cfg_from)
    assert np.isclose(si_to_km, value)


def test_scenario_load_km_to_si() -> None:
    defn = {
        "schema_version": 1,
        "units": {"preset": "KM", "enabled": True},
        "simulation": {"dt": 1.0, "steps": 1, "integrator": "velocity_verlet"},
        "entities": {
            "particles": {
                "pos": [[1.0, 0.0, 0.0]],
                "vel": [[0.0, 2.0, 0.0]],
                "mass": [2.0],
            }
        },
        "models": [{"uniform_gravity": {"g": [0.0, -0.00981, 0.0]}}],
    }
    state, _, _, dt, _, _ = scenario_to_runtime(defn)
    assert np.isclose(dt, 1.0)
    assert state.particles is not None
    assert np.allclose(state.particles.pos[0], [1000.0, 0.0, 0.0])
    assert np.allclose(state.particles.vel[0], [0.0, 2000.0, 0.0])
    assert np.allclose(state.particles.mass[0], 2.0)


def test_label_for_velocity() -> None:
    cfg = UnitsConfig(preset="KM", enabled=True)
    assert label_for("velocity", cfg) == "km/s"
