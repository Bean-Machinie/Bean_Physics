from __future__ import annotations

from dataclasses import replace

import numpy as np

from bean_physics.analysis.hohmann_planner import (
    G0,
    HohmannPlanInputs,
    apply_hohmann_plan,
    compute_hohmann,
    prograde_delta_v,
    rocket_equation_propellant,
)
from bean_physics.io.units import UnitsConfig


def test_hohmann_numbers_leo_to_geo() -> None:
    mu = 6.67430e-11 * 5.97219e24
    r1 = 6_378_137.0 + 300_000.0
    r2 = 42_164_000.0
    res = compute_hohmann(mu, r1, r2)
    assert np.isclose(res.v1, 7725.760232077137, rtol=1e-9)
    assert np.isclose(res.v_p, 10151.490141023442, rtol=1e-9)
    assert np.isclose(res.v_a, 1607.8418061830916, rtol=1e-9)
    assert np.isclose(res.v2, 3074.6662841276843, rtol=1e-9)
    assert np.isclose(res.dv1, 2425.729908946305, rtol=1e-9)
    assert np.isclose(res.dv2, 1466.8244779445927, rtol=1e-9)
    assert np.isclose(res.t_transfer, 18990.13173812482, rtol=1e-9)


def test_rocket_equation_propellant() -> None:
    delta_v = 1000.0
    isp = 300.0
    m_dry = 1000.0
    prop_required, mass_ratio, ve = rocket_equation_propellant(m_dry, delta_v, isp)
    expected_ve = isp * G0
    expected_mass_ratio = np.exp(delta_v / expected_ve)
    expected_prop = m_dry * (expected_mass_ratio - 1.0)
    assert np.isclose(ve, expected_ve)
    assert np.isclose(mass_ratio, expected_mass_ratio)
    assert np.isclose(prop_required, expected_prop)


def test_prograde_delta_v_aligns_with_velocity() -> None:
    velocity = np.array([3.0, 4.0, 0.0], dtype=np.float64)
    dv = prograde_delta_v(velocity, 10.0)
    assert np.isclose(np.linalg.norm(dv), 10.0)
    assert np.isclose(np.dot(dv, velocity) / np.linalg.norm(velocity), 10.0)


def test_apply_hohmann_plan_updates_impulses_and_meta() -> None:
    defn = {"metadata": {}}
    mu = 6.67430e-11 * 5.97219e24
    plan = HohmannPlanInputs(
        central_body_id="earth",
        spacecraft_id="sc",
        r1=6_678_137.0,
        r2=42_164_000.0,
        body_radius_m=6_378_137.0,
        r1_mode="Altitude above body radius",
        r2_mode="Absolute radius",
        r1_altitude_m=300_000.0,
        r2_altitude_m=35_786_000.0,
        coast_time_s=0.0,
        burn2_mode="Auto at apoapsis",
        burn2_time_s=0.0,
        dry_mass_kg=1000.0,
        prop_mass_kg=0.0,
        isp_s=300.0,
        thrust_n=0.0,
    )
    units_cfg = UnitsConfig(preset="SI", enabled=True)
    impulses, meta = apply_hohmann_plan(
        defn, mu, plan, np.array([0.0, 1.0, 0.0]), np.array([0.0, 1.0, 0.0]), units_cfg
    )
    assert meta["mission_analysis"]["kind"] == "hohmann"
    assert len(impulses) == 2

    plan2 = replace(plan, coast_time_s=600.0)
    impulses2, _ = apply_hohmann_plan(
        defn, mu, plan2, np.array([0.0, 1.0, 0.0]), np.array([0.0, 1.0, 0.0]), units_cfg
    )
    assert impulses2[0]["t"] != impulses[0]["t"]

    plan3 = replace(plan, r2=50_000_000.0)
    impulses3, _ = apply_hohmann_plan(
        defn, mu, plan3, np.array([0.0, 1.0, 0.0]), np.array([0.0, 1.0, 0.0]), units_cfg
    )
    dv1_a = np.linalg.norm(impulses[0]["delta_v_world"])
    dv1_b = np.linalg.norm(impulses3[0]["delta_v_world"])
    assert not np.isclose(dv1_a, dv1_b)
