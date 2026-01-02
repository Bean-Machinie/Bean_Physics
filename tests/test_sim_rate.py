from __future__ import annotations

from bean_physics.app.window import compute_sim_rate


def test_compute_sim_rate() -> None:
    assert compute_sim_rate(0.01, 2, 60.0) == 1.2
