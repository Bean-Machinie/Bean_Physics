"""Headless simulation controller for the desktop app."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from ..core.diagnostics.particles import total_energy_gravity
from ..core.forces.nbody_gravity import NBodyGravity
from ..core.state.system import SystemState
from ..io.scenario import load_scenario, scenario_to_runtime


@dataclass(slots=True)
class SimulationRuntime:
    state: SystemState
    model: object
    integrator: object
    dt: float
    steps: int
    aux: dict[str, Any]


class SimulationController:
    def __init__(self) -> None:
        self.scenario_path: Path | None = None
        self.scenario_def: dict[str, Any] | None = None
        self.initial_state: SystemState | None = None
        self.runtime: SimulationRuntime | None = None
        self.current_step = 0

    def load_scenario(self, path: str | Path) -> None:
        scenario_path = Path(path)
        defn = load_scenario(scenario_path)
        self.load_definition(defn)
        self.scenario_path = scenario_path

    def load_definition(self, defn: dict[str, Any]) -> None:
        state, model, integrator, dt, steps, aux = scenario_to_runtime(defn)
        self.scenario_def = defn
        self.runtime = SimulationRuntime(
            state=state,
            model=model,
            integrator=integrator,
            dt=dt,
            steps=steps,
            aux=aux,
        )
        self.initial_state = state.clone()
        self.current_step = 0

    def reset(self) -> bool:
        if self.runtime is None or self.initial_state is None:
            return False
        self.runtime.state = self.initial_state.clone()
        self.current_step = 0
        return True

    def can_step(self) -> bool:
        if self.runtime is None:
            return False
        return self.current_step < self.runtime.steps

    def step_once(self) -> bool:
        if not self.can_step():
            return False
        runtime = self.runtime
        assert runtime is not None
        runtime.integrator.step(runtime.state, runtime.model, runtime.dt)
        self.current_step += 1
        return True

    def diagnostics(self) -> dict[str, float | int]:
        if self.runtime is None:
            return {"step": 0, "time": 0.0}
        runtime = self.runtime
        info: dict[str, float | int] = {
            "step": self.current_step,
            "time": self.current_step * runtime.dt,
        }
        nbody = _find_nbody_model(runtime.model)
        if nbody is not None:
            info["energy"] = total_energy_gravity(
                runtime.state, nbody.G, nbody.softening
            )
        return info

    def particle_positions(self) -> np.ndarray:
        if self.runtime is None or self.runtime.state.particles is None:
            return np.zeros((0, 3), dtype=np.float32)
        return self.runtime.state.particles.pos.astype(np.float32, copy=False)

    def rigid_body_positions(self) -> np.ndarray:
        if self.runtime is None or self.runtime.state.rigid_bodies is None:
            return np.zeros((0, 3), dtype=np.float32)
        return self.runtime.state.rigid_bodies.pos.astype(np.float32, copy=False)

    def rigid_body_quat(self) -> np.ndarray:
        if self.runtime is None or self.runtime.state.rigid_bodies is None:
            return np.zeros((0, 4), dtype=np.float32)
        return self.runtime.state.rigid_bodies.quat.astype(np.float32, copy=False)


def _find_nbody_model(model: object) -> NBodyGravity | None:
    if isinstance(model, NBodyGravity):
        return model
    models = getattr(model, "models", None)
    if models is None:
        return None
    for entry in models:
        if isinstance(entry, NBodyGravity):
            return entry
    return None
