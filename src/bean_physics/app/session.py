"""Scenario session management (headless)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from ..io.scenario import load_scenario, save_scenario, ScenarioDefinition


@dataclass(slots=True)
class ScenarioSession:
    scenario_path: Optional[Path] = None
    scenario_def: ScenarioDefinition | None = None
    is_dirty: bool = False

    @staticmethod
    def new_default() -> ScenarioDefinition:
        return {
            "schema_version": 1,
            "metadata": {
                "name": "Untitled",
                "description": "Blank scenario.",
            },
            "simulation": {
                "dt": 0.001,
                "steps": 10000,
                "integrator": "velocity_verlet",
            },
            "models": [],
        }

    def load(self, path: str | Path) -> None:
        self.scenario_path = Path(path)
        self.scenario_def = load_scenario(self.scenario_path)
        self.is_dirty = False

    def save(self) -> None:
        if self.scenario_path is None:
            raise ValueError("scenario_path is not set; use save_as()")
        if self.scenario_def is None:
            raise ValueError("scenario_def is not set")
        save_scenario(self.scenario_path, self.scenario_def)
        self.is_dirty = False

    def save_as(self, path: str | Path) -> None:
        if self.scenario_def is None:
            raise ValueError("scenario_def is not set")
        self.scenario_path = Path(path)
        save_scenario(self.scenario_path, self.scenario_def)
        self.is_dirty = False

    def mark_dirty(self) -> None:
        self.is_dirty = True

    def window_title(self, app_name: str = "Bean Physics") -> str:
        name = "Untitled"
        if self.scenario_path is not None:
            name = self.scenario_path.name
        suffix = " *" if self.is_dirty else ""
        return f"{app_name} - {name}{suffix}"
