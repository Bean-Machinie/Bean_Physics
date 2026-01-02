"""Helpers for recording output paths and metadata."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any


@dataclass(frozen=True, slots=True)
class RecordingPaths:
    run_dir: Path
    frames_dir: Path


def scenario_stem(path: Path | None) -> str:
    if path is None:
        return "untitled"
    stem = path.stem.strip()
    return stem or "untitled"


def make_recording_paths(
    base_dir: Path, scenario_path: Path | None, timestamp: datetime | None = None
) -> RecordingPaths:
    stamp = (timestamp or datetime.now()).strftime("%Y%m%d_%H%M%S")
    run_dir = base_dir / scenario_stem(scenario_path) / stamp
    return RecordingPaths(run_dir=run_dir, frames_dir=run_dir / "frames")


def frame_filename(frame_index: int) -> str:
    return f"frame_{frame_index:06d}.png"


def video_filename() -> str:
    return "recording.mp4"


def build_metadata(
    *,
    scenario_path: Path | None,
    scenario_name: str | None,
    dt: float,
    steps_per_frame: int,
    timer_fps: float,
    camera: dict[str, Any],
    trails: dict[str, Any],
    labels: dict[str, Any],
    follow_enabled: bool,
    frames_written: int,
    start_wall_time: str,
) -> dict[str, Any]:
    return {
        "scenario_path": str(scenario_path) if scenario_path is not None else None,
        "scenario_name": scenario_name,
        "dt": dt,
        "steps_per_frame": steps_per_frame,
        "timer_fps": timer_fps,
        "camera": camera,
        "follow_enabled": follow_enabled,
        "trails": trails,
        "labels": labels,
        "frames_written": frames_written,
        "start_wall_time": start_wall_time,
    }
