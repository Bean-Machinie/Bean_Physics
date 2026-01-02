from __future__ import annotations

from datetime import datetime
from pathlib import Path

from bean_physics.app.recording_utils import (
    build_metadata,
    make_recording_paths,
    video_filename,
)


def test_make_recording_paths() -> None:
    base = Path("recordings")
    stamp = datetime(2024, 1, 2, 3, 4, 5)
    paths = make_recording_paths(base, Path("scenes/test.json"), stamp)
    assert paths.run_dir.as_posix().endswith("recordings/test/20240102_030405")
    assert paths.frames_dir.name == "frames"


def test_video_filename() -> None:
    assert video_filename() == "recording.mp4"


def test_build_metadata_fields() -> None:
    data = build_metadata(
        scenario_path=Path("demo.json"),
        scenario_name="Demo",
        dt=0.01,
        steps_per_frame=2,
        timer_fps=60.0,
        camera={"center": [0, 0, 0]},
        trails={"enabled": True, "length": 10, "stride": 1},
        labels={"enabled": False},
        follow_enabled=True,
        frames_written=5,
        start_wall_time="2024-01-02T03:04:05",
    )
    assert data["scenario_path"].endswith("demo.json")
    assert data["dt"] == 0.01
    assert data["frames_written"] == 5
