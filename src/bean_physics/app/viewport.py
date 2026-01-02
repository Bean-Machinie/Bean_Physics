"""3D viewport placeholder backed by VisPy."""

from __future__ import annotations

import numpy as np
from PySide6 import QtWidgets
from vispy import app, scene

from .viz_utils import compute_bounds, compute_selection_bounds

app.use_app("pyside6")


class ViewportWidget(QtWidgets.QWidget):
    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)

        self._canvas = scene.SceneCanvas(
            keys="interactive",
            bgcolor="#0b0b0e",
            size=(800, 600),
        )
        self._view = self._canvas.central_widget.add_view()
        self._view.camera = scene.TurntableCamera(
            fov=45, azimuth=45, elevation=25, distance=6
        )

        self._axis = scene.visuals.XYZAxis(parent=self._view.scene)
        self._grid = scene.visuals.GridLines(parent=self._view.scene, color=(1, 1, 1, 0.08))
        self._particles = scene.visuals.Markers(parent=self._view.scene)
        self._rigid_markers = scene.visuals.Markers(parent=self._view.scene)
        self._selected_particles = scene.visuals.Markers(parent=self._view.scene)
        self._selected_particles.set_gl_state("translucent", depth_test=False)
        self._trail = scene.visuals.Line(parent=self._view.scene, color=(0.9, 0.9, 0.2, 0.7))
        self._trail.set_gl_state("translucent", depth_test=False)
        self._label = scene.visuals.Text(
            "",
            color="white",
            font_size=10,
            parent=self._view.scene,
        )
        self._label.visible = False
        self._selected_indices: list[int] = []
        self._last_particle_pos = np.zeros((0, 3), dtype=np.float32)
        self._trail_enabled = False
        self._trail_length = 500
        self._trail_stride = 1
        self._trail_counter = 0
        self._trail_points: list[np.ndarray] = []
        self._follow_selection = False
        self._labels_enabled = False

        self._init_dummy_particles()

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self._canvas.native)

    def _init_dummy_particles(self) -> None:
        rng = np.random.default_rng(7)
        pos = rng.uniform(-1.0, 1.0, size=(100, 3)).astype(np.float32)
        self.set_particles(pos)

    def set_particles(self, pos: np.ndarray) -> None:
        pos = np.asarray(pos, dtype=np.float32)
        if pos.size == 0:
            pos = np.zeros((0, 3), dtype=np.float32)
        if pos.ndim != 2 or pos.shape[1] != 3:
            raise ValueError("pos must have shape (N, 3)")
        self._last_particle_pos = pos
        self._particles.set_data(
            pos,
            face_color=(0.2, 0.8, 1.0, 0.9),
            edge_color=None,
            size=6,
        )
        self._update_selected_particles()
        self._update_trail()
        self._update_labels()
        self._update_follow()

    def set_rigid_bodies(self, pos: np.ndarray, quat: np.ndarray) -> None:
        _ = quat
        pos = np.asarray(pos, dtype=np.float32)
        if pos.size == 0:
            pos = np.zeros((0, 3), dtype=np.float32)
        if pos.ndim != 2 or pos.shape[1] != 3:
            raise ValueError("pos must have shape (M, 3)")
        self._rigid_markers.set_data(
            pos,
            face_color=(1.0, 0.5, 0.2, 0.9),
            edge_color=None,
            size=8,
        )

    def set_selected_particles(self, indices: list[int]) -> None:
        self._selected_indices = indices
        self._trail_points = []
        self._trail_counter = 0
        self._update_selected_particles()
        self._update_trail()
        self._update_labels()

    def _update_selected_particles(self) -> None:
        if not self._selected_indices:
            self._selected_particles.set_data(
                np.zeros((0, 3), dtype=np.float32),
                face_color=(1.0, 1.0, 0.2, 1.0),
                edge_color=None,
                size=10,
            )
            return
        idx = [i for i in self._selected_indices if 0 <= i < self._last_particle_pos.shape[0]]
        if not idx:
            self._selected_particles.set_data(
                np.zeros((0, 3), dtype=np.float32),
                face_color=(1.0, 1.0, 0.2, 1.0),
                edge_color=None,
                size=10,
            )
            return
        pos = self._last_particle_pos[idx]
        self._selected_particles.set_data(
            pos,
            face_color=(1.0, 1.0, 0.2, 1.0),
            edge_color=None,
            size=10,
        )

    def set_follow_selection(self, enabled: bool) -> None:
        self._follow_selection = enabled

    def set_trails_enabled(self, enabled: bool) -> None:
        self._trail_enabled = enabled
        if not enabled:
            self.reset_trails()

    def set_trail_length(self, length: int) -> None:
        self._trail_length = max(0, int(length))
        if self._trail_length == 0:
            self.reset_trails()

    def set_trail_stride(self, stride: int) -> None:
        self._trail_stride = max(1, int(stride))

    def reset_trails(self) -> None:
        self._trail_points = []
        self._trail.set_data(np.zeros((0, 3), dtype=np.float32))

    def set_labels_enabled(self, enabled: bool) -> None:
        self._labels_enabled = enabled
        self._label.visible = enabled
        self._update_labels()

    def capture_frame_rgba(self) -> np.ndarray:
        image = self._canvas.render(alpha=True)
        return np.asarray(image, dtype=np.uint8)

    def camera_state(self) -> dict[str, object]:
        camera = self._view.camera
        if camera is None:
            return {}
        return {
            "center": [float(v) for v in camera.center],
            "distance": float(camera.distance),
            "fov": float(getattr(camera, "fov", 0.0)),
        }

    def trails_state(self) -> dict[str, object]:
        return {
            "enabled": self._trail_enabled,
            "length": self._trail_length,
            "stride": self._trail_stride,
        }

    def labels_state(self) -> dict[str, object]:
        return {
            "enabled": self._labels_enabled,
        }

    def follow_enabled(self) -> bool:
        return self._follow_selection

    def frame_all(self, pos: np.ndarray) -> None:
        center, radius = compute_bounds(np.asarray(pos, dtype=np.float32))
        self._frame(center, radius)

    def frame_selection(self, pos: np.ndarray, indices: list[int]) -> None:
        center, radius = compute_selection_bounds(
            np.asarray(pos, dtype=np.float32), indices
        )
        self._frame(center, radius)

    def _frame(self, center: np.ndarray, radius: float) -> None:
        camera = self._view.camera
        if camera is None:
            return
        camera.center = center
        camera.distance = max(radius * 3.0, 1.0)

    def _update_trail(self) -> None:
        if not self._trail_enabled or self._trail_length <= 0:
            return
        if not self._selected_indices:
            self._trail.set_data(np.zeros((0, 3), dtype=np.float32))
            return
        idx = self._selected_indices[0]
        if idx < 0 or idx >= self._last_particle_pos.shape[0]:
            self._trail.set_data(np.zeros((0, 3), dtype=np.float32))
            return
        self._trail_counter += 1
        if self._trail_counter % self._trail_stride != 0:
            return
        self._trail_points.append(self._last_particle_pos[idx].copy())
        if len(self._trail_points) > self._trail_length:
            self._trail_points.pop(0)
        trail = np.asarray(self._trail_points, dtype=np.float32)
        self._trail.set_data(trail)

    def _update_labels(self) -> None:
        if not self._labels_enabled:
            self._label.visible = False
            return
        if not self._selected_indices:
            self._label.visible = False
            return
        idx = self._selected_indices[0]
        if idx < 0 or idx >= self._last_particle_pos.shape[0]:
            self._label.visible = False
            return
        pos = self._last_particle_pos[idx]
        self._label.text = f"Particle {idx + 1}"
        self._label.pos = pos
        self._label.visible = True

    def _update_follow(self) -> None:
        if not self._follow_selection:
            return
        if not self._selected_indices:
            return
        idx = self._selected_indices[0]
        if idx < 0 or idx >= self._last_particle_pos.shape[0]:
            return
        self._view.camera.center = self._last_particle_pos[idx]
