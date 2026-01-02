"""3D viewport placeholder backed by VisPy."""

from __future__ import annotations

import numpy as np
from PySide6 import QtWidgets
from vispy import app, scene

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
        if pos.ndim != 2 or pos.shape[1] != 3:
            raise ValueError("pos must have shape (N, 3)")
        self._particles.set_data(
            pos,
            face_color=(0.2, 0.8, 1.0, 0.9),
            edge_color=None,
            size=6,
        )

    def set_rigid_bodies(self, pos: np.ndarray, quat: np.ndarray) -> None:
        _ = (pos, quat)
