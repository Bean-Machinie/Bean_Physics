"""3D viewport placeholder backed by VisPy."""

from __future__ import annotations

import numpy as np
from PySide6 import QtWidgets
from vispy import app, scene
from vispy.util import keys

from ..core.math.quat import quat_to_rotmat
from .viz_utils import compute_bounds, compute_selection_bounds, rigid_transform_matrix

app.use_app("pyside6")


class SpacePanTurntableCamera(scene.TurntableCamera):
    def __init__(self, *args: object, **kwargs: object) -> None:
        super().__init__(*args, **kwargs)
        self._space_pan = False

    def viewbox_key_event(self, event: object) -> None:
        if getattr(event, "key", None) == keys.SPACE:
            self._space_pan = event.type == "key_press"
            event.handled = True
            return
        super().viewbox_key_event(event)

    def viewbox_mouse_event(self, event: object) -> None:
        if (
            self._space_pan
            and getattr(event, "type", None) in {"mouse_move", "mouse_press", "mouse_release"}
            and 1 in event.buttons
        ):
            mouse_event = event.mouse_event
            prev_mods = mouse_event._modifiers
            if keys.SHIFT not in prev_mods:
                mouse_event._modifiers = tuple(prev_mods) + (keys.SHIFT,)
            try:
                super().viewbox_mouse_event(event)
            finally:
                mouse_event._modifiers = prev_mods
            event.handled = True
            return
        super().viewbox_mouse_event(event)


class ViewportWidget(QtWidgets.QWidget):
    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)

        self._canvas = scene.SceneCanvas(
            keys="interactive",
            bgcolor="#0b0b0e",
            size=(800, 600),
        )
        self._view = self._canvas.central_widget.add_view()
        self._view.camera = SpacePanTurntableCamera(
            fov=45, azimuth=45, elevation=25, distance=6
        )

        self._axis = scene.visuals.XYZAxis(parent=self._view.scene)
        self._grid = scene.visuals.GridLines(parent=self._view.scene, color=(1, 1, 1, 0.08))
        self._particles = scene.visuals.Markers(parent=self._view.scene)
        self._rigid_visuals: list[scene.visuals.Mesh] = []
        self._rigid_shapes: list[dict[str, object]] = []
        self._rigid_points = scene.visuals.Markers(parent=self._view.scene)
        self._selected_particles = scene.visuals.Markers(parent=self._view.scene)
        self._selected_particles.set_gl_state("translucent", depth_test=False)
        self._rigid_points.set_gl_state("translucent", depth_test=False)
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
        self._selected_rigid_index: int | None = None
        self._last_particle_pos = np.zeros((0, 3), dtype=np.float32)
        self._last_rigid_pos = np.zeros((0, 3), dtype=np.float32)
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

    def set_rigid_bodies(
        self,
        pos: np.ndarray,
        quat: np.ndarray,
        shapes: list[dict[str, object]] | None = None,
    ) -> None:
        pos = np.asarray(pos, dtype=np.float32)
        if pos.size == 0:
            pos = np.zeros((0, 3), dtype=np.float32)
        if pos.ndim != 2 or pos.shape[1] != 3:
            raise ValueError("pos must have shape (M, 3)")
        quat = np.asarray(quat, dtype=np.float32)
        if quat.size == 0:
            quat = np.zeros((0, 4), dtype=np.float32)
        if quat.ndim != 2 or quat.shape[1] != 4:
            raise ValueError("quat must have shape (M, 4)")
        self._last_rigid_pos = pos
        shapes = shapes or []
        if len(shapes) < pos.shape[0]:
            shapes = shapes + [{} for _ in range(pos.shape[0] - len(shapes))]
        if _shape_keys(shapes) != _shape_keys(self._rigid_shapes):
            self._sync_rigid_visuals(shapes)
        for i, visual in enumerate(self._rigid_visuals):
            if i >= pos.shape[0]:
                visual.visible = False
                continue
            visual.visible = True
            rot = quat_to_rotmat(quat[i])
            transform = rigid_transform_matrix(rot, pos[i])
            visual.transform.matrix = transform
        self._update_rigid_visual_styles()
        self._update_follow()

    def set_rigid_body_points(self, points_world: np.ndarray) -> None:
        points = np.asarray(points_world, dtype=np.float32)
        if points.size == 0:
            points = np.zeros((0, 3), dtype=np.float32)
        if points.ndim != 2 or points.shape[1] != 3:
            raise ValueError("points_world must have shape (K, 3)")
        self._rigid_points.set_data(
            points,
            face_color=(0.95, 0.95, 0.2, 0.9),
            edge_color=None,
            size=4,
        )

    def set_selected_particles(self, indices: list[int]) -> None:
        self._selected_indices = indices
        self._trail_points = []
        self._trail_counter = 0
        self._update_selected_particles()
        self._update_trail()
        self._update_labels()

    def set_selected_rigid_body(self, index: int | None) -> None:
        self._selected_rigid_index = index
        self._update_rigid_visual_styles()
        self._update_labels()
        self._update_follow()

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
        if self._selected_indices:
            idx = self._selected_indices[0]
            if 0 <= idx < self._last_particle_pos.shape[0]:
                pos = self._last_particle_pos[idx]
                self._label.text = f"Particle {idx + 1}"
                self._label.pos = pos
                self._label.visible = True
                return
        if self._selected_rigid_index is not None:
            idx = self._selected_rigid_index
            if 0 <= idx < self._last_rigid_pos.shape[0]:
                pos = self._last_rigid_pos[idx]
                self._label.text = f"Rigid Body {idx + 1}"
                self._label.pos = pos
                self._label.visible = True
                return
        self._label.visible = False

    def _update_follow(self) -> None:
        if not self._follow_selection:
            return
        if self._selected_indices:
            idx = self._selected_indices[0]
            if 0 <= idx < self._last_particle_pos.shape[0]:
                self._view.camera.center = self._last_particle_pos[idx]
                return
        if self._selected_rigid_index is not None:
            idx = self._selected_rigid_index
            if 0 <= idx < self._last_rigid_pos.shape[0]:
                self._view.camera.center = self._last_rigid_pos[idx]
                return

    def _sync_rigid_visuals(self, shapes: list[dict[str, object]]) -> None:
        for visual in self._rigid_visuals:
            visual.parent = None
        self._rigid_visuals = []
        self._rigid_shapes = []
        for shape in shapes:
            kind = shape.get("kind", "sphere")
            if kind == "box":
                size = np.asarray(shape.get("size", [1.0, 1.0, 1.0]), dtype=np.float64)
                vertices, faces = _box_mesh(size)
            else:
                radius = float(shape.get("radius", 0.5))
                vertices, faces = _sphere_mesh(radius, 12, 24)
            visual = scene.visuals.Mesh(
                vertices=vertices,
                faces=faces,
                color=(1.0, 0.5, 0.2, 0.9),
                parent=self._view.scene,
            )
            visual.transform = scene.transforms.MatrixTransform()
            self._rigid_visuals.append(visual)
            self._rigid_shapes.append(shape)

    def _update_rigid_visual_styles(self) -> None:
        for i, visual in enumerate(self._rigid_visuals):
            if self._selected_rigid_index is not None and i == self._selected_rigid_index:
                visual.color = (1.0, 0.9, 0.2, 0.95)
            else:
                visual.color = (1.0, 0.5, 0.2, 0.85)


def _shape_keys(shapes: list[dict[str, object]]) -> list[tuple[str, tuple[float, ...]]]:
    keys_list: list[tuple[str, tuple[float, ...]]] = []
    for shape in shapes:
        kind = shape.get("kind", "sphere")
        if kind == "box":
            size = tuple(float(v) for v in shape.get("size", [1.0, 1.0, 1.0]))
            keys_list.append(("box", size))
        else:
            radius = float(shape.get("radius", 0.5))
            keys_list.append(("sphere", (radius,)))
    return keys_list


def _box_mesh(size: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    sx, sy, sz = (np.asarray(size, dtype=np.float32) * 0.5).tolist()
    vertices = np.array(
        [
            [-sx, -sy, -sz],
            [sx, -sy, -sz],
            [sx, sy, -sz],
            [-sx, sy, -sz],
            [-sx, -sy, sz],
            [sx, -sy, sz],
            [sx, sy, sz],
            [-sx, sy, sz],
        ],
        dtype=np.float32,
    )
    faces = np.array(
        [
            [0, 1, 2],
            [0, 2, 3],
            [4, 5, 6],
            [4, 6, 7],
            [0, 1, 5],
            [0, 5, 4],
            [1, 2, 6],
            [1, 6, 5],
            [2, 3, 7],
            [2, 7, 6],
            [3, 0, 4],
            [3, 4, 7],
        ],
        dtype=np.uint32,
    )
    return vertices, faces


def _sphere_mesh(
    radius: float, lat_segments: int, lon_segments: int
) -> tuple[np.ndarray, np.ndarray]:
    lat_segments = max(3, int(lat_segments))
    lon_segments = max(6, int(lon_segments))
    vertices = []
    for i in range(lat_segments + 1):
        theta = np.pi * i / lat_segments
        sin_t = np.sin(theta)
        cos_t = np.cos(theta)
        for j in range(lon_segments):
            phi = 2.0 * np.pi * j / lon_segments
            x = radius * sin_t * np.cos(phi)
            y = radius * cos_t
            z = radius * sin_t * np.sin(phi)
            vertices.append([x, y, z])
    vertices = np.asarray(vertices, dtype=np.float32)
    faces = []
    for i in range(lat_segments):
        for j in range(lon_segments):
            next_j = (j + 1) % lon_segments
            a = i * lon_segments + j
            b = i * lon_segments + next_j
            c = (i + 1) * lon_segments + j
            d = (i + 1) * lon_segments + next_j
            faces.append([a, c, b])
            faces.append([b, c, d])
    return vertices, np.asarray(faces, dtype=np.uint32)
