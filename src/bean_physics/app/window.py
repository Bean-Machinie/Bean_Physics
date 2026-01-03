"""Main window scaffolding for the desktop app."""

from __future__ import annotations

from PySide6 import QtCore, QtGui, QtWidgets

import json
import shutil
import subprocess
from datetime import datetime
from pathlib import Path
from queue import Queue
from threading import Thread

import numpy as np

from .viewport import ViewportWidget
from .sim_controller import SimulationController
from .session import ScenarioSession
from .panels.objects_panel import ObjectsPanel
from .panels.objects_utils import (
    ObjectRef,
    add_nbody_gravity,
    add_particle,
    add_rigid_body_template,
    add_uniform_gravity,
    list_forces,
    list_particles,
    list_rigid_bodies,
    particle_trail_enabled,
    particle_visual,
    remove_force,
    remove_particle,
    remove_rigid_body,
    rigid_body_trail_enabled,
    rigid_body_visual,
    rigid_body_shapes,
    rigid_body_points_body,
)
from .inspector import ObjectInspector
from .recording_utils import build_metadata, make_recording_paths, video_filename
from ..io.units import UnitsConfig, config_from_defn, convert_definition_units, preset_names, to_si
from ..core.math.quat import quat_to_rotmat


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.resize(1200, 800)

        self._controller = SimulationController()
        self._session = ScenarioSession()
        self._units_cfg = UnitsConfig(preset="SI", enabled=True)
        self._units_syncing = False
        self._duration_syncing = False
        self._running = False
        self._substeps_per_tick = 1
        self._timer = QtCore.QTimer(self)
        self._timer.setInterval(16)
        self._timer.timeout.connect(self._on_tick)

        self._viewport = ViewportWidget(self)
        self._viewport.visual_warning.connect(self.statusBar().showMessage)
        self.setCentralWidget(self._viewport)

        self._objects_panel = ObjectsPanel(self)
        self._objects_panel.selection_changed.connect(self._on_object_selected)
        self._objects_panel.item_activated.connect(self._on_object_activated)
        self._objects_panel.add_object_requested.connect(self._on_add_object)
        self._objects_panel.remove_requested.connect(self._on_remove_selected)

        self._inspector = ObjectInspector(self)
        self._inspector.applied.connect(self._on_inspector_applied)

        self._recording_enabled = False
        self._recording_stride = 1
        self._recording_frame = 0
        self._recording_paths = None
        self._recording_start = None
        self._recording_queue: Queue[object | None] | None = None
        self._recording_thread: Thread | None = None
        self._recording_video_path: Path | None = None
        self._recording_base_dir = Path("recordings")

        self._build_docks()
        self._build_toolbar()
        self._build_menus()
        self._sync_units_from_defn()
        self._sync_duration_controls()
        self._update_window_title()
        self.statusBar().showMessage("Ready")

    def _build_toolbar(self) -> None:
        self._toolbar = QtWidgets.QToolBar("Main", self)
        self._toolbar.setMovable(False)
        self._toolbar.setIconSize(QtCore.QSize(18, 18))
        self.addToolBar(QtCore.Qt.ToolBarArea.TopToolBarArea, self._toolbar)

        self._action_frame_all = QtGui.QAction("Frame All", self)
        self._action_frame_all.triggered.connect(self._on_frame_all)
        self._toolbar.addAction(self._action_frame_all)

        self._action_frame_selection = QtGui.QAction("Frame Selection", self)
        self._action_frame_selection.triggered.connect(self._on_frame_selection)
        self._toolbar.addAction(self._action_frame_selection)

        self._follow_toggle = QtWidgets.QToolButton(self)
        self._follow_toggle.setText("Follow Selection")
        self._follow_toggle.setCheckable(True)
        self._follow_toggle.toggled.connect(self._on_follow_toggled)
        self._toolbar.addWidget(self._follow_toggle)

        self._toolbar.addSeparator()

        trail_label = QtWidgets.QLabel("Trail Length")
        trail_label.setContentsMargins(6, 0, 6, 0)
        self._toolbar.addWidget(trail_label)

        self._trail_length = QtWidgets.QSpinBox(self)
        self._trail_length.setRange(0, 5000)
        self._trail_length.setValue(500)
        self._trail_length.valueChanged.connect(self._on_trail_length_changed)
        self._toolbar.addWidget(self._trail_length)

        self._labels_toggle = QtWidgets.QToolButton(self)
        self._labels_toggle.setText("Labels")
        self._labels_toggle.setCheckable(True)
        self._labels_toggle.toggled.connect(self._on_labels_toggled)
        self._toolbar.addWidget(self._labels_toggle)

        self._toolbar.addSeparator()

        self._record_toggle = QtWidgets.QToolButton(self)
        self._record_toggle.setText("Record")
        self._record_toggle.setCheckable(True)
        self._record_toggle.toggled.connect(self._on_record_toggled)
        self._toolbar.addWidget(self._record_toggle)

        self._record_output = QtWidgets.QToolButton(self)
        self._record_output.setText("Output Folder")
        self._record_output.clicked.connect(self._on_record_output)
        self._toolbar.addWidget(self._record_output)

        stride_label = QtWidgets.QLabel("Capture Every")
        stride_label.setContentsMargins(6, 0, 6, 0)
        self._toolbar.addWidget(stride_label)

        self._record_stride_spin = QtWidgets.QSpinBox(self)
        self._record_stride_spin.setRange(1, 1000)
        self._record_stride_spin.setValue(self._recording_stride)
        self._record_stride_spin.valueChanged.connect(self._on_record_stride_changed)
        self._toolbar.addWidget(self._record_stride_spin)

        self._toolbar.addSeparator()

        steps_label = QtWidgets.QLabel("Steps/Frame")
        steps_label.setContentsMargins(6, 0, 6, 0)
        self._toolbar.addWidget(steps_label)

        self._steps_spin = QtWidgets.QSpinBox(self)
        self._steps_spin.setRange(1, 200)
        self._steps_spin.setValue(self._substeps_per_tick)
        self._steps_spin.valueChanged.connect(self._on_steps_per_frame_changed)
        self._toolbar.addWidget(self._steps_spin)

        duration_label = QtWidgets.QLabel("Duration")
        duration_label.setContentsMargins(6, 0, 6, 0)
        self._toolbar.addWidget(duration_label)
        self._duration_spin = QtWidgets.QDoubleSpinBox(self)
        self._duration_spin.setRange(0.0, 1e12)
        self._duration_spin.setDecimals(4)
        self._duration_spin.setSingleStep(1.0)
        self._duration_spin.valueChanged.connect(self._on_duration_changed)
        self._toolbar.addWidget(self._duration_spin)
        self._duration_unit = QtWidgets.QComboBox(self)
        self._duration_unit.addItem("s", 1.0)
        self._duration_unit.addItem("min", 60.0)
        self._duration_unit.addItem("hour", 3600.0)
        self._duration_unit.addItem("day", 86400.0)
        self._duration_unit.addItem("year", 31557600.0)
        self._duration_unit.currentIndexChanged.connect(self._on_duration_unit_changed)
        self._toolbar.addWidget(self._duration_unit)

        self._toolbar.addSeparator()

        self._action_run = QtGui.QAction("Run", self)
        self._action_run.setEnabled(False)
        self._action_run.triggered.connect(self._on_run)
        self._toolbar.addAction(self._action_run)

        self._action_pause = QtGui.QAction("Pause", self)
        self._action_pause.setEnabled(False)
        self._action_pause.triggered.connect(self._on_pause)
        self._toolbar.addAction(self._action_pause)

        self._action_step = QtGui.QAction("Step", self)
        self._action_step.setEnabled(False)
        self._action_step.triggered.connect(self._on_step)
        self._toolbar.addAction(self._action_step)

        self._action_reset = QtGui.QAction("Reset", self)
        self._action_reset.setEnabled(False)
        self._action_reset.triggered.connect(self._on_reset)
        self._toolbar.addAction(self._action_reset)

        self._toolbar.addSeparator()
        units_label = QtWidgets.QLabel("Units")
        units_label.setContentsMargins(6, 0, 6, 0)
        self._toolbar.addWidget(units_label)
        self._units_combo = QtWidgets.QComboBox(self)
        for name in preset_names():
            self._units_combo.addItem(name, name)
        self._units_combo.currentIndexChanged.connect(self._on_units_changed)
        self._toolbar.addWidget(self._units_combo)
        self._units_enabled = QtWidgets.QCheckBox("Enabled", self)
        self._units_enabled.setChecked(True)
        self._units_enabled.toggled.connect(self._on_units_enabled)
        self._toolbar.addWidget(self._units_enabled)
        self._units_reset = QtWidgets.QToolButton(self)
        self._units_reset.setText("Reset SI")
        self._units_reset.clicked.connect(self._on_units_reset)
        self._toolbar.addWidget(self._units_reset)

    def _build_docks(self) -> None:
        objects = QtWidgets.QDockWidget("Objects", self)
        objects.setWidget(self._objects_panel)
        objects.setAllowedAreas(
            QtCore.Qt.DockWidgetArea.LeftDockWidgetArea
            | QtCore.Qt.DockWidgetArea.RightDockWidgetArea
        )
        self.addDockWidget(QtCore.Qt.DockWidgetArea.LeftDockWidgetArea, objects)

    def _build_menus(self) -> None:
        self._action_new = QtGui.QAction("New", self)
        self._action_new.setShortcut(QtGui.QKeySequence.New)
        self._action_new.triggered.connect(self._on_new)

        self._action_load = QtGui.QAction("Load", self)
        self._action_load.setShortcut(QtGui.QKeySequence.Open)
        self._action_load.triggered.connect(self._on_load)

        self._action_save = QtGui.QAction("Save", self)
        self._action_save.setShortcut(QtGui.QKeySequence.Save)
        self._action_save.setEnabled(False)
        self._action_save.triggered.connect(self._on_save)

        self._action_save_as = QtGui.QAction("Save As", self)
        self._action_save_as.setShortcut(QtGui.QKeySequence.SaveAs)
        self._action_save_as.setEnabled(False)
        self._action_save_as.triggered.connect(self._on_save_as)

        self.addAction(self._action_new)
        self.addAction(self._action_load)
        self.addAction(self._action_save)
        self.addAction(self._action_save_as)

        menu_bar = QtWidgets.QMenuBar(self)
        menu_bar.setNativeMenuBar(False)
        file_menu = menu_bar.addMenu("&File")
        file_menu.addAction(self._action_new)
        file_menu.addAction(self._action_load)
        file_menu.addSeparator()
        file_menu.addAction(self._action_save)
        file_menu.addAction(self._action_save_as)

        if hasattr(self, "_toolbar"):
            if self._toolbar.actions():
                self._toolbar.insertWidget(self._toolbar.actions()[0], menu_bar)
            else:
                self._toolbar.addWidget(menu_bar)

    def _sync_units_from_defn(self) -> None:
        if self._session.scenario_def is None:
            cfg = UnitsConfig(preset="SI", enabled=True)
        else:
            cfg = config_from_defn(self._session.scenario_def)
            self._session.scenario_def.setdefault(
                "units", {"preset": cfg.preset, "enabled": cfg.enabled}
            )
        self._units_cfg = cfg
        if not hasattr(self, "_units_combo"):
            return
        self._units_syncing = True
        try:
            idx = self._units_combo.findData(cfg.preset)
            if idx >= 0:
                self._units_combo.setCurrentIndex(idx)
            self._units_enabled.setChecked(cfg.enabled)
        finally:
            self._units_syncing = False
        self._inspector.set_units(cfg)

    def _apply_units_change(self, new_cfg: UnitsConfig) -> None:
        if self._session.scenario_def is None:
            self._units_cfg = new_cfg
            return
        if new_cfg == self._units_cfg:
            return
        convert_definition_units(self._session.scenario_def, self._units_cfg, new_cfg)
        self._units_cfg = new_cfg
        self._session.mark_dirty()
        self._controller.load_definition(self._session.scenario_def)
        self._controller.scenario_path = self._session.scenario_path
        self._apply_state_to_viewport()
        self._refresh_objects_panel()
        self._inspector.set_units(new_cfg)
        self._inspector.refresh()
        self._sync_duration_controls()

    def _on_units_changed(self, *_: object) -> None:
        if self._units_syncing:
            return
        preset = self._units_combo.currentData()
        enabled = self._units_enabled.isChecked()
        if preset is None:
            return
        self._apply_units_change(UnitsConfig(preset=preset, enabled=enabled))

    def _on_units_enabled(self, enabled: bool) -> None:
        if self._units_syncing:
            return
        preset = self._units_combo.currentData()
        if preset is None:
            return
        self._apply_units_change(UnitsConfig(preset=preset, enabled=enabled))

    def _on_units_reset(self) -> None:
        if not hasattr(self, "_units_combo"):
            return
        self._units_syncing = True
        try:
            idx = self._units_combo.findData("SI")
            if idx >= 0:
                self._units_combo.setCurrentIndex(idx)
            self._units_enabled.setChecked(True)
        finally:
            self._units_syncing = False
        self._apply_units_change(UnitsConfig(preset="SI", enabled=True))

    def _on_new(self) -> None:
        if not self._confirm_discard_if_dirty():
            return
        self._session.scenario_def = self._session.new_default()
        self._session.scenario_path = None
        self._session.is_dirty = True
        self._sync_units_from_defn()
        self._sync_duration_controls()
        self._controller.load_definition(self._session.scenario_def)
        self._controller.scenario_path = None
        self._running = False
        if not self._timer.isActive():
            self._timer.start()
        self._apply_state_to_viewport()
        self._refresh_objects_panel()
        if self._inspector.isVisible():
            self._inspector.set_target(
                self._session.scenario_def, None, self._session.scenario_path
            )
        self._update_action_state()
        self._update_status()
        self._update_window_title()

    def _on_load(self) -> None:
        if not self._confirm_discard_if_dirty():
            return
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            "Load Scenario",
            "",
            "Scenario JSON (*.json)",
        )
        if not path:
            return
        try:
            self._session.load(path)
            if self._session.scenario_def is not None:
                self._sync_units_from_defn()
                self._sync_duration_controls()
                self._controller.load_definition(self._session.scenario_def)
                self._controller.scenario_path = self._session.scenario_path
        except Exception as exc:  # pragma: no cover - Qt error path
            QtWidgets.QMessageBox.critical(self, "Load Failed", str(exc))
            return
        if not self._timer.isActive():
            self._timer.start()
        self._running = False
        self._apply_state_to_viewport()
        self._refresh_objects_panel()
        if self._inspector.isVisible():
            self._inspector.set_target(
                self._session.scenario_def, None, self._session.scenario_path
            )
        self._update_action_state()
        self._update_status()
        self._update_window_title()

    def _on_save(self) -> None:
        if self._session.scenario_def is None:
            return
        if self._session.scenario_path is None:
            self._on_save_as()
            return
        try:
            self._session.save()
        except Exception as exc:  # pragma: no cover - Qt error path
            QtWidgets.QMessageBox.critical(self, "Save Failed", str(exc))
            return
        self._update_window_title()

    def _on_save_as(self) -> None:
        if self._session.scenario_def is None:
            return
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self,
            "Save Scenario As",
            "",
            "Scenario JSON (*.json)",
        )
        if not path:
            return
        try:
            self._session.save_as(path)
        except Exception as exc:  # pragma: no cover - Qt error path
            QtWidgets.QMessageBox.critical(self, "Save Failed", str(exc))
            return
        self._update_window_title()

    def _on_run(self) -> None:
        if not self._controller.can_step():
            return
        self._running = True
        self._timer.start()
        self._update_action_state()

    def _on_pause(self) -> None:
        self._running = False
        self._update_action_state()

    def _on_step(self) -> None:
        if self._controller.step_once():
            self._apply_state_to_viewport()
            self._capture_recording_frame()
        self._update_action_state()
        self._update_status()

    def _on_reset(self) -> None:
        if self._controller.reset():
            self._apply_state_to_viewport()
            self._viewport.reset_trails()
        self._running = False
        self._update_action_state()
        self._update_status()

    def _on_tick(self) -> None:
        if self._running:
            stepped = False
            for _ in range(self._substeps_per_tick):
                if not self._controller.step_once():
                    self._running = False
                    break
                stepped = True
            if stepped:
                self._apply_state_to_viewport()
                self._capture_recording_frame()
        self._update_action_state()
        self._update_status()

    def _on_steps_per_frame_changed(self, value: int) -> None:
        self._substeps_per_tick = value
        self._update_status()

    def _duration_unit_seconds(self) -> float:
        if not hasattr(self, "_duration_unit"):
            return 1.0
        value = self._duration_unit.currentData()
        try:
            return float(value)
        except (TypeError, ValueError):
            return 1.0

    def _sync_duration_controls(self) -> None:
        if not hasattr(self, "_duration_spin"):
            return
        if self._session.scenario_def is None:
            self._duration_syncing = True
            try:
                self._duration_spin.setValue(0.0)
                self._duration_spin.setEnabled(False)
                self._duration_unit.setEnabled(False)
            finally:
                self._duration_syncing = False
            return
        sim = self._session.scenario_def.get("simulation", {})
        dt = float(sim.get("dt", 0.0))
        steps = int(sim.get("steps", 0))
        dt_si = float(to_si(dt, "time", self._units_cfg))
        duration_si = dt_si * steps
        unit_seconds = self._duration_unit_seconds()
        self._duration_syncing = True
        try:
            self._duration_spin.setEnabled(True)
            self._duration_unit.setEnabled(True)
            if unit_seconds > 0:
                self._duration_spin.setValue(duration_si / unit_seconds)
            else:
                self._duration_spin.setValue(duration_si)
        finally:
            self._duration_syncing = False

    def _on_duration_changed(self, *_: object) -> None:
        if self._duration_syncing:
            return
        if self._session.scenario_def is None:
            return
        sim = self._session.scenario_def.get("simulation", {})
        dt = float(sim.get("dt", 0.0))
        if dt <= 0.0:
            return
        dt_si = float(to_si(dt, "time", self._units_cfg))
        duration_seconds = float(self._duration_spin.value()) * self._duration_unit_seconds()
        steps = int(round(duration_seconds / dt_si)) if dt_si > 0 else 0
        sim["steps"] = max(0, steps)
        self._session.mark_dirty()
        self._controller.load_definition(self._session.scenario_def)
        self._controller.scenario_path = self._session.scenario_path
        self._running = False
        self._apply_state_to_viewport()
        self._update_action_state()

    def _on_duration_unit_changed(self, *_: object) -> None:
        if self._duration_syncing:
            return
        self._sync_duration_controls()

    def _apply_state_to_viewport(self) -> None:
        self._sync_trail_target()
        self._viewport.set_particles(self._controller.particle_positions())
        self._viewport.set_particle_visuals(
            self._particle_visual_specs(),
            self._controller.particle_positions(),
        )
        self._viewport.set_rigid_bodies(
            self._controller.rigid_body_positions(),
            self._controller.rigid_body_quat(),
            rigid_body_shapes(self._session.scenario_def),
        )
        self._viewport.set_rigid_body_visuals(
            self._rigid_body_visual_specs(),
            self._controller.rigid_body_positions(),
            self._controller.rigid_body_quat(),
        )
        self._update_rigid_body_points()

    def _sync_trail_target(self) -> None:
        if self._session.scenario_def is None:
            self._viewport.set_trail_targets([])
            self._viewport.set_trails_enabled(False)
            return
        targets: list[tuple[str, int]] = []
        for ref in list_particles(self._session.scenario_def):
            if particle_trail_enabled(self._session.scenario_def, ref.index):
                targets.append(("particle", ref.index))
        for ref in list_rigid_bodies(self._session.scenario_def):
            if rigid_body_trail_enabled(self._session.scenario_def, ref.index):
                targets.append(("rigid_body", ref.index))
        self._viewport.set_trail_targets(targets)
        self._viewport.set_trails_enabled(bool(targets))

    def _particle_visual_specs(self) -> list[dict[str, object] | None]:
        if self._session.scenario_def is None:
            return []
        particles = list_particles(self._session.scenario_def)
        visuals: list[dict[str, object] | None] = []
        for obj in particles:
            visual = particle_visual(self._session.scenario_def, obj.index)
            visuals.append(self._resolve_visual_spec(visual))
        return visuals

    def _rigid_body_visual_specs(self) -> list[dict[str, object] | None]:
        if self._session.scenario_def is None:
            return []
        rigid_bodies = list_rigid_bodies(self._session.scenario_def)
        visuals: list[dict[str, object] | None] = []
        for obj in rigid_bodies:
            visual = rigid_body_visual(self._session.scenario_def, obj.index)
            visuals.append(self._resolve_visual_spec(visual))
        return visuals

    @staticmethod
    def _normalize_visual_scale(value: object) -> list[float]:
        if isinstance(value, (int, float)):
            scale = float(value)
            return [scale, scale, scale]
        if isinstance(value, (list, tuple, np.ndarray)) and len(value) > 0:
            if len(value) >= 3:
                return [float(value[0]), float(value[1]), float(value[2])]
            scale = float(value[0])
            return [scale, scale, scale]
        return [1.0, 1.0, 1.0]

    def _resolve_visual_spec(
        self, visual: dict[str, object] | None
    ) -> dict[str, object] | None:
        if visual is None:
            return None
        units_cfg = config_from_defn(self._session.scenario_def or {})
        if visual.get("kind") != "mesh":
            return None
        mesh_path = visual.get("mesh_path")
        if not isinstance(mesh_path, str) or not mesh_path:
            return None
        resolved = Path(mesh_path)
        if not resolved.is_absolute() and self._session.scenario_path is not None:
            resolved = self._session.scenario_path.parent / resolved
        if not resolved.exists():
            self.statusBar().showMessage(f"Mesh not found: {resolved}")
            return {
                "scale": self._normalize_visual_scale(visual.get("scale", 1.0)),
                "offset_body": to_si(
                    visual.get("offset_body", [0.0, 0.0, 0.0]),
                    "length",
                    units_cfg,
                ).tolist(),
                "rotation_body_quat": visual.get(
                    "rotation_body_quat", [1.0, 0.0, 0.0, 0.0]
                ),
                "color_tint": visual.get("color_tint", [1.0, 1.0, 1.0]),
                "fallback": "sphere",
                "fallback_radius": 0.5,
            }
        return {
            "mesh_path": str(resolved),
            "scale": self._normalize_visual_scale(visual.get("scale", 1.0)),
            "offset_body": to_si(
                visual.get("offset_body", [0.0, 0.0, 0.0]),
                "length",
                units_cfg,
            ).tolist(),
            "rotation_body_quat": visual.get(
                "rotation_body_quat", [1.0, 0.0, 0.0, 0.0]
            ),
            "color_tint": visual.get("color_tint", [1.0, 1.0, 1.0]),
            "fallback": "sphere",
            "fallback_radius": 0.5,
        }

    def _update_action_state(self) -> None:
        loaded = self._controller.runtime is not None
        can_step = self._controller.can_step()
        self._action_run.setEnabled(loaded and not self._running and can_step)
        self._action_pause.setEnabled(loaded and self._running)
        self._action_step.setEnabled(loaded and not self._running and can_step)
        self._action_reset.setEnabled(loaded)
        self._action_frame_all.setEnabled(loaded)
        self._action_frame_selection.setEnabled(loaded)
        has_session = self._session.scenario_def is not None
        self._action_save.setEnabled(has_session)
        self._action_save_as.setEnabled(has_session)

    def _update_status(self) -> None:
        info = self._controller.diagnostics()
        if self._controller.runtime is None:
            self.statusBar().showMessage("No scenario loaded")
            return
        runtime = self._controller.runtime
        step = info.get("step", 0)
        time = info.get("time", 0.0)
        steps = runtime.steps
        sim_rate = compute_sim_rate(runtime.dt, self._substeps_per_tick)
        msg = (
            f"step {step}/{steps}  t={time:.4f}  dt={runtime.dt:.4f}  "
            f"steps/frame={self._substeps_per_tick}  sim_rate≈{sim_rate:.4f}"
        )
        energy = info.get("energy")
        if energy is not None:
            msg += f"  E={energy:.6f}"
        if self._recording_enabled and self._recording_paths is not None:
            msg += (
                f"  REC {self._recording_frame}  "
                f"{self._recording_paths.run_dir}"
            )
        self.statusBar().showMessage(msg)

    def _update_window_title(self) -> None:
        if self._session.scenario_def is None:
            self.setWindowTitle("Bean Physics")
            return
        self.setWindowTitle(self._session.window_title())

    def _refresh_objects_panel(self) -> None:
        if self._session.scenario_def is None:
            self._objects_panel.set_items({}, [], [], [])
            self._objects_panel.select_object(None)
            self._viewport.set_selected_particles([])
            self._viewport.set_selected_rigid_body(None)
            self._viewport.set_trail_targets([])
            self._viewport.set_trails_enabled(False)
            if self._inspector.isVisible():
                self._inspector.set_target(
                    self._session.scenario_def, None, self._session.scenario_path
                )
            return
        particles = list_particles(self._session.scenario_def)
        rigid_bodies = list_rigid_bodies(self._session.scenario_def)
        forces = list_forces(self._session.scenario_def)
        self._objects_panel.set_items(
            self._session.scenario_def, particles, rigid_bodies, forces
        )
        if particles:
            self._objects_panel.select_object(particles[0])
            self._on_object_selected(particles[0])
        elif rigid_bodies:
            self._objects_panel.select_object(rigid_bodies[0])
            self._on_object_selected(rigid_bodies[0])
        elif forces:
            self._objects_panel.select_object(forces[0])
            self._on_object_selected(forces[0])
        else:
            self._objects_panel.select_object(None)
            self._on_object_selected(None)

    def _on_object_selected(self, obj: ObjectRef | None) -> None:
        if obj is None:
            self._viewport.set_selected_particles([])
            self._viewport.set_selected_rigid_body(None)
            self._viewport.set_rigid_body_points(np.zeros((0, 3), dtype=np.float32))
            self._sync_trail_target()
            self._inspector.set_target(
                self._session.scenario_def, None, self._session.scenario_path
            )
            return
        if obj.type == "particle":
            self._viewport.set_selected_particles([obj.index])
            self._viewport.set_selected_rigid_body(None)
        elif obj.type == "rigid_body":
            self._viewport.set_selected_particles([])
            self._viewport.set_selected_rigid_body(obj.index)
        else:
            self._viewport.set_selected_particles([])
            self._viewport.set_selected_rigid_body(None)
        self._update_rigid_body_points()
        self._sync_trail_target()
        if self._inspector.isVisible():
            self._inspector.set_target(
                self._session.scenario_def, obj, self._session.scenario_path
            )

    def _on_object_activated(self, obj: ObjectRef | None) -> None:
        if obj is None:
            return
        self._inspector.set_target(
            self._session.scenario_def, obj, self._session.scenario_path
        )
        self._inspector.show()
        self._inspector.raise_()
        self._inspector.activateWindow()

    def _on_add_particle(self) -> None:
        self._add_object("particle", None)

    def _on_add_uniform_gravity(self) -> None:
        self._add_object("force", "uniform_gravity")

    def _on_add_nbody_gravity(self) -> None:
        self._add_object("force", "nbody_gravity")

    def _on_add_object(self, payload: dict[str, object]) -> None:
        obj_type = payload.get("type")
        subtype = payload.get("subtype")
        if not isinstance(obj_type, str):
            return
        if subtype is not None and not isinstance(subtype, str):
            return
        self._add_object(obj_type, subtype)

    def _add_object(self, obj_type: str, subtype: str | None) -> None:
        if self._session.scenario_def is None:
            self._session.scenario_def = self._session.new_default()
        if obj_type == "particle":
            index = add_particle(self._session.scenario_def)
            self._session.mark_dirty()
            self._controller.load_definition(self._session.scenario_def)
            self._controller.scenario_path = self._session.scenario_path
            self._running = False
            self._apply_state_to_viewport()
            self._refresh_objects_panel()
            obj = ObjectRef(type="particle", index=index)
            self._objects_panel.select_object(obj)
            self._inspector.set_target(
                self._session.scenario_def, obj, self._session.scenario_path
            )
            self._update_action_state()
            self._update_status()
            self._update_window_title()
            return
        if obj_type == "force" and subtype == "uniform_gravity":
            index = add_uniform_gravity(self._session.scenario_def, [0.0, -9.81, 0.0])
            obj = ObjectRef(type="force", index=index, subtype="uniform_gravity")
            self._after_force_change(obj)
            return
        if obj_type == "force" and subtype == "nbody_gravity":
            index = add_nbody_gravity(self._session.scenario_def, 1.0, 0.0, None)
            obj = ObjectRef(type="force", index=index, subtype="nbody_gravity")
            self._after_force_change(obj)
            return
        if obj_type == "rigid_body" and subtype in {"box", "sphere", "points"}:
            index = add_rigid_body_template(self._session.scenario_def, subtype)
            self._session.mark_dirty()
            self._controller.load_definition(self._session.scenario_def)
            self._controller.scenario_path = self._session.scenario_path
            self._running = False
            self._apply_state_to_viewport()
            self._refresh_objects_panel()
            obj = ObjectRef(type="rigid_body", index=index, subtype=subtype)
            self._objects_panel.select_object(obj)
            self._inspector.set_target(
                self._session.scenario_def, obj, self._session.scenario_path
            )
            self._update_action_state()
            self._update_status()
            self._update_window_title()

    def _on_remove_selected(self, obj: ObjectRef | None) -> None:
        if obj is None or self._session.scenario_def is None:
            return
        try:
            if obj.type == "particle":
                remove_particle(self._session.scenario_def, obj.index)
            elif obj.type == "force":
                remove_force(self._session.scenario_def, obj.index)
            elif obj.type == "rigid_body":
                remove_rigid_body(self._session.scenario_def, obj.index)
            else:
                return
        except Exception as exc:  # pragma: no cover - Qt error path
            QtWidgets.QMessageBox.critical(self, "Remove Failed", str(exc))
            return
        self._session.mark_dirty()
        self._controller.load_definition(self._session.scenario_def)
        self._controller.scenario_path = self._session.scenario_path
        self._running = False
        self._apply_state_to_viewport()
        self._refresh_objects_panel()
        if self._inspector.isVisible():
            self._inspector.set_target(
                self._session.scenario_def, None, self._session.scenario_path
            )
        self._update_action_state()
        self._update_status()
        self._update_window_title()

    def _on_inspector_applied(self, obj: ObjectRef) -> None:
        if self._session.scenario_def is None:
            return
        self._session.mark_dirty()
        self._controller.load_definition(self._session.scenario_def)
        self._controller.scenario_path = self._session.scenario_path
        self._running = False
        self._apply_state_to_viewport()
        self._refresh_objects_panel()
        self._objects_panel.select_object(obj)
        self._sync_trail_target()
        self._update_action_state()
        self._update_status()
        self._update_window_title()

    def _after_force_change(self, obj: ObjectRef) -> None:
        self._session.mark_dirty()
        self._controller.load_definition(self._session.scenario_def)
        self._controller.scenario_path = self._session.scenario_path
        self._running = False
        self._apply_state_to_viewport()
        self._refresh_objects_panel()
        self._objects_panel.select_object(obj)
        self._inspector.set_target(
            self._session.scenario_def, obj, self._session.scenario_path
        )
        self._update_action_state()
        self._update_status()
        self._update_window_title()

    def _update_rigid_body_points(self) -> None:
        obj = self._objects_panel.selected_object()
        if (
            obj is None
            or obj.type != "rigid_body"
            or self._session.scenario_def is None
            or self._controller.runtime is None
            or self._controller.runtime.state.rigid_bodies is None
        ):
            self._viewport.set_rigid_body_points(np.zeros((0, 3), dtype=np.float32))
            return
        points_body = rigid_body_points_body(self._session.scenario_def, obj.index)
        if points_body.size == 0:
            self._viewport.set_rigid_body_points(np.zeros((0, 3), dtype=np.float32))
            return
        pos = self._controller.rigid_body_positions()
        quat = self._controller.rigid_body_quat()
        if obj.index >= pos.shape[0]:
            self._viewport.set_rigid_body_points(np.zeros((0, 3), dtype=np.float32))
            return
        rot = quat_to_rotmat(quat[obj.index].astype(np.float64, copy=False))
        points_world = points_body @ rot.T + pos[obj.index]
        self._viewport.set_rigid_body_points(points_world.astype(np.float32, copy=False))

    def _on_frame_all(self) -> None:
        particles = self._controller.particle_positions()
        rigids = self._controller.rigid_body_positions()
        pos = (
            particles
            if rigids.size == 0
            else np.vstack([particles, rigids])
        )
        self._viewport.frame_all(pos)

    def _on_frame_selection(self) -> None:
        obj = self._objects_panel.selected_object()
        if obj is None:
            return
        if obj.type == "particle":
            pos = self._controller.particle_positions()
            self._viewport.frame_selection(pos, [obj.index])
        elif obj.type == "rigid_body":
            pos = self._controller.rigid_body_positions()
            self._viewport.frame_selection(pos, [obj.index])

    def _on_follow_toggled(self, enabled: bool) -> None:
        self._viewport.set_follow_selection(enabled)

    def _on_trail_length_changed(self, value: int) -> None:
        self._viewport.set_trail_length(value)

    def _on_labels_toggled(self, enabled: bool) -> None:
        self._viewport.set_labels_enabled(enabled)

    def _on_record_toggled(self, enabled: bool) -> None:
        if enabled:
            if self._controller.runtime is None:
                self._record_toggle.setChecked(False)
                return
            try:
                self._start_recording()
            except Exception as exc:  # pragma: no cover - Qt error path
                QtWidgets.QMessageBox.critical(self, "Recording Failed", str(exc))
                self._record_toggle.setChecked(False)
        else:
            self._stop_recording()

    def _on_record_output(self) -> None:
        path = QtWidgets.QFileDialog.getExistingDirectory(
            self, "Select Recording Output Folder"
        )
        if not path:
            return
        self._recording_base_dir = Path(path)

    def _on_record_stride_changed(self, value: int) -> None:
        self._recording_stride = value

    def _start_recording(self) -> None:
        if shutil.which("ffmpeg") is None:
            raise RuntimeError(
                "Recording requires ffmpeg on PATH. Install ffmpeg and retry."
            )
        paths = make_recording_paths(
            self._recording_base_dir,
            self._session.scenario_path,
        )
        paths.frames_dir.mkdir(parents=True, exist_ok=True)
        self._recording_paths = paths
        self._recording_frame = 0
        self._recording_start = datetime.now().isoformat()
        self._recording_queue = Queue()
        self._recording_video_path = paths.run_dir / video_filename()
        self._recording_thread = Thread(
            target=_recording_worker,
            args=(self._recording_queue, self._recording_video_path),
            daemon=True,
        )
        self._recording_thread.start()
        self._recording_enabled = True

    def _stop_recording(self) -> None:
        if not self._recording_enabled:
            return
        if self._recording_queue is not None:
            self._recording_queue.put(None)
        if self._recording_thread is not None:
            self._recording_thread.join(timeout=5.0)
        self._write_recording_metadata()
        self._recording_enabled = False
        self._recording_queue = None
        self._recording_thread = None
        self._recording_video_path = None

    def _capture_recording_frame(self) -> None:
        if not self._recording_enabled or self._recording_paths is None:
            return
        if self._recording_frame % self._recording_stride != 0:
            self._recording_frame += 1
            return
        frame = self._viewport.capture_frame_rgba()
        if self._recording_queue is not None:
            self._recording_queue.put(frame)
        self._recording_frame += 1

    def _write_recording_metadata(self) -> None:
        if self._recording_paths is None or self._recording_start is None:
            return
        runtime = self._controller.runtime
        if runtime is None:
            return
        scenario_name = None
        if self._session.scenario_def is not None:
            meta = self._session.scenario_def.get("metadata", {})
            scenario_name = meta.get("name")
        data = build_metadata(
            scenario_path=self._session.scenario_path,
            scenario_name=scenario_name,
            dt=runtime.dt,
            steps_per_frame=self._substeps_per_tick,
            timer_fps=60.0,
            camera=self._viewport.camera_state(),
            trails=self._viewport.trails_state(),
            labels=self._viewport.labels_state(),
            follow_enabled=self._viewport.follow_enabled(),
            frames_written=self._recording_frame,
            start_wall_time=self._recording_start,
        )
        if self._recording_video_path is not None:
            data["video_path"] = str(self._recording_video_path)
        meta_path = self._recording_paths.run_dir / "meta.json"
        meta_path.write_text(json.dumps(data, indent=2), encoding="utf-8")

    def _confirm_discard_if_dirty(self) -> bool:
        if not self._session.is_dirty:
            return True
        box = QtWidgets.QMessageBox(self)
        box.setIcon(QtWidgets.QMessageBox.Icon.Warning)
        box.setWindowTitle("Unsaved Changes")
        box.setText("You have unsaved changes. Save before continuing?")
        box.setStandardButtons(
            QtWidgets.QMessageBox.StandardButton.Save
            | QtWidgets.QMessageBox.StandardButton.Discard
            | QtWidgets.QMessageBox.StandardButton.Cancel
        )
        choice = box.exec()
        if choice == QtWidgets.QMessageBox.StandardButton.Cancel:
            return False
        if choice == QtWidgets.QMessageBox.StandardButton.Discard:
            return True
        if choice == QtWidgets.QMessageBox.StandardButton.Save:
            if self._session.scenario_path is None:
                self._on_save_as()
            else:
                self._on_save()
            return not self._session.is_dirty
        return False


def compute_sim_rate(dt: float, steps_per_frame: int, fps: float = 60.0) -> float:
    return dt * steps_per_frame * fps


def _recording_worker(queue: Queue[object | None], video_path: Path) -> None:
    proc: subprocess.Popen[bytes] | None = None
    while True:
        item = queue.get()
        if item is None:
            break
        frame = item
        if proc is None:
            height, width = frame.shape[0], frame.shape[1]
            cmd = [
                "ffmpeg",
                "-y",
                "-f",
                "rawvideo",
                "-vcodec",
                "rawvideo",
                "-pix_fmt",
                "rgba",
                "-s",
                f"{width}x{height}",
                "-r",
                "60",
                "-i",
                "-",
                "-an",
                "-c:v",
                "libx264",
                "-preset",
                "medium",
                "-crf",
                "18",
                "-pix_fmt",
                "yuv420p",
                str(video_path),
            ]
            proc = subprocess.Popen(
                cmd, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
            )
        if proc is not None and proc.stdin is not None:
            proc.stdin.write(frame.tobytes())
    if proc is not None and proc.stdin is not None:
        proc.stdin.close()
        proc.wait(timeout=10.0)
