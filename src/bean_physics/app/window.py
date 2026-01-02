"""Main window scaffolding for the desktop app."""

from __future__ import annotations

from PySide6 import QtCore, QtGui, QtWidgets

from .viewport import ViewportWidget
from .sim_controller import SimulationController
from .session import ScenarioSession
from .panels.objects_panel import ObjectsPanel
from .panels.objects_utils import (
    ObjectRef,
    add_nbody_gravity,
    add_particle,
    add_uniform_gravity,
    list_forces,
    list_particles,
    remove_force,
    remove_particle,
)
from .inspector import ObjectInspector


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.resize(1200, 800)

        self._controller = SimulationController()
        self._session = ScenarioSession()
        self._running = False
        self._substeps_per_tick = 1
        self._timer = QtCore.QTimer(self)
        self._timer.setInterval(16)
        self._timer.timeout.connect(self._on_tick)

        self._viewport = ViewportWidget(self)
        self.setCentralWidget(self._viewport)

        self._objects_panel = ObjectsPanel(self)
        self._objects_panel.selection_changed.connect(self._on_object_selected)
        self._objects_panel.item_activated.connect(self._on_object_activated)
        self._objects_panel.add_particle_requested.connect(self._on_add_particle)
        self._objects_panel.add_uniform_requested.connect(self._on_add_uniform_gravity)
        self._objects_panel.add_nbody_requested.connect(self._on_add_nbody_gravity)
        self._objects_panel.remove_requested.connect(self._on_remove_selected)

        self._inspector = ObjectInspector(self)
        self._inspector.applied.connect(self._on_inspector_applied)

        self._build_docks()
        self._build_toolbar()
        self._build_menus()
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

        self._trail_toggle = QtWidgets.QToolButton(self)
        self._trail_toggle.setText("Trails")
        self._trail_toggle.setCheckable(True)
        self._trail_toggle.toggled.connect(self._on_trails_toggled)
        self._toolbar.addWidget(self._trail_toggle)

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

        steps_label = QtWidgets.QLabel("Steps/Frame")
        steps_label.setContentsMargins(6, 0, 6, 0)
        self._toolbar.addWidget(steps_label)

        self._steps_spin = QtWidgets.QSpinBox(self)
        self._steps_spin.setRange(1, 200)
        self._steps_spin.setValue(self._substeps_per_tick)
        self._steps_spin.valueChanged.connect(self._on_steps_per_frame_changed)
        self._toolbar.addWidget(self._steps_spin)

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

    def _on_new(self) -> None:
        if not self._confirm_discard_if_dirty():
            return
        self._session.scenario_def = self._session.new_default()
        self._session.scenario_path = None
        self._session.is_dirty = True
        self._controller.load_definition(self._session.scenario_def)
        self._controller.scenario_path = None
        self._running = False
        if not self._timer.isActive():
            self._timer.start()
        self._apply_state_to_viewport()
        self._refresh_objects_panel()
        if self._inspector.isVisible():
            self._inspector.set_target(self._session.scenario_def, None)
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
            self._inspector.set_target(self._session.scenario_def, None)
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
        self._update_action_state()
        self._update_status()

    def _on_steps_per_frame_changed(self, value: int) -> None:
        self._substeps_per_tick = value
        self._update_status()

    def _apply_state_to_viewport(self) -> None:
        self._viewport.set_particles(self._controller.particle_positions())
        self._viewport.set_rigid_bodies(
            self._controller.rigid_body_positions(),
            self._controller.rigid_body_quat(),
        )

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
        self.statusBar().showMessage(msg)

    def _update_window_title(self) -> None:
        if self._session.scenario_def is None:
            self.setWindowTitle("Bean Physics")
            return
        self.setWindowTitle(self._session.window_title())

    def _refresh_objects_panel(self) -> None:
        if self._session.scenario_def is None:
            self._objects_panel.set_items({}, [], [])
            self._objects_panel.select_object(None)
            self._viewport.set_selected_particles([])
            if self._inspector.isVisible():
                self._inspector.set_target(self._session.scenario_def, None)
            return
        particles = list_particles(self._session.scenario_def)
        forces = list_forces(self._session.scenario_def)
        self._objects_panel.set_items(self._session.scenario_def, particles, forces)
        if particles:
            self._objects_panel.select_object(particles[0])
            self._on_object_selected(particles[0])
        elif forces:
            self._objects_panel.select_object(forces[0])
            self._on_object_selected(forces[0])
        else:
            self._objects_panel.select_object(None)
            self._on_object_selected(None)

    def _on_object_selected(self, obj: ObjectRef | None) -> None:
        if obj is None:
            self._viewport.set_selected_particles([])
            self._inspector.set_target(self._session.scenario_def, None)
            return
        if obj.type == "particle":
            self._viewport.set_selected_particles([obj.index])
        else:
            self._viewport.set_selected_particles([])
        if self._inspector.isVisible():
            self._inspector.set_target(self._session.scenario_def, obj)

    def _on_object_activated(self, obj: ObjectRef | None) -> None:
        if obj is None:
            return
        self._inspector.set_target(self._session.scenario_def, obj)
        self._inspector.show()
        self._inspector.raise_()
        self._inspector.activateWindow()

    def _on_add_particle(self) -> None:
        if self._session.scenario_def is None:
            self._session.scenario_def = self._session.new_default()
        index = add_particle(self._session.scenario_def)
        self._session.mark_dirty()
        self._controller.load_definition(self._session.scenario_def)
        self._controller.scenario_path = self._session.scenario_path
        self._running = False
        self._apply_state_to_viewport()
        self._refresh_objects_panel()
        obj = ObjectRef(type="particle", index=index)
        self._objects_panel.select_object(obj)
        self._inspector.set_target(self._session.scenario_def, obj)
        self._update_action_state()
        self._update_status()
        self._update_window_title()

    def _on_add_uniform_gravity(self) -> None:
        if self._session.scenario_def is None:
            self._session.scenario_def = self._session.new_default()
        index = add_uniform_gravity(self._session.scenario_def, [0.0, -9.81, 0.0])
        obj = ObjectRef(type="force", index=index, subtype="uniform_gravity")
        self._after_force_change(obj)

    def _on_add_nbody_gravity(self) -> None:
        if self._session.scenario_def is None:
            self._session.scenario_def = self._session.new_default()
        index = add_nbody_gravity(self._session.scenario_def, 1.0, 0.0, None)
        obj = ObjectRef(type="force", index=index, subtype="nbody_gravity")
        self._after_force_change(obj)

    def _on_remove_selected(self, obj: ObjectRef | None) -> None:
        if obj is None or self._session.scenario_def is None:
            return
        try:
            if obj.type == "particle":
                remove_particle(self._session.scenario_def, obj.index)
            elif obj.type == "force":
                remove_force(self._session.scenario_def, obj.index)
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
            self._inspector.set_target(self._session.scenario_def, None)
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
        self._inspector.set_target(self._session.scenario_def, obj)
        self._update_action_state()
        self._update_status()
        self._update_window_title()

    def _on_frame_all(self) -> None:
        pos = self._controller.particle_positions()
        self._viewport.frame_all(pos)

    def _on_frame_selection(self) -> None:
        obj = self._objects_panel.selected_object()
        if obj is None or obj.type != "particle":
            return
        pos = self._controller.particle_positions()
        self._viewport.frame_selection(pos, [obj.index])

    def _on_follow_toggled(self, enabled: bool) -> None:
        self._viewport.set_follow_selection(enabled)

    def _on_trails_toggled(self, enabled: bool) -> None:
        self._viewport.set_trails_enabled(enabled)

    def _on_trail_length_changed(self, value: int) -> None:
        self._viewport.set_trail_length(value)

    def _on_labels_toggled(self, enabled: bool) -> None:
        self._viewport.set_labels_enabled(enabled)

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
