"""Main window scaffolding for the desktop app."""

from __future__ import annotations

from PySide6 import QtCore, QtGui, QtWidgets

from .viewport import ViewportWidget
from .sim_controller import SimulationController


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Bean Physics")
        self.resize(1200, 800)

        self._controller = SimulationController()
        self._running = False
        self._substeps_per_tick = 1
        self._timer = QtCore.QTimer(self)
        self._timer.setInterval(16)
        self._timer.timeout.connect(self._on_tick)

        self._viewport = ViewportWidget(self)
        self.setCentralWidget(self._viewport)

        self._build_docks()
        self._build_toolbar()
        self.statusBar().showMessage("Ready")

    def _build_toolbar(self) -> None:
        toolbar = QtWidgets.QToolBar("Main", self)
        toolbar.setMovable(False)
        toolbar.setIconSize(QtCore.QSize(18, 18))
        self.addToolBar(QtCore.Qt.ToolBarArea.TopToolBarArea, toolbar)

        self._action_load = QtGui.QAction("Load", self)
        self._action_load.triggered.connect(self._on_load)
        toolbar.addAction(self._action_load)

        self._action_save = QtGui.QAction("Save", self)
        self._action_save.setEnabled(False)
        toolbar.addAction(self._action_save)

        toolbar.addSeparator()

        steps_label = QtWidgets.QLabel("Steps/Frame")
        steps_label.setContentsMargins(6, 0, 6, 0)
        toolbar.addWidget(steps_label)

        self._steps_spin = QtWidgets.QSpinBox(self)
        self._steps_spin.setRange(1, 200)
        self._steps_spin.setValue(self._substeps_per_tick)
        self._steps_spin.valueChanged.connect(self._on_steps_per_frame_changed)
        toolbar.addWidget(self._steps_spin)

        toolbar.addSeparator()

        self._action_run = QtGui.QAction("Run", self)
        self._action_run.setEnabled(False)
        self._action_run.triggered.connect(self._on_run)
        toolbar.addAction(self._action_run)

        self._action_pause = QtGui.QAction("Pause", self)
        self._action_pause.setEnabled(False)
        self._action_pause.triggered.connect(self._on_pause)
        toolbar.addAction(self._action_pause)

        self._action_step = QtGui.QAction("Step", self)
        self._action_step.setEnabled(False)
        self._action_step.triggered.connect(self._on_step)
        toolbar.addAction(self._action_step)

        self._action_reset = QtGui.QAction("Reset", self)
        self._action_reset.setEnabled(False)
        self._action_reset.triggered.connect(self._on_reset)
        toolbar.addAction(self._action_reset)

    def _build_docks(self) -> None:
        scenario = QtWidgets.QDockWidget("Scenario", self)
        scenario.setWidget(self._placeholder("Scenario panel (coming soon)"))
        scenario.setAllowedAreas(
            QtCore.Qt.DockWidgetArea.LeftDockWidgetArea
            | QtCore.Qt.DockWidgetArea.RightDockWidgetArea
        )
        self.addDockWidget(QtCore.Qt.DockWidgetArea.LeftDockWidgetArea, scenario)

        inspector = QtWidgets.QDockWidget("Inspector", self)
        inspector.setWidget(self._placeholder("Inspector panel (coming soon)"))
        inspector.setAllowedAreas(
            QtCore.Qt.DockWidgetArea.LeftDockWidgetArea
            | QtCore.Qt.DockWidgetArea.RightDockWidgetArea
        )
        self.addDockWidget(QtCore.Qt.DockWidgetArea.RightDockWidgetArea, inspector)

    @staticmethod
    def _placeholder(text: str) -> QtWidgets.QWidget:
        label = QtWidgets.QLabel(text)
        label.setAlignment(QtCore.Qt.AlignmentFlag.AlignTop)
        label.setWordWrap(True)
        container = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(container)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.addWidget(label)
        layout.addStretch(1)
        return container

    def _on_load(self) -> None:
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            "Load Scenario",
            "",
            "Scenario JSON (*.json)",
        )
        if not path:
            return
        try:
            self._controller.load_scenario(path)
        except Exception as exc:  # pragma: no cover - Qt error path
            QtWidgets.QMessageBox.critical(self, "Load Failed", str(exc))
            return
        if not self._timer.isActive():
            self._timer.start()
        self._running = False
        self._apply_state_to_viewport()
        self._update_action_state()
        self._update_status()

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


def compute_sim_rate(dt: float, steps_per_frame: int, fps: float = 60.0) -> float:
    return dt * steps_per_frame * fps
