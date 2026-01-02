"""Main window scaffolding for the desktop app."""

from __future__ import annotations

from PySide6 import QtCore, QtGui, QtWidgets

from .viewport import ViewportWidget


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Bean Physics")
        self.resize(1200, 800)

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

        for label in ("Load", "Save", "Run", "Pause", "Step", "Reset"):
            action = QtGui.QAction(label, self)
            action.setEnabled(False)
            toolbar.addAction(action)

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
