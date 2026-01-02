"""Object details inspector dialog."""

from __future__ import annotations

from typing import Sequence

from PySide6 import QtCore, QtWidgets

from .panels.objects_utils import (
    ObjectRef,
    apply_nbody_gravity,
    apply_particle_edit,
    apply_uniform_gravity,
    force_summary,
    particle_summary,
)


class ObjectInspector(QtWidgets.QDialog):
    applied = QtCore.Signal(object)

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Object Details")
        self.setModal(False)
        self._defn: dict | None = None
        self._obj: ObjectRef | None = None

        self._stack = QtWidgets.QStackedWidget(self)
        self._page_empty = QtWidgets.QWidget()
        self._stack.addWidget(self._page_empty)

        self._particle_page = QtWidgets.QWidget()
        particle_form = QtWidgets.QFormLayout(self._particle_page)
        self._pos = _vector_fields()
        self._vel = _vector_fields()
        self._mass = QtWidgets.QDoubleSpinBox(self)
        self._mass.setRange(1e-12, 1e12)
        self._mass.setDecimals(6)
        self._mass.setValue(1.0)
        particle_form.addRow("Position", _vector_widget(self._pos))
        particle_form.addRow("Velocity", _vector_widget(self._vel))
        particle_form.addRow("Mass", self._mass)
        self._stack.addWidget(self._particle_page)

        self._uniform_page = QtWidgets.QWidget()
        uniform_form = QtWidgets.QFormLayout(self._uniform_page)
        self._g = _vector_fields()
        uniform_form.addRow("g", _vector_widget(self._g))
        self._stack.addWidget(self._uniform_page)

        self._nbody_page = QtWidgets.QWidget()
        nbody_form = QtWidgets.QFormLayout(self._nbody_page)
        self._G = QtWidgets.QDoubleSpinBox(self)
        self._G.setRange(1e-12, 1e12)
        self._G.setDecimals(6)
        self._G.setValue(1.0)
        self._softening = QtWidgets.QDoubleSpinBox(self)
        self._softening.setRange(0.0, 1e12)
        self._softening.setDecimals(6)
        self._softening.setValue(0.0)
        self._chunk = QtWidgets.QLineEdit(self)
        self._chunk.setPlaceholderText("blank = None")
        nbody_form.addRow("G", self._G)
        nbody_form.addRow("Softening", self._softening)
        nbody_form.addRow("Chunk size", self._chunk)
        self._stack.addWidget(self._nbody_page)

        self._btn_apply = QtWidgets.QPushButton("Apply", self)
        self._btn_revert = QtWidgets.QPushButton("Revert", self)
        self._btn_close = QtWidgets.QPushButton("Close", self)
        self._btn_apply.clicked.connect(self._on_apply)
        self._btn_revert.clicked.connect(self._on_revert)
        self._btn_close.clicked.connect(self.close)

        button_row = QtWidgets.QHBoxLayout()
        button_row.addStretch(1)
        button_row.addWidget(self._btn_revert)
        button_row.addWidget(self._btn_apply)
        button_row.addWidget(self._btn_close)

        layout = QtWidgets.QVBoxLayout(self)
        layout.addWidget(self._stack)
        layout.addLayout(button_row)

    def set_target(self, defn: dict | None, obj: ObjectRef | None) -> None:
        self._defn = defn
        self._obj = obj
        self._sync_from_defn()

    def _sync_from_defn(self) -> None:
        if self._defn is None or self._obj is None:
            self._set_enabled(False)
            self._stack.setCurrentWidget(self._page_empty)
            return
        if self._obj.type == "particle":
            summary = particle_summary(self._defn, self._obj.index)
            _set_vector(self._pos, summary["x"], summary["y"], summary["z"])
            _set_vector(self._vel, summary["vx"], summary["vy"], summary["vz"])
            self._mass.setValue(summary["mass"])
            self._stack.setCurrentWidget(self._particle_page)
            self._set_enabled(True)
            return
        if self._obj.type == "force":
            summary = force_summary(self._defn, self._obj.index)
            if summary["type"] == "uniform_gravity":
                g = summary["g"]
                _set_vector(self._g, g[0], g[1], g[2])
                self._stack.setCurrentWidget(self._uniform_page)
                self._set_enabled(True)
                return
            if summary["type"] == "nbody_gravity":
                self._G.setValue(summary["G"])
                self._softening.setValue(summary["softening"])
                chunk = summary["chunk_size"]
                self._chunk.setText("" if chunk is None else str(chunk))
                self._stack.setCurrentWidget(self._nbody_page)
                self._set_enabled(True)
                return
        self._set_enabled(False)
        self._stack.setCurrentWidget(self._page_empty)

    def _set_enabled(self, enabled: bool) -> None:
        widgets = [
            *self._pos,
            *self._vel,
            self._mass,
            *self._g,
            self._G,
            self._softening,
            self._chunk,
        ]
        for field in widgets:
            field.setEnabled(enabled)
        self._btn_apply.setEnabled(enabled)
        self._btn_revert.setEnabled(enabled)

    def _on_apply(self) -> None:
        if self._defn is None or self._obj is None:
            return
        try:
            if self._obj.type == "particle":
                values = [
                    *self._vector_values(self._pos),
                    *self._vector_values(self._vel),
                    self._mass.value(),
                ]
                apply_particle_edit(self._defn, self._obj.index, values)
            elif self._obj.type == "force":
                if self._obj.subtype == "uniform_gravity":
                    g = self._vector_values(self._g)
                    apply_uniform_gravity(self._defn, self._obj.index, g)
                elif self._obj.subtype == "nbody_gravity":
                    apply_nbody_gravity(
                        self._defn,
                        self._obj.index,
                        self._G.value(),
                        self._softening.value(),
                        self._chunk.text().strip(),
                    )
                else:
                    raise ValueError("unsupported force type")
            else:
                return
        except Exception as exc:  # pragma: no cover - Qt error path
            QtWidgets.QMessageBox.critical(self, "Invalid Values", str(exc))
            return
        self.applied.emit(self._obj)

    def _on_revert(self) -> None:
        self._sync_from_defn()

    @staticmethod
    def _vector_values(fields: Sequence[QtWidgets.QDoubleSpinBox]) -> list[float]:
        return [field.value() for field in fields]


def _vector_fields() -> tuple[QtWidgets.QDoubleSpinBox, QtWidgets.QDoubleSpinBox, QtWidgets.QDoubleSpinBox]:
    fields = []
    for _ in range(3):
        spin = QtWidgets.QDoubleSpinBox()
        spin.setRange(-1e9, 1e9)
        spin.setDecimals(6)
        spin.setSingleStep(0.1)
        fields.append(spin)
    return fields[0], fields[1], fields[2]


def _vector_widget(fields: Sequence[QtWidgets.QDoubleSpinBox]) -> QtWidgets.QWidget:
    widget = QtWidgets.QWidget()
    layout = QtWidgets.QHBoxLayout(widget)
    layout.setContentsMargins(0, 0, 0, 0)
    for field in fields:
        layout.addWidget(field)
    return widget


def _set_vector(
    fields: Sequence[QtWidgets.QDoubleSpinBox], x: float, y: float, z: float
) -> None:
    fields[0].setValue(x)
    fields[1].setValue(y)
    fields[2].setValue(z)
