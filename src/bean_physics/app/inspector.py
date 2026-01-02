"""Object details inspector dialog."""

from __future__ import annotations

from typing import Sequence

from PySide6 import QtCore, QtWidgets

from .panels.objects_utils import ObjectRef, apply_particle_edit, particle_summary


class ObjectInspector(QtWidgets.QDialog):
    applied = QtCore.Signal(object)

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Object Details")
        self.setModal(False)
        self._defn: dict | None = None
        self._obj: ObjectRef | None = None

        form = QtWidgets.QFormLayout()
        self._pos = _vector_fields()
        self._vel = _vector_fields()
        self._mass = QtWidgets.QDoubleSpinBox(self)
        self._mass.setRange(1e-12, 1e12)
        self._mass.setDecimals(6)
        self._mass.setValue(1.0)

        form.addRow("Position", _vector_widget(self._pos))
        form.addRow("Velocity", _vector_widget(self._vel))
        form.addRow("Mass", self._mass)

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
        layout.addLayout(form)
        layout.addLayout(button_row)

    def set_target(self, defn: dict | None, obj: ObjectRef | None) -> None:
        self._defn = defn
        self._obj = obj
        self._sync_from_defn()

    def _sync_from_defn(self) -> None:
        if self._defn is None or self._obj is None:
            self._set_enabled(False)
            return
        if self._obj.type != "particle":
            self._set_enabled(False)
            return
        summary = particle_summary(self._defn, self._obj.index)
        _set_vector(self._pos, summary["x"], summary["y"], summary["z"])
        _set_vector(self._vel, summary["vx"], summary["vy"], summary["vz"])
        self._mass.setValue(summary["mass"])
        self._set_enabled(True)

    def _set_enabled(self, enabled: bool) -> None:
        for field in (*self._pos, *self._vel, self._mass):
            field.setEnabled(enabled)
        self._btn_apply.setEnabled(enabled)
        self._btn_revert.setEnabled(enabled)

    def _on_apply(self) -> None:
        if self._defn is None or self._obj is None:
            return
        values = [*self._vector_values(self._pos), *self._vector_values(self._vel), self._mass.value()]
        try:
            apply_particle_edit(self._defn, self._obj.index, values)
        except Exception as exc:  # pragma: no cover - Qt error path
            QtWidgets.QMessageBox.critical(self, "Invalid Particle", str(exc))
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
