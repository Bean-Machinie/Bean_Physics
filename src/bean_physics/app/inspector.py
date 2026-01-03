"""Object details inspector dialog."""

from __future__ import annotations

from typing import Sequence

import numpy as np
from PySide6 import QtCore, QtWidgets

from .panels.objects_utils import (
    ObjectRef,
    apply_nbody_gravity,
    apply_particle_edit,
    apply_rigid_body_edit,
    apply_uniform_gravity,
    force_summary,
    particle_summary,
    rigid_body_summary,
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

        self._rigid_page = QtWidgets.QWidget()
        rigid_form = QtWidgets.QFormLayout(self._rigid_page)
        self._rb_kind = QtWidgets.QComboBox(self)
        self._rb_kind.addItem("Box", "box")
        self._rb_kind.addItem("Sphere", "sphere")
        self._rb_kind.currentIndexChanged.connect(self._on_rigid_kind_changed)
        self._rb_size = _vector_fields(min_value=1e-6)
        self._rb_radius = QtWidgets.QDoubleSpinBox(self)
        self._rb_radius.setRange(1e-6, 1e6)
        self._rb_radius.setDecimals(6)
        self._rb_radius.setValue(0.5)
        self._rb_mass = QtWidgets.QDoubleSpinBox(self)
        self._rb_mass.setRange(1e-12, 1e12)
        self._rb_mass.setDecimals(6)
        self._rb_mass.setValue(1.0)
        self._rb_pos = _vector_fields()
        self._rb_vel = _vector_fields()
        self._rb_quat = _quat_fields()
        self._rb_omega = _vector_fields()
        self._rb_inertia = _vector_fields(read_only=True)
        rigid_form.addRow("Template", self._rb_kind)
        rigid_form.addRow("Size", _vector_widget(self._rb_size))
        rigid_form.addRow("Radius", self._rb_radius)
        rigid_form.addRow("Mass", self._rb_mass)
        rigid_form.addRow("Position", _vector_widget(self._rb_pos))
        rigid_form.addRow("Velocity", _vector_widget(self._rb_vel))
        rigid_form.addRow("Quat (w,x,y,z)", _vector_widget(self._rb_quat))
        rigid_form.addRow("Omega body", _vector_widget(self._rb_omega))
        rigid_form.addRow("Inertia diag", _vector_widget(self._rb_inertia))
        self._stack.addWidget(self._rigid_page)

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
        if self._obj.type == "rigid_body":
            summary = rigid_body_summary(self._defn, self._obj.index)
            source = summary["source"] or {}
            kind = source.get("kind", "box")
            params = source.get("params", {})
            self._rb_kind.setCurrentIndex(
                0 if kind == "box" else 1
            )
            size = params.get("size", [1.0, 1.0, 1.0])
            _set_vector(self._rb_size, size[0], size[1], size[2])
            self._rb_radius.setValue(float(params.get("radius", 0.5)))
            self._rb_mass.setValue(summary["mass"])
            _set_vector(self._rb_pos, summary["x"], summary["y"], summary["z"])
            _set_vector(self._rb_vel, summary["vx"], summary["vy"], summary["vz"])
            _set_quat(self._rb_quat, summary["qw"], summary["qx"], summary["qy"], summary["qz"])
            _set_vector(self._rb_omega, summary["wx"], summary["wy"], summary["wz"])
            inertia = summary["inertia_body"]
            inertia_diag = np.diag(inertia)
            _set_vector(self._rb_inertia, inertia_diag[0], inertia_diag[1], inertia_diag[2])
            self._on_rigid_kind_changed()
            self._stack.setCurrentWidget(self._rigid_page)
            self._set_enabled(True)
            if summary["source"] is None:
                self._set_rigid_custom()
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
            self._rb_kind,
            *self._rb_size,
            self._rb_radius,
            self._rb_mass,
            *self._rb_pos,
            *self._rb_vel,
            *self._rb_quat,
            *self._rb_omega,
            *self._rb_inertia,
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
            elif self._obj.type == "rigid_body":
                kind = self._rb_kind.currentData()
                if kind == "box":
                    params = {"size": self._vector_values(self._rb_size)}
                else:
                    params = {"radius": self._rb_radius.value()}
                apply_rigid_body_edit(
                    self._defn,
                    self._obj.index,
                    kind,
                    params,
                    self._rb_mass.value(),
                    self._vector_values(self._rb_pos),
                    self._vector_values(self._rb_vel),
                    self._quat_values(self._rb_quat),
                    self._vector_values(self._rb_omega),
                )
            else:
                return
        except Exception as exc:  # pragma: no cover - Qt error path
            QtWidgets.QMessageBox.critical(self, "Invalid Values", str(exc))
            return
        self.applied.emit(self._obj)

    def _on_revert(self) -> None:
        self._sync_from_defn()

    def _set_rigid_custom(self) -> None:
        for widget in [self._rb_kind, *self._rb_size, self._rb_radius, self._rb_mass]:
            widget.setEnabled(False)

    def _on_rigid_kind_changed(self) -> None:
        kind = self._rb_kind.currentData()
        is_box = kind == "box"
        for widget in self._rb_size:
            widget.setVisible(is_box)
        self._rb_radius.setVisible(not is_box)

    @staticmethod
    def _vector_values(fields: Sequence[QtWidgets.QDoubleSpinBox]) -> list[float]:
        return [field.value() for field in fields]

    @staticmethod
    def _quat_values(fields: Sequence[QtWidgets.QDoubleSpinBox]) -> list[float]:
        return [field.value() for field in fields]


def _vector_fields(
    min_value: float = -1e9, read_only: bool = False
) -> tuple[QtWidgets.QDoubleSpinBox, QtWidgets.QDoubleSpinBox, QtWidgets.QDoubleSpinBox]:
    fields = []
    for _ in range(3):
        spin = QtWidgets.QDoubleSpinBox()
        spin.setRange(min_value, 1e9)
        spin.setDecimals(6)
        spin.setSingleStep(0.1)
        spin.setReadOnly(read_only)
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


def _quat_fields() -> tuple[
    QtWidgets.QDoubleSpinBox,
    QtWidgets.QDoubleSpinBox,
    QtWidgets.QDoubleSpinBox,
    QtWidgets.QDoubleSpinBox,
]:
    fields = []
    for _ in range(4):
        spin = QtWidgets.QDoubleSpinBox()
        spin.setRange(-10.0, 10.0)
        spin.setDecimals(6)
        spin.setSingleStep(0.05)
        fields.append(spin)
    return fields[0], fields[1], fields[2], fields[3]


def _set_quat(
    fields: Sequence[QtWidgets.QDoubleSpinBox], w: float, x: float, y: float, z: float
) -> None:
    fields[0].setValue(w)
    fields[1].setValue(x)
    fields[2].setValue(y)
    fields[3].setValue(z)
