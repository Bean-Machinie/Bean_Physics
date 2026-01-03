"""Object details inspector dialog."""

from __future__ import annotations

from pathlib import Path
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
    particle_visual,
    rigid_body_force_points,
    rigid_body_summary,
    rigid_body_visual,
    set_rigid_body_force_points,
    set_particle_visual,
    set_rigid_body_visual,
)
from ..core.rigid_body.mass_properties import rigid_body_from_points, shift_points_to_com
from .visual_assets import loader_available


class ObjectInspector(QtWidgets.QDialog):
    applied = QtCore.Signal(object)

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Object Details")
        self.setModal(False)
        self._defn: dict | None = None
        self._obj: ObjectRef | None = None
        self._scenario_path: Path | None = None
        self._updating_points = False

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
        self._particle_visual = _build_visual_controls(self)
        self._particle_visual["attach"].clicked.connect(
            lambda: self._on_attach_visual(self._particle_visual)
        )
        self._particle_visual["clear"].clicked.connect(
            lambda: self._on_clear_visual(self._particle_visual)
        )
        particle_form.addRow("Visual", self._particle_visual["container"])
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
        self._rb_kind.addItem("Points", "points")
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
        self._rb_points_widget = QtWidgets.QWidget(self)
        points_layout = QtWidgets.QVBoxLayout(self._rb_points_widget)
        points_layout.setContentsMargins(0, 0, 0, 0)
        self._rb_points_table = QtWidgets.QTableWidget(0, 4, self)
        self._rb_points_table.setHorizontalHeaderLabels(["mass", "x", "y", "z"])
        self._rb_points_table.horizontalHeader().setStretchLastSection(True)
        self._rb_points_table.verticalHeader().setVisible(False)
        self._rb_points_table.setSelectionBehavior(
            QtWidgets.QAbstractItemView.SelectionBehavior.SelectRows
        )
        self._rb_points_table.setSelectionMode(
            QtWidgets.QAbstractItemView.SelectionMode.SingleSelection
        )
        self._rb_points_table.cellChanged.connect(self._on_points_changed)
        points_layout.addWidget(self._rb_points_table)
        points_buttons = QtWidgets.QHBoxLayout()
        self._rb_points_add = QtWidgets.QPushButton("Add Point", self)
        self._rb_points_remove = QtWidgets.QPushButton("Remove Selected", self)
        self._rb_points_recenter = QtWidgets.QPushButton("Recenter to CoM", self)
        self._rb_points_add.clicked.connect(self._on_add_point)
        self._rb_points_remove.clicked.connect(self._on_remove_point)
        self._rb_points_recenter.clicked.connect(self._on_recenter_points)
        points_buttons.addWidget(self._rb_points_add)
        points_buttons.addWidget(self._rb_points_remove)
        points_buttons.addWidget(self._rb_points_recenter)
        points_buttons.addStretch(1)
        points_layout.addLayout(points_buttons)

        self._rb_points_mass = QtWidgets.QDoubleSpinBox(self)
        self._rb_points_mass.setRange(0.0, 1e12)
        self._rb_points_mass.setDecimals(6)
        self._rb_points_mass.setReadOnly(True)
        self._rb_points_com = _vector_fields(read_only=True)
        self._rb_points_inertia = _matrix_fields(read_only=True)
        points_layout.addLayout(_form_row("Total Mass", self._rb_points_mass))
        points_layout.addLayout(_form_row("CoM (body)", _vector_widget(self._rb_points_com)))
        points_layout.addLayout(
            _form_row("Inertia (body)", _matrix_widget(self._rb_points_inertia))
        )

        rigid_form.addRow("Source Kind", self._rb_kind)
        rigid_form.addRow("Size", _vector_widget(self._rb_size))
        rigid_form.addRow("Radius", self._rb_radius)
        rigid_form.addRow("Mass", self._rb_mass)
        rigid_form.addRow("Position", _vector_widget(self._rb_pos))
        rigid_form.addRow("Velocity", _vector_widget(self._rb_vel))
        rigid_form.addRow("Quat (w,x,y,z)", _vector_widget(self._rb_quat))
        rigid_form.addRow("Omega body", _vector_widget(self._rb_omega))
        rigid_form.addRow("Inertia diag", _vector_widget(self._rb_inertia))
        self._rigid_visual = _build_visual_controls(self)
        self._rigid_visual["attach"].clicked.connect(
            lambda: self._on_attach_visual(self._rigid_visual)
        )
        self._rigid_visual["clear"].clicked.connect(
            lambda: self._on_clear_visual(self._rigid_visual)
        )
        rigid_form.addRow("Visual", self._rigid_visual["container"])
        self._rb_forces_widget = QtWidgets.QWidget(self)
        forces_layout = QtWidgets.QVBoxLayout(self._rb_forces_widget)
        forces_layout.setContentsMargins(0, 0, 0, 0)
        self._rb_forces_table = QtWidgets.QTableWidget(0, 10, self)
        self._rb_forces_table.setHorizontalHeaderLabels(
            ["enabled", "name", "group", "throttle", "fx (body)", "fy (body)", "fz (body)", "rx", "ry", "rz"]
        )
        self._rb_forces_table.horizontalHeader().setStretchLastSection(True)
        self._rb_forces_table.verticalHeader().setVisible(False)
        self._rb_forces_table.setSelectionBehavior(
            QtWidgets.QAbstractItemView.SelectionBehavior.SelectRows
        )
        self._rb_forces_table.setSelectionMode(
            QtWidgets.QAbstractItemView.SelectionMode.SingleSelection
        )
        forces_layout.addWidget(self._rb_forces_table)
        forces_buttons = QtWidgets.QHBoxLayout()
        self._rb_forces_add = QtWidgets.QPushButton("Add Force", self)
        self._rb_forces_remove = QtWidgets.QPushButton("Remove Selected", self)
        self._rb_forces_enable_all = QtWidgets.QPushButton("Enable All", self)
        self._rb_forces_disable_all = QtWidgets.QPushButton("Disable All", self)
        self._rb_forces_add.clicked.connect(self._on_add_force)
        self._rb_forces_remove.clicked.connect(self._on_remove_force)
        self._rb_forces_enable_all.clicked.connect(self._on_enable_all_forces)
        self._rb_forces_disable_all.clicked.connect(self._on_disable_all_forces)
        forces_buttons.addWidget(self._rb_forces_add)
        forces_buttons.addWidget(self._rb_forces_remove)
        forces_buttons.addWidget(self._rb_forces_enable_all)
        forces_buttons.addWidget(self._rb_forces_disable_all)
        forces_buttons.addStretch(1)
        forces_layout.addLayout(forces_buttons)
        rigid_form.addRow("Force Points", self._rb_forces_widget)
        rigid_form.addRow("Points", self._rb_points_widget)
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

    def set_target(
        self,
        defn: dict | None,
        obj: ObjectRef | None,
        scenario_path: Path | None = None,
    ) -> None:
        self._defn = defn
        self._obj = obj
        self._scenario_path = scenario_path
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
            _set_visual_controls(
                self._particle_visual, particle_visual(self._defn, self._obj.index)
            )
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
                0 if kind == "box" else (1 if kind == "sphere" else 2)
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
            _set_visual_controls(
                self._rigid_visual, rigid_body_visual(self._defn, self._obj.index)
            )
            self._load_force_table(
                rigid_body_force_points(self._defn, self._obj.index)
            )
            if kind == "points":
                points = source.get("points")
                if points is None:
                    points = params.get("points", [])
                self._load_points_table(points)
                self._update_points_derived()
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
            self._rb_points_table,
            self._rb_points_add,
            self._rb_points_remove,
            self._rb_points_recenter,
            self._rb_points_mass,
            *self._rb_points_com,
            self._rb_forces_table,
            self._rb_forces_add,
            self._rb_forces_remove,
            self._rb_forces_enable_all,
            self._rb_forces_disable_all,
            self._particle_visual["path"],
            self._particle_visual["attach"],
            self._particle_visual["clear"],
            *self._particle_visual["scale"],
            *self._particle_visual["offset"],
            *self._particle_visual["rotation"],
            *self._particle_visual["color"],
            self._rigid_visual["path"],
            self._rigid_visual["attach"],
            self._rigid_visual["clear"],
            *self._rigid_visual["scale"],
            *self._rigid_visual["offset"],
            *self._rigid_visual["rotation"],
            *self._rigid_visual["color"],
        ]
        for row in self._rb_points_inertia:
            widgets.extend(row)
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
                set_particle_visual(
                    self._defn,
                    self._obj.index,
                    _visual_from_controls(
                        self._particle_visual, self._scenario_path
                    ),
                )
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
                    mass_val = self._rb_mass.value()
                elif kind == "sphere":
                    params = {"radius": self._rb_radius.value()}
                    mass_val = self._rb_mass.value()
                else:
                    params = {"points": self._points_from_table()}
                    mass_val = self._rb_points_mass.value()
                apply_rigid_body_edit(
                    self._defn,
                    self._obj.index,
                    kind,
                    params,
                    mass_val,
                    self._vector_values(self._rb_pos),
                    self._vector_values(self._rb_vel),
                    self._quat_values(self._rb_quat),
                    self._vector_values(self._rb_omega),
                )
                set_rigid_body_visual(
                    self._defn,
                    self._obj.index,
                    _visual_from_controls(self._rigid_visual, self._scenario_path),
                )
                forces = self._forces_from_table()
                set_rigid_body_force_points(self._defn, self._obj.index, forces)
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
        is_sphere = kind == "sphere"
        is_points = kind == "points"
        for widget in self._rb_size:
            widget.setVisible(is_box)
        self._rb_radius.setVisible(is_sphere)
        self._rb_mass.setVisible(not is_points)
        self._rb_points_widget.setVisible(is_points)
        if is_points:
            self._update_points_derived()

    def _load_points_table(self, points: list[dict[str, object]]) -> None:
        self._updating_points = True
        try:
            self._rb_points_table.setRowCount(0)
            for point in points:
                self._append_point_row(point.get("mass", 1.0), point.get("pos", [0.0, 0.0, 0.0]))
        finally:
            self._updating_points = False

    def _append_point_row(self, mass: object, pos: object) -> None:
        row = self._rb_points_table.rowCount()
        self._rb_points_table.insertRow(row)
        entries = [mass, *pos]
        for col, value in enumerate(entries):
            item = QtWidgets.QTableWidgetItem(str(value))
            self._rb_points_table.setItem(row, col, item)

    def _points_from_table(self) -> list[dict[str, object]]:
        points = []
        for row in range(self._rb_points_table.rowCount()):
            mass_item = self._rb_points_table.item(row, 0)
            coords = [
                self._rb_points_table.item(row, col) for col in (1, 2, 3)
            ]
            if mass_item is None or any(item is None for item in coords):
                continue
            points.append(
                {
                    "mass": float(mass_item.text()),
                    "pos": [float(item.text()) for item in coords],
                }
            )
        if not points:
            raise ValueError("points list must not be empty")
        return points

    def _update_points_derived(self) -> None:
        if self._updating_points:
            return
        try:
            points = self._points_from_table()
        except Exception:
            return
        masses = np.array([p["mass"] for p in points], dtype=np.float64)
        positions = np.array([p["pos"] for p in points], dtype=np.float64)
        total_mass, com, inertia = rigid_body_from_points(positions, masses)
        self._rb_points_mass.setValue(float(total_mass))
        _set_vector(self._rb_points_com, com[0], com[1], com[2])
        _set_matrix(self._rb_points_inertia, inertia)
        inertia_diag = np.diag(inertia)
        _set_vector(self._rb_inertia, inertia_diag[0], inertia_diag[1], inertia_diag[2])

    def _on_points_changed(self) -> None:
        self._update_points_derived()

    def _on_add_point(self) -> None:
        self._append_point_row(1.0, [0.0, 0.0, 0.0])
        self._update_points_derived()

    def _on_remove_point(self) -> None:
        row = self._rb_points_table.currentRow()
        if row < 0:
            return
        self._rb_points_table.removeRow(row)
        self._update_points_derived()

    def _on_recenter_points(self) -> None:
        points = self._points_from_table()
        masses = np.array([p["mass"] for p in points], dtype=np.float64)
        positions = np.array([p["pos"] for p in points], dtype=np.float64)
        shifted, _ = shift_points_to_com(positions, masses)
        self._updating_points = True
        try:
            for row in range(self._rb_points_table.rowCount()):
                for col, value in enumerate(shifted[row], start=1):
                    item = self._rb_points_table.item(row, col)
                    if item is None:
                        item = QtWidgets.QTableWidgetItem()
                        self._rb_points_table.setItem(row, col, item)
                    item.setText(f"{value:.6g}")
        finally:
            self._updating_points = False
        self._update_points_derived()

    def _load_force_table(self, forces: list[dict[str, object]]) -> None:
        self._rb_forces_table.setRowCount(0)
        for idx, force in enumerate(forces, start=1):
            self._append_force_row(
                force.get("enabled", True),
                force.get("force_body", force.get("force_world", [0.0, 0.0, 0.0])),
                force.get("point_body", [0.0, 0.0, 0.0]),
                force.get("name", f"Thruster {idx}"),
                force.get("group", ""),
                force.get("throttle", 1.0),
            )

    def _append_force_row(
        self,
        enabled: object,
        force_body: object,
        point_body: object,
        name: object,
        group: object,
        throttle: object,
    ) -> None:
        row = self._rb_forces_table.rowCount()
        self._rb_forces_table.insertRow(row)
        enabled_item = QtWidgets.QTableWidgetItem()
        enabled_item.setFlags(enabled_item.flags() | QtCore.Qt.ItemFlag.ItemIsUserCheckable)
        enabled_item.setCheckState(
            QtCore.Qt.CheckState.Checked if bool(enabled) else QtCore.Qt.CheckState.Unchecked
        )
        self._rb_forces_table.setItem(row, 0, enabled_item)
        entries = [name, group, throttle, *force_body, *point_body]
        for col, value in enumerate(entries, start=1):
            item = QtWidgets.QTableWidgetItem(str(value))
            self._rb_forces_table.setItem(row, col, item)

    def _forces_from_table(self) -> list[dict[str, object]]:
        forces = []
        for row in range(self._rb_forces_table.rowCount()):
            enabled_item = self._rb_forces_table.item(row, 0)
            enabled = True if enabled_item is None else enabled_item.checkState() == QtCore.Qt.CheckState.Checked
            name_item = self._rb_forces_table.item(row, 1)
            group_item = self._rb_forces_table.item(row, 2)
            throttle_item = self._rb_forces_table.item(row, 3)
            name = "" if name_item is None else name_item.text()
            group = "" if group_item is None else group_item.text()
            throttle = 1.0 if throttle_item is None else float(throttle_item.text())
            values = []
            for col in range(4, 10):
                item = self._rb_forces_table.item(row, col)
                if item is None:
                    values.append(0.0)
                else:
                    values.append(float(item.text()))
            force_body = values[:3]
            point_body = values[3:]
            forces.append(
                {
                    "body_index": self._obj.index if self._obj is not None else 0,
                    "force_body": force_body,
                    "point_body": point_body,
                    "enabled": enabled,
                    "name": name,
                    "group": group,
                    "throttle": throttle,
                }
            )
        return forces

    def _on_add_force(self) -> None:
        row = self._rb_forces_table.rowCount() + 1
        self._append_force_row(
            True,
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            f"Thruster {row}",
            "",
            1.0,
        )

    def _on_remove_force(self) -> None:
        row = self._rb_forces_table.currentRow()
        if row < 0:
            return
        self._rb_forces_table.removeRow(row)

    def _on_enable_all_forces(self) -> None:
        for row in range(self._rb_forces_table.rowCount()):
            item = self._rb_forces_table.item(row, 0)
            if item is None:
                item = QtWidgets.QTableWidgetItem()
                item.setFlags(item.flags() | QtCore.Qt.ItemFlag.ItemIsUserCheckable)
                self._rb_forces_table.setItem(row, 0, item)
            item.setCheckState(QtCore.Qt.CheckState.Checked)

    def _on_disable_all_forces(self) -> None:
        for row in range(self._rb_forces_table.rowCount()):
            item = self._rb_forces_table.item(row, 0)
            if item is None:
                item = QtWidgets.QTableWidgetItem()
                item.setFlags(item.flags() | QtCore.Qt.ItemFlag.ItemIsUserCheckable)
                self._rb_forces_table.setItem(row, 0, item)
            item.setCheckState(QtCore.Qt.CheckState.Unchecked)

    def _on_attach_visual(self, controls: dict[str, object]) -> None:
        if not loader_available():
            QtWidgets.QMessageBox.information(
                self,
                "Model Loader Missing",
                "Mesh loading requires the optional 'trimesh' dependency.",
            )
            return
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            "Attach Model",
            "",
            "3D Models (*.glb *.gltf *.obj *.stl *.ply)",
        )
        if not path:
            return
        controls["path"].setText(path)

    def _on_clear_visual(self, controls: dict[str, object]) -> None:
        controls["path"].setText("")

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


def _color_fields() -> tuple[
    QtWidgets.QDoubleSpinBox, QtWidgets.QDoubleSpinBox, QtWidgets.QDoubleSpinBox
]:
    fields = []
    for _ in range(3):
        spin = QtWidgets.QDoubleSpinBox()
        spin.setRange(0.0, 1.0)
        spin.setDecimals(3)
        spin.setSingleStep(0.05)
        fields.append(spin)
    return fields[0], fields[1], fields[2]


def _vector_widget(fields: Sequence[QtWidgets.QDoubleSpinBox]) -> QtWidgets.QWidget:
    widget = QtWidgets.QWidget()
    layout = QtWidgets.QHBoxLayout(widget)
    layout.setContentsMargins(0, 0, 0, 0)
    for field in fields:
        layout.addWidget(field)
    return widget


def _matrix_fields(
    read_only: bool = False,
) -> tuple[
    tuple[QtWidgets.QDoubleSpinBox, QtWidgets.QDoubleSpinBox, QtWidgets.QDoubleSpinBox],
    tuple[QtWidgets.QDoubleSpinBox, QtWidgets.QDoubleSpinBox, QtWidgets.QDoubleSpinBox],
    tuple[QtWidgets.QDoubleSpinBox, QtWidgets.QDoubleSpinBox, QtWidgets.QDoubleSpinBox],
]:
    return (
        _vector_fields(read_only=read_only),
        _vector_fields(read_only=read_only),
        _vector_fields(read_only=read_only),
    )


def _matrix_widget(
    fields: Sequence[Sequence[QtWidgets.QDoubleSpinBox]],
) -> QtWidgets.QWidget:
    widget = QtWidgets.QWidget()
    layout = QtWidgets.QVBoxLayout(widget)
    layout.setContentsMargins(0, 0, 0, 0)
    for row in fields:
        layout.addWidget(_vector_widget(row))
    return widget


def _set_vector(
    fields: Sequence[QtWidgets.QDoubleSpinBox], x: float, y: float, z: float
) -> None:
    fields[0].setValue(x)
    fields[1].setValue(y)
    fields[2].setValue(z)


def _set_matrix(
    fields: Sequence[Sequence[QtWidgets.QDoubleSpinBox]], mat: np.ndarray
) -> None:
    mat = np.asarray(mat, dtype=np.float64)
    for i in range(3):
        _set_vector(fields[i], mat[i, 0], mat[i, 1], mat[i, 2])


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


def _form_row(label: str, widget: QtWidgets.QWidget) -> QtWidgets.QHBoxLayout:
    layout = QtWidgets.QHBoxLayout()
    layout.setContentsMargins(0, 0, 0, 0)
    layout.addWidget(QtWidgets.QLabel(label))
    layout.addWidget(widget, 1)
    return layout


def _build_visual_controls(parent: QtWidgets.QWidget) -> dict[str, object]:
    container = QtWidgets.QWidget(parent)
    layout = QtWidgets.QFormLayout(container)
    path = QtWidgets.QLineEdit(parent)
    path.setReadOnly(True)
    attach = QtWidgets.QPushButton("Attach Model...", parent)
    clear = QtWidgets.QPushButton("Clear", parent)
    button_row = QtWidgets.QHBoxLayout()
    button_row.addWidget(attach)
    button_row.addWidget(clear)
    button_row.addStretch(1)
    scale = _vector_fields(min_value=1e-6)
    offset = _vector_fields()
    rotation = _quat_fields()
    color = _color_fields()
    _set_vector(scale, 1.0, 1.0, 1.0)
    _set_vector(offset, 0.0, 0.0, 0.0)
    _set_quat(rotation, 1.0, 0.0, 0.0, 0.0)
    _set_vector(color, 1.0, 1.0, 1.0)
    layout.addRow("Path", path)
    layout.addRow("", _layout_widget(button_row))
    layout.addRow("Scale", _vector_widget(scale))
    layout.addRow("Offset", _vector_widget(offset))
    layout.addRow("Rotation", _vector_widget(rotation))
    layout.addRow("Color Tint", _vector_widget(color))
    return {
        "container": container,
        "path": path,
        "attach": attach,
        "clear": clear,
        "scale": scale,
        "offset": offset,
        "rotation": rotation,
        "color": color,
    }


def _set_visual_controls(
    controls: dict[str, object], visual: dict[str, object] | None
) -> None:
    if visual is None:
        controls["path"].setText("")
        _set_vector(controls["scale"], 1.0, 1.0, 1.0)
        _set_vector(controls["offset"], 0.0, 0.0, 0.0)
        _set_quat(controls["rotation"], 1.0, 0.0, 0.0, 0.0)
        _set_vector(controls["color"], 1.0, 1.0, 1.0)
        return
    controls["path"].setText(str(visual.get("mesh_path", "")))
    scale = visual.get("scale", [1.0, 1.0, 1.0])
    offset = visual.get("offset_body", [0.0, 0.0, 0.0])
    rotation = visual.get("rotation_body_quat", [1.0, 0.0, 0.0, 0.0])
    color = visual.get("color_tint", [1.0, 1.0, 1.0])
    _set_vector(controls["scale"], scale[0], scale[1], scale[2])
    _set_vector(controls["offset"], offset[0], offset[1], offset[2])
    _set_quat(controls["rotation"], rotation[0], rotation[1], rotation[2], rotation[3])
    _set_vector(controls["color"], color[0], color[1], color[2])


def _visual_from_controls(
    controls: dict[str, object], scenario_path: Path | None
) -> dict[str, object] | None:
    path = controls["path"].text().strip()
    if not path:
        return None
    mesh_path = _relativize_mesh_path(path, scenario_path)
    scale = [field.value() for field in controls["scale"]]
    offset = [field.value() for field in controls["offset"]]
    rotation = [field.value() for field in controls["rotation"]]
    color = [field.value() for field in controls["color"]]
    q = np.asarray(rotation, dtype=np.float64)
    q_norm = np.linalg.norm(q)
    if q_norm > 0:
        q = q / q_norm
    rotation = [float(v) for v in q]
    return {
        "kind": "mesh",
        "mesh_path": mesh_path,
        "scale": scale,
        "offset_body": offset,
        "rotation_body_quat": rotation,
        "color_tint": color,
    }


def _relativize_mesh_path(path: str, scenario_path: Path | None) -> str:
    if scenario_path is None:
        return path
    try:
        return str(Path(path).resolve().relative_to(scenario_path.parent.resolve()))
    except ValueError:
        return path


def _layout_widget(layout: QtWidgets.QLayout) -> QtWidgets.QWidget:
    widget = QtWidgets.QWidget()
    widget.setLayout(layout)
    return widget
