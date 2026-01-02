"""Particles table editor panel."""

from __future__ import annotations

from typing import Sequence

from PySide6 import QtCore, QtWidgets

from .particles_utils import ParticleRow


class ParticlesPanel(QtWidgets.QWidget):
    apply_requested = QtCore.Signal(list)

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self._table = QtWidgets.QTableWidget(0, 7, self)
        self._table.setHorizontalHeaderLabels(
            ["x", "y", "z", "vx", "vy", "vz", "mass"]
        )
        self._table.setSelectionBehavior(
            QtWidgets.QAbstractItemView.SelectionBehavior.SelectRows
        )
        self._table.setSelectionMode(
            QtWidgets.QAbstractItemView.SelectionMode.ExtendedSelection
        )
        self._table.horizontalHeader().setStretchLastSection(True)

        self._btn_add = QtWidgets.QPushButton("Add Particle", self)
        self._btn_remove = QtWidgets.QPushButton("Remove Selected", self)
        self._btn_apply = QtWidgets.QPushButton("Apply", self)

        self._btn_add.clicked.connect(self._on_add)
        self._btn_remove.clicked.connect(self._on_remove_selected)
        self._btn_apply.clicked.connect(self._on_apply)

        button_row = QtWidgets.QHBoxLayout()
        button_row.addWidget(self._btn_add)
        button_row.addWidget(self._btn_remove)
        button_row.addStretch(1)
        button_row.addWidget(self._btn_apply)

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.addWidget(self._table)
        layout.addLayout(button_row)

    def set_rows(self, rows: Sequence[Sequence[float]]) -> None:
        self._table.blockSignals(True)
        self._table.setRowCount(0)
        for row in rows:
            self._append_row([str(val) for val in row])
        self._table.blockSignals(False)

    def rows(self) -> list[list[str]]:
        rows: list[list[str]] = []
        for r in range(self._table.rowCount()):
            row_values: list[str] = []
            for c in range(self._table.columnCount()):
                item = self._table.item(r, c)
                row_values.append(item.text() if item is not None else "")
            rows.append(row_values)
        return rows

    def _append_row(self, values: Sequence[str]) -> None:
        row = self._table.rowCount()
        self._table.insertRow(row)
        for col, value in enumerate(values):
            item = QtWidgets.QTableWidgetItem(value)
            item.setTextAlignment(QtCore.Qt.AlignmentFlag.AlignRight)
            self._table.setItem(row, col, item)

    def _on_add(self) -> None:
        self._append_row(["0", "0", "0", "0", "0", "0", "1"])

    def _on_remove_selected(self) -> None:
        ranges = self._table.selectedRanges()
        rows = set()
        for rng in ranges:
            for r in range(rng.topRow(), rng.bottomRow() + 1):
                rows.add(r)
        for row in sorted(rows, reverse=True):
            self._table.removeRow(row)

    def _on_apply(self) -> None:
        self.apply_requested.emit(self.rows())
