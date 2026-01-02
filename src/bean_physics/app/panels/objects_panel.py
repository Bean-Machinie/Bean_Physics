"""Objects list panel."""

from __future__ import annotations

from typing import Iterable

from PySide6 import QtCore, QtWidgets

from .objects_utils import ObjectRef, particle_summary


class ObjectsPanel(QtWidgets.QWidget):
    selection_changed = QtCore.Signal(object)
    item_activated = QtCore.Signal(object)
    add_requested = QtCore.Signal()
    remove_requested = QtCore.Signal(object)

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self._list = QtWidgets.QListWidget(self)
        self._list.setSelectionMode(
            QtWidgets.QAbstractItemView.SelectionMode.SingleSelection
        )
        self._list.itemSelectionChanged.connect(self._on_selection_changed)
        self._list.itemClicked.connect(self._on_item_clicked)
        self._list.itemDoubleClicked.connect(self._on_item_activated)

        self._btn_add = QtWidgets.QPushButton("Add Particle", self)
        self._btn_remove = QtWidgets.QPushButton("Remove Selected", self)
        self._btn_add.clicked.connect(self.add_requested.emit)
        self._btn_remove.clicked.connect(self._on_remove_clicked)

        button_row = QtWidgets.QHBoxLayout()
        button_row.addWidget(self._btn_add)
        button_row.addWidget(self._btn_remove)
        button_row.addStretch(1)

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.addWidget(self._list)
        layout.addLayout(button_row)

    def set_items(
        self,
        defn: dict,
        objects: Iterable[ObjectRef],
    ) -> None:
        self._list.blockSignals(True)
        self._list.clear()
        for obj in objects:
            if obj.type != "particle":
                continue
            summary = particle_summary(defn, obj.index)
            title = f"Particle {obj.index + 1}"
            detail = (
                f"mass {summary['mass']:.3g}  "
                f"pos ({summary['x']:.3g}, {summary['y']:.3g}, {summary['z']:.3g})"
            )
            item = QtWidgets.QListWidgetItem(f"{title}\n{detail}")
            item.setData(QtCore.Qt.ItemDataRole.UserRole, obj)
            item.setSizeHint(QtCore.QSize(200, 44))
            self._list.addItem(item)
        self._list.blockSignals(False)

    def selected_object(self) -> ObjectRef | None:
        item = self._list.currentItem()
        if item is None:
            return None
        return item.data(QtCore.Qt.ItemDataRole.UserRole)

    def select_object(self, obj: ObjectRef | None) -> None:
        if obj is None:
            self._list.setCurrentRow(-1)
            return
        for row in range(self._list.count()):
            item = self._list.item(row)
            if item.data(QtCore.Qt.ItemDataRole.UserRole) == obj:
                self._list.setCurrentRow(row)
                return

    def _on_selection_changed(self) -> None:
        self.selection_changed.emit(self.selected_object())

    def _on_item_clicked(self, item: QtWidgets.QListWidgetItem) -> None:
        self.selection_changed.emit(item.data(QtCore.Qt.ItemDataRole.UserRole))

    def _on_item_activated(self, item: QtWidgets.QListWidgetItem) -> None:
        self.item_activated.emit(item.data(QtCore.Qt.ItemDataRole.UserRole))

    def _on_remove_clicked(self) -> None:
        self.remove_requested.emit(self.selected_object())
