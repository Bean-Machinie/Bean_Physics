"""Objects list panel."""

from __future__ import annotations

from typing import Iterable

from PySide6 import QtCore, QtGui, QtWidgets

from .objects_utils import ObjectRef, force_summary, particle_summary


class ObjectsPanel(QtWidgets.QWidget):
    selection_changed = QtCore.Signal(object)
    item_activated = QtCore.Signal(object)
    add_object_requested = QtCore.Signal(object)
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

        self._btn_add = QtWidgets.QPushButton("Add...", self)
        self._btn_add.setMenu(self._build_add_menu())
        self._btn_remove = QtWidgets.QPushButton("Delete Selected", self)
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
        particles: Iterable[ObjectRef],
        forces: Iterable[ObjectRef],
    ) -> None:
        self._list.blockSignals(True)
        self._list.clear()
        self._add_section("Particles")
        for obj in particles:
            summary = particle_summary(defn, obj.index)
            title = f"Particle {obj.index + 1}"
            detail = (
                f"mass {summary['mass']:.3g}  "
                f"pos ({summary['x']:.3g}, {summary['y']:.3g}, {summary['z']:.3g})"
            )
            self._add_item(obj, title, detail)

        self._add_section("Forces")
        for obj in forces:
            summary = force_summary(defn, obj.index)
            if summary["type"] == "uniform_gravity":
                title = "Uniform Gravity"
                g = summary["g"]
                detail = f"g=({g[0]:.3g}, {g[1]:.3g}, {g[2]:.3g})"
            else:
                title = "N-body Gravity"
                chunk = summary["chunk_size"]
                chunk_label = "None" if chunk is None else str(chunk)
                detail = (
                    f"G={summary['G']:.3g}  eps={summary['softening']:.3g}  "
                    f"chunk={chunk_label}"
                )
            self._add_item(obj, title, detail)
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

    def _build_add_menu(self) -> QtWidgets.QMenu:
        menu = QtWidgets.QMenu(self)
        particle_action = menu.addAction("Particle")
        particle_action.setData({"type": "particle"})

        force_menu = menu.addMenu("Force")
        uniform_action = force_menu.addAction("Uniform Gravity")
        uniform_action.setData({"type": "force", "subtype": "uniform_gravity"})
        nbody_action = force_menu.addAction("N-body Gravity")
        nbody_action.setData({"type": "force", "subtype": "nbody_gravity"})

        rigid_menu = menu.addMenu("Rigid Body")
        box_action = rigid_menu.addAction("Box (Template)")
        box_action.setData({"type": "rigid_body", "subtype": "box"})
        sphere_action = rigid_menu.addAction("Sphere (Template)")
        sphere_action.setData({"type": "rigid_body", "subtype": "sphere"})

        menu.triggered.connect(self._on_add_action_triggered)
        return menu

    def _on_add_action_triggered(self, action: QtGui.QAction) -> None:
        data = action.data()
        if not data:
            return
        self.add_object_requested.emit(data)

    def _add_section(self, label: str) -> None:
        item = QtWidgets.QListWidgetItem(label)
        item.setFlags(QtCore.Qt.ItemFlag.NoItemFlags)
        item.setForeground(QtCore.Qt.GlobalColor.darkGray)
        item.setSizeHint(QtCore.QSize(200, 24))
        self._list.addItem(item)

    def _add_item(self, obj: ObjectRef, title: str, detail: str) -> None:
        item = QtWidgets.QListWidgetItem(f"{title}\n{detail}")
        item.setData(QtCore.Qt.ItemDataRole.UserRole, obj)
        item.setSizeHint(QtCore.QSize(200, 44))
        self._list.addItem(item)
