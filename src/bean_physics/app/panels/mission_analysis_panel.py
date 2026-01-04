"""Mission analysis panel for Hohmann transfers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from PySide6 import QtCore, QtWidgets

from ...analysis.hohmann_planner import (
    G0,
    HohmannPlanInputs,
    build_hohmann_impulse_events,
    build_hohmann_metadata,
    burn_mass_sequence,
    compute_hohmann,
    prograde_delta_v,
    preview_velocity_at_time,
    rocket_equation_propellant,
)
from ...core.impulse_events import ImpulseEvent
from ...io.units import UnitsConfig, config_from_defn, from_si, to_si
from .objects_utils import particle_radius_m, rigid_body_radius_m
from ..session import ScenarioSession
from ..sim_controller import SimulationController


@dataclass(frozen=True, slots=True)
class _EntityEntry:
    obj_id: str
    obj_type: str
    index: int
    mass_si: float


class MissionAnalysisPanel(QtWidgets.QWidget):
    apply_requested = QtCore.Signal()

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self._session: ScenarioSession | None = None
        self._controller: SimulationController | None = None
        self._defn: dict[str, Any] | None = None
        self._units_cfg = UnitsConfig(preset="SI", enabled=True)
        self._entity_entries: list[_EntityEntry] = []

        self._central_body = QtWidgets.QComboBox(self)
        self._spacecraft = QtWidgets.QComboBox(self)
        self._central_mass = QtWidgets.QLabel("-", self)
        self._central_mu = QtWidgets.QLabel("-", self)
        self._body_radius_source = QtWidgets.QLabel("", self)

        self._r1_mode = QtWidgets.QComboBox(self)
        self._r1_mode.addItems(["Altitude above body radius", "Absolute radius"])
        self._r2_mode = QtWidgets.QComboBox(self)
        self._r2_mode.addItems(["Altitude above body radius", "Absolute radius"])

        self._body_radius = self._spin(0.0, 1e12, 1.0)
        self._r1_altitude = self._spin(0.0, 1e12, 1.0)
        self._r1_radius = self._spin(0.0, 1e12, 1.0)
        self._r2_altitude = self._spin(0.0, 1e12, 1.0)
        self._r2_radius = self._spin(0.0, 1e12, 1.0)
        self._r1_computed = QtWidgets.QLabel("-", self)
        self._r2_computed = QtWidgets.QLabel("-", self)

        self._coast_time = self._spin(0.0, 1e12, 1.0)
        self._burn2_mode = QtWidgets.QComboBox(self)
        self._burn2_mode.addItems(["Auto at apoapsis", "Manual time"])
        self._burn2_time = self._spin(0.0, 1e12, 1.0)

        self._dry_mass = self._spin(0.0, 1e12, 1.0)
        self._prop_mass = self._spin(0.0, 1e12, 1.0)
        self._isp = self._spin(0.0, 1e6, 1.0)
        self._thrust = self._spin(0.0, 1e12, 1.0)

        self._ve = QtWidgets.QLabel("-", self)
        self._mass_ratio = QtWidgets.QLabel("-", self)
        self._prop_required = QtWidgets.QLabel("-", self)
        self._prop_margin = QtWidgets.QLabel("-", self)
        self._burn1_time_est = QtWidgets.QLabel("-", self)
        self._burn2_time_est = QtWidgets.QLabel("-", self)

        self._dv1 = QtWidgets.QLabel("-", self)
        self._dv2 = QtWidgets.QLabel("-", self)
        self._dv_total = QtWidgets.QLabel("-", self)
        self._t_transfer = QtWidgets.QLabel("-", self)

        self._formula_text = QtWidgets.QPlainTextEdit(self)
        self._formula_text.setReadOnly(True)
        self._formula_text.setMinimumHeight(160)

        self._status = QtWidgets.QLabel("", self)
        self._status.setStyleSheet("color: #b00020;")

        self._apply_button = QtWidgets.QPushButton("Apply to Scenario", self)
        self._apply_button.clicked.connect(self._on_apply)

        content = QtWidgets.QWidget(self)
        content_layout = QtWidgets.QVBoxLayout(content)
        content_layout.setContentsMargins(8, 8, 8, 8)
        content_layout.addWidget(self._build_central_group())
        content_layout.addWidget(self._build_orbit_group())
        content_layout.addWidget(self._build_timing_group())
        content_layout.addWidget(self._build_spacecraft_group())
        content_layout.addWidget(self._build_results_group())
        content_layout.addWidget(self._build_formula_group())
        content_layout.addWidget(self._status)
        content_layout.addWidget(self._apply_button)
        content_layout.addStretch(1)

        scroll = QtWidgets.QScrollArea(self)
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        scroll.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        scroll.setWidget(content)

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(scroll)

        self._wire_signals()
        self._apply_defaults()
        self._set_enabled(False)

    def bind_session(
        self, session: ScenarioSession, controller: SimulationController
    ) -> None:
        self._session = session
        self._controller = controller
        self.load_from_scenario()

    def unbind(self) -> None:
        self._session = None
        self._controller = None
        self._defn = None
        self._entity_entries = []
        self._central_body.clear()
        self._spacecraft.clear()
        self._set_enabled(False)
        self._status.setText("")

    def load_from_scenario(self) -> None:
        defn = self._session.scenario_def if self._session is not None else None
        self._defn = defn
        self._units_cfg = config_from_defn(defn or {})
        self._refresh_entities()
        self._load_metadata()
        self._set_enabled(defn is not None)
        self._recompute()

    def apply_to_scenario(self) -> bool:
        if self._session is None or self._controller is None:
            return False
        return self._apply_to_scenario()

    def _spin(self, low: float, high: float, step: float) -> QtWidgets.QDoubleSpinBox:
        spin = QtWidgets.QDoubleSpinBox(self)
        spin.setRange(low, high)
        spin.setDecimals(6)
        spin.setSingleStep(step)
        return spin

    def _build_central_group(self) -> QtWidgets.QGroupBox:
        box = QtWidgets.QGroupBox("Central Body", self)
        form = QtWidgets.QFormLayout(box)
        form.addRow("Central body id", self._central_body)
        form.addRow("Spacecraft id", self._spacecraft)
        form.addRow("M_central (kg)", self._central_mass)
        form.addRow("mu (m^3/s^2)", self._central_mu)
        form.addRow("Body radius source", self._body_radius_source)
        return box

    def _build_orbit_group(self) -> QtWidgets.QGroupBox:
        box = QtWidgets.QGroupBox("Orbit Inputs", self)
        grid = QtWidgets.QGridLayout(box)
        grid.addWidget(QtWidgets.QLabel(""), 0, 0)
        grid.addWidget(QtWidgets.QLabel("Initial"), 0, 1)
        grid.addWidget(QtWidgets.QLabel("Target"), 0, 2)
        grid.addWidget(QtWidgets.QLabel("Mode"), 1, 0)
        grid.addWidget(self._r1_mode, 1, 1)
        grid.addWidget(self._r2_mode, 1, 2)
        grid.addWidget(QtWidgets.QLabel("Body radius (m)"), 2, 0)
        grid.addWidget(self._body_radius, 2, 1)
        grid.addWidget(QtWidgets.QLabel(""), 2, 2)
        grid.addWidget(QtWidgets.QLabel("Altitude (m)"), 3, 0)
        grid.addWidget(self._r1_altitude, 3, 1)
        grid.addWidget(self._r2_altitude, 3, 2)
        grid.addWidget(QtWidgets.QLabel("Radius (m)"), 4, 0)
        grid.addWidget(self._r1_radius, 4, 1)
        grid.addWidget(self._r2_radius, 4, 2)
        grid.addWidget(QtWidgets.QLabel("Computed r (m)"), 5, 0)
        grid.addWidget(self._r1_computed, 5, 1)
        grid.addWidget(self._r2_computed, 5, 2)
        return box

    def _build_timing_group(self) -> QtWidgets.QGroupBox:
        box = QtWidgets.QGroupBox("Timing", self)
        form = QtWidgets.QFormLayout(box)
        form.addRow("Coast before burn 1 (s)", self._coast_time)
        form.addRow("Burn 2 mode", self._burn2_mode)
        form.addRow("Burn 2 time (s)", self._burn2_time)
        return box

    def _build_spacecraft_group(self) -> QtWidgets.QGroupBox:
        box = QtWidgets.QGroupBox("Spacecraft / Propulsion", self)
        form = QtWidgets.QFormLayout(box)
        form.addRow("Dry mass (kg)", self._dry_mass)
        form.addRow("Propellant mass (kg)", self._prop_mass)
        form.addRow("Isp (s)", self._isp)
        form.addRow("Thrust (N)", self._thrust)
        form.addRow("ve (m/s)", self._ve)
        form.addRow("Mass ratio m0/mf", self._mass_ratio)
        form.addRow("Prop required (kg)", self._prop_required)
        form.addRow("Prop margin (kg)", self._prop_margin)
        form.addRow("Burn 1 time (s)", self._burn1_time_est)
        form.addRow("Burn 2 time (s)", self._burn2_time_est)
        return box

    def _build_results_group(self) -> QtWidgets.QGroupBox:
        box = QtWidgets.QGroupBox("Computed Outputs", self)
        form = QtWidgets.QFormLayout(box)
        form.addRow("Delta-v1 (m/s)", self._dv1)
        form.addRow("Delta-v2 (m/s)", self._dv2)
        form.addRow("Total delta-v (m/s)", self._dv_total)
        form.addRow("Transfer time (s)", self._t_transfer)
        return box

    def _build_formula_group(self) -> QtWidgets.QGroupBox:
        box = QtWidgets.QGroupBox("Formulas", self)
        layout = QtWidgets.QVBoxLayout(box)
        layout.addWidget(self._formula_text)
        return box

    def _wire_signals(self) -> None:
        widgets = [
            self._central_body,
            self._spacecraft,
            self._r1_mode,
            self._r2_mode,
            self._body_radius,
            self._r1_altitude,
            self._r1_radius,
            self._r2_altitude,
            self._r2_radius,
            self._coast_time,
            self._burn2_mode,
            self._burn2_time,
            self._dry_mass,
            self._prop_mass,
            self._isp,
            self._thrust,
        ]
        for widget in widgets:
            if isinstance(widget, QtWidgets.QComboBox):
                widget.currentIndexChanged.connect(self._recompute)
            elif isinstance(widget, QtWidgets.QDoubleSpinBox):
                widget.valueChanged.connect(self._recompute)

    def _apply_defaults(self) -> None:
        self._body_radius.setValue(6_378_137.0)
        self._r1_altitude.setValue(300_000.0)
        self._r1_radius.setValue(6_678_137.0)
        self._r2_altitude.setValue(35_786_000.0)
        self._r2_radius.setValue(42_164_000.0)
        self._coast_time.setValue(0.0)
        self._burn2_time.setValue(0.0)
        self._dry_mass.setValue(1000.0)
        self._prop_mass.setValue(0.0)
        self._isp.setValue(300.0)
        self._thrust.setValue(0.0)

    def _set_enabled(self, enabled: bool) -> None:
        for widget in [
            self._central_body,
            self._spacecraft,
            self._r1_mode,
            self._r2_mode,
            self._body_radius,
            self._r1_altitude,
            self._r1_radius,
            self._r2_altitude,
            self._r2_radius,
            self._coast_time,
            self._burn2_mode,
            self._burn2_time,
            self._dry_mass,
            self._prop_mass,
            self._isp,
            self._thrust,
            self._apply_button,
        ]:
            widget.setEnabled(enabled)

    def _refresh_entities(self) -> None:
        self._entity_entries = []
        self._central_body.clear()
        self._spacecraft.clear()
        if self._defn is None:
            return
        entries = _collect_entities(self._defn, self._units_cfg)
        self._entity_entries = entries
        for entry in entries:
            self._central_body.addItem(entry.obj_id, entry)
            self._spacecraft.addItem(entry.obj_id, entry)
        self._select_default_ids(entries)

    def _select_default_ids(self, entries: list[_EntityEntry]) -> None:
        if not entries:
            return
        central_idx = _find_entry_index(entries, "earth")
        if central_idx is None:
            central_idx = 0
        self._central_body.setCurrentIndex(central_idx)

        sc_idx = _find_entry_index(entries, "sc")
        if sc_idx is None:
            sc_idx = 0
            if central_idx == 0 and len(entries) > 1:
                sc_idx = 1
        self._spacecraft.setCurrentIndex(sc_idx)

    def _load_metadata(self) -> None:
        if self._defn is None:
            return
        meta = self._defn.get("metadata", {})
        mission = meta.get("mission_analysis", {})
        hohmann = mission.get("hohmann", {})
        self._set_combo_value(self._central_body, hohmann.get("central_body_id"))
        self._set_combo_value(self._spacecraft, hohmann.get("spacecraft_id"))
        self._set_combo_value(self._r1_mode, hohmann.get("r1_mode"))
        self._set_combo_value(self._r2_mode, hohmann.get("r2_mode"))
        self._set_combo_value(self._burn2_mode, hohmann.get("burn2_mode"))
        self._set_spin_value(self._body_radius, hohmann.get("body_radius_m"))
        self._set_spin_value(self._r1_altitude, hohmann.get("r1_altitude_m"))
        self._set_spin_value(self._r1_radius, hohmann.get("r1_radius_m"))
        self._set_spin_value(self._r2_altitude, hohmann.get("r2_altitude_m"))
        self._set_spin_value(self._r2_radius, hohmann.get("r2_radius_m"))
        self._set_spin_value(self._coast_time, hohmann.get("coast_time_s"))
        self._set_spin_value(self._burn2_time, hohmann.get("burn2_time_s"))
        self._set_spin_value(self._dry_mass, hohmann.get("dry_mass_kg"))
        self._set_spin_value(self._prop_mass, hohmann.get("prop_mass_kg"))
        self._set_spin_value(self._isp, hohmann.get("isp_s"))
        self._set_spin_value(self._thrust, hohmann.get("thrust_n"))
        if "body_radius_m" not in hohmann:
            entry = self._current_entry(self._central_body)
            if entry is not None:
                radius = _physical_radius(self._defn, entry)
                if radius is not None:
                    self._body_radius.setValue(radius)

    def _set_combo_value(self, combo: QtWidgets.QComboBox, value: object) -> None:
        if value is None:
            return
        for idx in range(combo.count()):
            entry = combo.itemData(idx)
            if isinstance(entry, _EntityEntry) and entry.obj_id == value:
                combo.setCurrentIndex(idx)
                return
            if isinstance(value, str) and combo.itemText(idx) == value:
                combo.setCurrentIndex(idx)
                return
        if isinstance(value, str):
            idx = combo.findText(value)
            if idx >= 0:
                combo.setCurrentIndex(idx)

    def _set_spin_value(self, spin: QtWidgets.QDoubleSpinBox, value: object) -> None:
        if value is None:
            return
        try:
            spin.setValue(float(value))
        except (TypeError, ValueError):
            return

    def _recompute(self) -> None:
        self._status.setText("")
        entry = self._current_entry(self._central_body)
        if entry is None:
            self._central_mass.setText("-")
            self._central_mu.setText("-")
            self._body_radius_source.setText("")
            self._set_results(None)
            return
        g_val = _find_nbody_g(self._defn, self._units_cfg)
        if g_val is None:
            self._central_mass.setText(f"{entry.mass_si:.6g}")
            self._central_mu.setText("-")
            self._set_results(None)
            return
        mu = g_val * entry.mass_si
        self._central_mass.setText(f"{entry.mass_si:.6g}")
        self._central_mu.setText(f"{mu:.6g}")
        radius_source = _physical_radius(self._defn, entry)
        if radius_source is not None:
            if self._body_radius.value() != radius_source:
                self._body_radius.setValue(radius_source)
            self._body_radius_source.setText(
                f"Using {entry.obj_id} radius_m={radius_source:.6g}"
            )
        else:
            self._body_radius_source.setText("Using manual body radius")

        r1 = self._compute_radius(self._r1_mode, self._r1_altitude, self._r1_radius)
        r2 = self._compute_radius(self._r2_mode, self._r2_altitude, self._r2_radius)
        self._r1_computed.setText(f"{r1:.6g}")
        self._r2_computed.setText(f"{r2:.6g}")
        if r2 <= r1:
            self._status.setText("Target radius must be greater than initial radius.")
            self._set_results(None)
            return

        try:
            res = compute_hohmann(mu, r1, r2)
        except ValueError as exc:
            self._status.setText(str(exc))
            self._set_results(None)
            return

        self._set_results(res)
        self._update_propellant(res.dv_total, res.dv1, res.dv2)
        self._update_formula_text(res)

    def _set_results(self, res: object | None) -> None:
        if res is None:
            for label in (self._dv1, self._dv2, self._dv_total, self._t_transfer):
                label.setText("-")
            self._formula_text.setPlainText("")
            self._ve.setText("-")
            self._mass_ratio.setText("-")
            self._prop_required.setText("-")
            self._prop_margin.setText("-")
            self._burn1_time_est.setText("-")
            self._burn2_time_est.setText("-")
            return
        assert isinstance(res, object)
        self._dv1.setText(f"{res.dv1:.6g}")
        self._dv2.setText(f"{res.dv2:.6g}")
        self._dv_total.setText(f"{res.dv_total:.6g}")
        self._t_transfer.setText(f"{res.t_transfer:.6g}")

    def _update_propellant(self, dv_total: float, dv1: float, dv2: float) -> None:
        dry_mass = self._dry_mass.value()
        prop_mass = self._prop_mass.value()
        isp = self._isp.value()
        thrust = self._thrust.value()
        if dry_mass <= 0.0 or isp <= 0.0:
            self._ve.setText("-")
            self._mass_ratio.setText("-")
            self._prop_required.setText("-")
            self._prop_margin.setText("-")
            self._burn1_time_est.setText("-")
            self._burn2_time_est.setText("-")
            return
        prop_required, mass_ratio, ve = rocket_equation_propellant(dry_mass, dv_total, isp)
        self._ve.setText(f"{ve:.6g}")
        self._mass_ratio.setText(f"{mass_ratio:.6g}")
        self._prop_required.setText(f"{prop_required:.6g}")
        self._prop_margin.setText(f"{prop_mass - prop_required:.6g}")
        if thrust <= 0.0:
            self._burn1_time_est.setText("-")
            self._burn2_time_est.setText("-")
            return
        m0 = dry_mass + prop_mass
        masses = burn_mass_sequence(m0, [dv1, dv2], ve)
        t1 = (masses[0][0] * dv1) / thrust if dv1 > 0.0 else 0.0
        t2 = (masses[1][0] * dv2) / thrust if dv2 > 0.0 else 0.0
        self._burn1_time_est.setText(f"{t1:.6g}")
        self._burn2_time_est.setText(f"{t2:.6g}")

    def _update_formula_text(self, res: object) -> None:
        burn2_mode = "auto" if self._burn2_mode.currentIndex() == 0 else "manual"
        t1 = self._coast_time.value()
        t2 = self._burn2_time.value()
        if burn2_mode == "auto":
            t2 = t1 + res.t_transfer
        text = (
            "v_circ(r) = sqrt(mu/r)\n"
            f"  v_circ(r1) = {res.v1:.6f} m/s\n"
            f"  v_circ(r2) = {res.v2:.6f} m/s\n"
            "a_transfer = (r1 + r2)/2\n"
            f"  a_transfer = {res.a_transfer:.6f} m\n"
            "v_p = sqrt( mu*(2/r1 - 1/a_transfer) )\n"
            f"  v_p = {res.v_p:.6f} m/s\n"
            "v_a = sqrt( mu*(2/r2 - 1/a_transfer) )\n"
            f"  v_a = {res.v_a:.6f} m/s\n"
            "Delta-v1 = v_p - v_circ(r1)\n"
            f"  Delta-v1 = {res.dv1:.6f} m/s\n"
            "Delta-v2 = v_circ(r2) - v_a\n"
            f"  Delta-v2 = {res.dv2:.6f} m/s\n"
            "t_transfer = pi * sqrt(a_transfer^3 / mu)\n"
            f"  t_transfer = {res.t_transfer:.6f} s\n"
            "Rocket equation:\n"
            "  Delta-v_total = ve * ln(m0/mf), ve = Isp*g0\n"
            f"  Delta-v_total = {res.dv_total:.6f} m/s\n"
            f"  ve = {self._ve.text()} m/s (g0={G0})\n"
            f"  burn1 at t = {t1:.3f} s, burn2 at t = {t2:.3f} s\n"
        )
        self._formula_text.setPlainText(text)

    def _compute_radius(
        self,
        mode_combo: QtWidgets.QComboBox,
        altitude_spin: QtWidgets.QDoubleSpinBox,
        radius_spin: QtWidgets.QDoubleSpinBox,
    ) -> float:
        if mode_combo.currentIndex() == 0:
            return self._body_radius.value() + altitude_spin.value()
        return radius_spin.value()

    def _current_entry(self, combo: QtWidgets.QComboBox) -> _EntityEntry | None:
        data = combo.currentData()
        if isinstance(data, _EntityEntry):
            return data
        return None

    def _on_apply(self) -> None:
        if not self.apply_to_scenario():
            return
        self.apply_requested.emit()

    def _apply_to_scenario(self) -> bool:
        if self._defn is None or self._controller is None:
            return False
        entry = self._current_entry(self._central_body)
        target = self._current_entry(self._spacecraft)
        if entry is None or target is None:
            self._status.setText("Select a central body and spacecraft.")
            return False
        g_val = _find_nbody_g(self._defn, self._units_cfg)
        if g_val is None:
            self._status.setText("Scenario requires N-body gravity for mu.")
            return False
        r1 = self._compute_radius(self._r1_mode, self._r1_altitude, self._r1_radius)
        r2 = self._compute_radius(self._r2_mode, self._r2_altitude, self._r2_radius)
        if r2 <= r1:
            self._status.setText("Target radius must be greater than initial radius.")
            return False
        mu = g_val * entry.mass_si
        plan = HohmannPlanInputs(
            central_body_id=entry.obj_id,
            spacecraft_id=target.obj_id,
            r1=r1,
            r2=r2,
            body_radius_m=float(self._body_radius.value()),
            r1_mode=self._r1_mode.currentText(),
            r2_mode=self._r2_mode.currentText(),
            r1_altitude_m=float(self._r1_altitude.value()),
            r2_altitude_m=float(self._r2_altitude.value()),
            coast_time_s=float(self._coast_time.value()),
            burn2_mode=self._burn2_mode.currentText(),
            burn2_time_s=float(self._burn2_time.value()),
            dry_mass_kg=float(self._dry_mass.value()),
            prop_mass_kg=float(self._prop_mass.value()),
            isp_s=float(self._isp.value()),
            thrust_n=float(self._thrust.value()),
        )
        runtime = getattr(self._controller, "runtime", None)
        if runtime is None:
            self._status.setText("Scenario must be loaded before applying impulses.")
            return False

        try:
            res = compute_hohmann(mu, r1, r2)
            v1 = preview_velocity_at_time(
                runtime.state,
                runtime.model,
                runtime.integrator,
                runtime.dt,
                plan.coast_time_s,
                target.obj_type,
                target.index,
                impulse_events=None,
            )
            dv1_world = prograde_delta_v(v1, res.dv1)
            burn1 = ImpulseEvent(
                t=float(plan.coast_time_s),
                target_type=target.obj_type,
                target_index=target.index,
                target_id=target.obj_id,
                delta_v=np.asarray(dv1_world, dtype=np.float64),
                label=None,
            )
            t2_preview = plan.burn2_time_s
            if plan.burn2_mode.lower().startswith("auto"):
                t2_preview = plan.coast_time_s + res.t_transfer
            v2 = preview_velocity_at_time(
                runtime.state,
                runtime.model,
                runtime.integrator,
                runtime.dt,
                t2_preview,
                target.obj_type,
                target.index,
                impulse_events=[burn1],
            )
        except Exception as exc:
            self._status.setText(f"Impulse preview failed: {exc}")
            return False

        impulse_events, res, t1, t2 = build_hohmann_impulse_events(
            mu,
            plan,
            v1,
            v2,
            self._units_cfg,
        )
        self._apply_initial_orbit(entry, target, mu, plan.r1)
        _ensure_ids(self._defn)
        self._defn["impulse_events"] = impulse_events
        meta_update = build_hohmann_metadata(plan)
        meta = self._defn.setdefault("metadata", {})
        meta.update(meta_update)

        summary = (
            f"Applied: Burn1 t={t1:.2f}s dv1={res.dv1:.2f} m/s; "
            f"Burn2 t={t2:.2f}s dv2={res.dv2:.2f} m/s"
        )
        self._status.setText(summary)
        print(
            "Mission Analysis applied: updated impulse_events (2 burns) "
            f"for spacecraft={target.obj_id}"
        )
        return True

    def _apply_initial_orbit(
        self,
        central: _EntityEntry,
        spacecraft: _EntityEntry,
        mu: float,
        r1: float,
    ) -> None:
        if self._defn is None:
            return
        entities = self._defn.get("entities", {})
        sc_block = entities.get(
            "particles" if spacecraft.obj_type == "particle" else "rigid_bodies"
        )
        c_block = entities.get(
            "particles" if central.obj_type == "particle" else "rigid_bodies"
        )
        if not isinstance(sc_block, dict) or not isinstance(c_block, dict):
            return
        sc_pos = np.asarray(sc_block.get("pos", []), dtype=np.float64)
        sc_vel = np.asarray(sc_block.get("vel", []), dtype=np.float64)
        c_pos = np.asarray(c_block.get("pos", []), dtype=np.float64)
        c_vel = np.asarray(c_block.get("vel", []), dtype=np.float64)
        if sc_pos.shape[0] <= spacecraft.index or c_pos.shape[0] <= central.index:
            return
        units_cfg = self._units_cfg
        sc_pos_si = np.asarray(to_si(sc_pos, "length", units_cfg), dtype=np.float64)
        sc_vel_si = np.asarray(to_si(sc_vel, "velocity", units_cfg), dtype=np.float64)
        c_pos_si = np.asarray(to_si(c_pos, "length", units_cfg), dtype=np.float64)
        c_vel_si = np.asarray(to_si(c_vel, "velocity", units_cfg), dtype=np.float64)
        radial = sc_pos_si[spacecraft.index] - c_pos_si[central.index]
        radial_norm = float(np.linalg.norm(radial))
        if radial_norm == 0.0:
            radial = np.array([1.0, 0.0, 0.0], dtype=np.float64)
            radial_norm = 1.0
        radial_hat = radial / radial_norm
        vel = sc_vel_si[spacecraft.index]
        vel_proj = vel - np.dot(vel, radial_hat) * radial_hat
        if np.linalg.norm(vel_proj) == 0.0:
            axis = np.array([0.0, 0.0, 1.0], dtype=np.float64)
            if abs(np.dot(axis, radial_hat)) > 0.9:
                axis = np.array([0.0, 1.0, 0.0], dtype=np.float64)
            vel_proj = np.cross(axis, radial_hat)
        tangential_hat = vel_proj / np.linalg.norm(vel_proj)
        v1 = np.sqrt(mu / r1)
        sc_pos_si[spacecraft.index] = c_pos_si[central.index] + radial_hat * r1
        sc_vel_si[spacecraft.index] = c_vel_si[central.index] + tangential_hat * v1
        sc_block["pos"][spacecraft.index] = from_si(
            sc_pos_si[spacecraft.index], "length", units_cfg
        ).tolist()
        sc_block["vel"][spacecraft.index] = from_si(
            sc_vel_si[spacecraft.index], "velocity", units_cfg
        ).tolist()



def _find_entry_index(entries: list[_EntityEntry], obj_id: str) -> int | None:
    for idx, entry in enumerate(entries):
        if entry.obj_id == obj_id:
            return idx
    return None


def _collect_entities(defn: dict[str, Any], units_cfg: UnitsConfig) -> list[_EntityEntry]:
    entries: list[_EntityEntry] = []
    entities = defn.get("entities", {})
    particles = entities.get("particles")
    if isinstance(particles, dict):
        ids = _resolved_ids(particles.get("ids"), len(particles.get("mass", [])), "p")
        masses = to_si(particles.get("mass", []), "mass", units_cfg)
        for idx, obj_id in enumerate(ids):
            entries.append(
                _EntityEntry(
                    obj_id=obj_id,
                    obj_type="particle",
                    index=idx,
                    mass_si=float(masses[idx]),
                )
            )
    rigid = entities.get("rigid_bodies")
    if isinstance(rigid, dict):
        ids = _resolved_ids(rigid.get("ids"), len(rigid.get("mass", [])), "r")
        masses = to_si(rigid.get("mass", []), "mass", units_cfg)
        for idx, obj_id in enumerate(ids):
            entries.append(
                _EntityEntry(
                    obj_id=obj_id,
                    obj_type="rigid_body",
                    index=idx,
                    mass_si=float(masses[idx]),
                )
            )
    return entries


def _resolved_ids(ids: Any, count: int, prefix: str) -> list[str]:
    if isinstance(ids, list) and len(ids) >= count:
        return [str(val) for val in ids[:count]]
    return [f"{prefix}{idx}" for idx in range(count)]


def _find_nbody_g(defn: dict[str, Any] | None, units_cfg: UnitsConfig) -> float | None:
    if defn is None:
        return None
    for entry in defn.get("models", []):
        if "nbody_gravity" in entry:
            g = entry["nbody_gravity"].get("G")
            if g is None:
                return None
            return float(to_si(g, "G", units_cfg))
    return None


def _physical_radius(defn: dict[str, Any] | None, entry: _EntityEntry) -> float | None:
    if defn is None:
        return None
    if entry.obj_type == "particle":
        return particle_radius_m(defn, entry.index)
    if entry.obj_type == "rigid_body":
        return rigid_body_radius_m(defn, entry.index)
    return None


def _ensure_ids(defn: dict[str, Any]) -> None:
    entities = defn.get("entities", {})
    particles = entities.get("particles")
    if isinstance(particles, dict):
        ids = particles.get("ids")
        count = len(particles.get("mass", []))
        if not isinstance(ids, list):
            ids = []
        if len(ids) < count:
            existing = {str(val) for val in ids}
            for idx in range(count - len(ids)):
                candidate = f"p{len(ids) + idx}"
                while candidate in existing:
                    candidate = f"p{len(ids) + idx + 1}"
                ids.append(candidate)
        particles["ids"] = [str(val) for val in ids[:count]]
    rigid = entities.get("rigid_bodies")
    if isinstance(rigid, dict):
        ids = rigid.get("ids")
        count = len(rigid.get("mass", []))
        if not isinstance(ids, list):
            ids = []
        if len(ids) < count:
            existing = {str(val) for val in ids}
            for idx in range(count - len(ids)):
                candidate = f"r{len(ids) + idx}"
                while candidate in existing:
                    candidate = f"r{len(ids) + idx + 1}"
                ids.append(candidate)
        rigid["ids"] = [str(val) for val in ids[:count]]
