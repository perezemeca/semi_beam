# path: src/semi_beam/ui/main_window.py
from __future__ import annotations

import sys
import os
import tempfile
from datetime import datetime
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple, Dict, Any

import matplotlib
matplotlib.use("QtAgg")
import matplotlib.pyplot as plt
import numpy as np

from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QWheelEvent, QIcon, QColor, QBrush, QAction
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QFormLayout,
    QLabel, QDoubleSpinBox, QPushButton, QTableWidget, QTableWidgetItem,
    QSizePolicy, QSplitter, QScrollArea, QTabWidget, QMessageBox, QFileDialog,
    QToolButton, QFrame, QComboBox, QDialog, QCheckBox, QGroupBox, QAbstractItemView
)
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas

# Delegados numéricos (solo números / admite vacío)
from semi_beam.ui.numeric_delegate import (
    NullableFloatDelegate,
    FlexibleDoubleSpinBox,
    apply_table_readability_style,
    combo_cell_style,
    TABLE_INPUT_BG,
    TABLE_READONLY_BG,
    TABLE_TEXT_COLOR,
)

# ---- Dominio / motor / view ----
from semi_beam.domain.beam import Beam
from semi_beam.domain.loads import PointForce, DistUniform, PointMoment
from semi_beam.domain.labels import p_index, next_free_p_index, to_internal_Fy, to_internal_w_up
from semi_beam.engine.normalize import normalize_inputs
from semi_beam.engine.constraints import check_no_overlap, dist_interval
from semi_beam.view.style import RenderStyle
from semi_beam.view.renderer_fbd import render_fbd

from semi_beam.domain.supports import FixedSupport, TandemSupport, DirectionalSupport
from semi_beam.domain.unknowns import UnknownUniformLoad
from semi_beam.domain.cases import BeamCase
from semi_beam.domain.results import EquilibriumResult
from semi_beam.engine.equilibrium import solve_equilibrium
from semi_beam.engine.reactions import ReactionsResult, solve_reactions_2support
from semi_beam.engine.deflection import compute_total_deflection
from semi_beam.engine.diagrams import build_V_M
from semi_beam.view.diagram_hover import DiagramHoverInspector, HoverCurve
from semi_beam.view.renderer_vm import render_shear, render_moment, render_deflection
from semi_beam.services.memoria_calculo_docx import (
    export_memoria_docx,
    MemoriaHeader,
    MemoriaCaso,
    MemoriaResultados,
    MemoriaSeccion,
)
from semi_beam.services.branding import ensure_calculeitor_icon
from semi_beam.services.study_storage import load_study_file, save_study_file

# ---- Verificador (TU UI anterior) ----
from semi_beam.ui.section_check_panel import SectionCheckPanel
from semi_beam.ui.memoria_header_dialog import MemoriaHeaderDialog
from semi_beam.ui.reactions_tab import SemiTrailerReactionsTab
from semi_beam.ui.number_parsing import normalize_decimal_text, try_parse_user_float


# ============================================================
# CollapsibleBox
# ============================================================
class CollapsibleBox(QWidget):
    def __init__(self, title: str, parent=None):
        super().__init__(parent)

        self._btn = QToolButton()
        self._btn.setText(title)
        self._btn.setCheckable(True)
        self._btn.setChecked(False)
        self._btn.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)
        self._btn.setArrowType(Qt.RightArrow)  # arranque contraído
        self._btn.setStyleSheet("""
            QToolButton {
                border: none;
                background: transparent;
                padding: 4px 6px;
                text-align: left;
                font-weight: 600;
            }
            QToolButton:checked { background: transparent; color: black; }
            QToolButton:hover { background: rgba(0,0,0,0.04); }
            QToolButton:pressed { background: rgba(0,0,0,0.06); }
            QToolButton:focus { outline: none; }
        """)

        self._content = QWidget()
        self._content_layout = QVBoxLayout(self._content)
        self._content_layout.setContentsMargins(6, 6, 6, 6)
        self._content_layout.setSpacing(8)

        self._line = QFrame()
        self._line.setFrameShape(QFrame.HLine)
        self._line.setFrameShadow(QFrame.Sunken)

        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(4)
        lay.addWidget(self._btn)
        lay.addWidget(self._line)
        lay.addWidget(self._content)

        # ✅ iniciar contraído (contenido oculto)
        self._content.setVisible(False)
        self._line.setVisible(False)

        self._btn.toggled.connect(self._on_toggled)

    def content_layout(self) -> QVBoxLayout:
        return self._content_layout

    def set_collapsed(self, collapsed: bool):
        self._btn.setChecked(not collapsed)  # toggled hará el resto

    def _on_toggled(self, checked: bool):
        self._content.setVisible(checked)
        self._line.setVisible(checked)
        self._btn.setArrowType(Qt.DownArrow if checked else Qt.RightArrow)


# ============================================================
# Utilidades
# ============================================================
def _set_item(tbl: QTableWidget, r: int, c: int, text: str):
    it = QTableWidgetItem(str(text))
    it.setTextAlignment(Qt.AlignCenter)
    tbl.setItem(r, c, it)


def _get_text(tbl: QTableWidget, r: int, c: int) -> str:
    it = tbl.item(r, c)
    return "" if it is None else (it.text() or "").strip()


def _try_float(s: str) -> Optional[float]:
    return try_parse_user_float(s)


def _spin_text(sp: QDoubleSpinBox) -> str:
    try:
        le = sp.lineEdit()
        if le is not None:
            return (le.text() or "").strip()
    except Exception:
        pass
    return (sp.text() or "").strip()


def _spin_value_or_none(sp: QDoubleSpinBox) -> Optional[float]:
    return try_parse_user_float(_spin_text(sp), allow_negative=float(sp.minimum()) < 0.0)


def _is_reaction_label(label: str) -> bool:
    l = (label or "").strip().upper()
    return l in {"RP1", "RP2", "RT", "RD"}


def _compute_x_view(beam_L: float, points: List[PointForce], dists: List[DistUniform], moms: List[PointMoment]) -> Tuple[float, float]:
    xs = [0.0, float(beam_L)]
    for p in points:
        xs.append(float(p.x_mm))
    for m in moms:
        xs.append(float(m.x_mm))
    for d in dists:
        x0 = float(d.x0_mm)
        xs.append(x0)
        xs.append(x0 + float(d.Lq_mm))

    x_min = min(xs)
    x_max = max(xs)
    span = max(1.0, x_max - x_min)
    margin = max(0.05 * span, 200.0)
    return x_min - margin, x_max + margin


def _fmt_plain(v, decimals: int = 2) -> str:
    """Formatea números para UI/texto. Tolerante a None (devuelve '-')."""
    if v is None:
        return "-"
    try:
        s = f"{float(v):.{decimals}f}"
    except Exception:
        return str(v)
    if "." in s:
        s = s.rstrip("0").rstrip(".")
    return s


def _point_value(points: List[PointForce], label: str, default: float = 0.0) -> float:
    for pf in points:
        if pf.label == label:
            return float(pf.value_user)
    return float(default)




APP_VERSION = "1.0.0"
APP_TITLE_BASE = "Calculeitor - Acoplado / Semirremolque / Bitren"
MEMORIA_EXPORT_IMAGE_DPI = 220


# ============================================================
# Cache de sesión
# ============================================================
@dataclass
class SessionCache:
    beam_plot: Beam
    points: List[PointForce]
    dists: List[DistUniform]
    moms: List[PointMoment]
    note_text: str
    deflection_supports: Optional[Tuple[float, float]] = None
    memoria_header: dict = field(default_factory=dict)


# ============================================================
# Un TAB completo (estado independiente)
# ============================================================
class UnitTab(QWidget):
    COL_TYPE = 0
    COL_MAG = 1
    COL_POS = 2
    COL_LEN = 3

    LOAD_TYPES = ["Puntual", "Distribuida", "Momento"]

    def __init__(self, title: str, *, is_bitren: bool = False, is_acoplado: bool = False):
        super().__init__()
        self.title = title
        self.is_bitren = is_bitren
        self.is_acoplado = is_acoplado

        self._cached: Optional[SessionCache] = None
        self._last_diag = None
        self._view_mode = "inputs"

        self._all_boxes: List[CollapsibleBox] = []

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)
        self.content_scroll = QScrollArea()
        self.content_scroll.setWidgetResizable(True)
        self.content_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        outer.addWidget(self.content_scroll)

        content = QWidget()
        self.content_scroll.setWidget(content)
        root = QVBoxLayout(content)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(10)

        # ==========================
        # Collapsible: Motor
        # ==========================
        motor_box = CollapsibleBox("Cálculo de equilibrio y posición de tándem")
        self._all_boxes.append(motor_box)
        root.addWidget(motor_box)
        motor_v = motor_box.content_layout()

        formw = QWidget()
        form = QFormLayout(formw)
        form.setRowWrapPolicy(QFormLayout.WrapAllRows)
        form.setFieldGrowthPolicy(QFormLayout.ExpandingFieldsGrow)

        # --- Selectores de configuración ---
        self.cmb_semi_tipo = QComboBox()
        self.cmb_semi_tipo.addItems(["Escalado", "Estándar"])
        self.cmb_semi_tipo.setVisible(not self.is_acoplado and not self.is_bitren)  # solo semirremolque

        self.cmb_config = QComboBox()
        self.chk_axis_lift = QCheckBox("Levante de eje")
        self.chk_axis_lift.setToolTip(
            "Agrega una carga puntual descendente automática para evaluar esfuerzos. "
            "No modifica apoyos ni la posición calculada del tándem; recalcula reacciones."
        )

        # --- Entradas motor comunes ---
        self.Lc = FlexibleDoubleSpinBox()
        self._setup_motor_spin(self.Lc, minv=1.0, maxv=1e12, decimals=2, step=50.0)

        self.x_front_or_kp = FlexibleDoubleSpinBox()
        self._setup_motor_spin(self.x_front_or_kp, minv=0.0, maxv=1e12, decimals=2, step=50.0)

        self.R_front_or_kp = FlexibleDoubleSpinBox()
        self._setup_motor_spin(self.R_front_or_kp, minv=0.0, maxv=1e12, decimals=2, step=50.0)

        self.Rt = FlexibleDoubleSpinBox()
        self._setup_motor_spin(self.Rt, minv=0.0, maxv=1e12, decimals=2, step=50.0)

        # Direccional (semi/bitren)
        self.Rd = FlexibleDoubleSpinBox()
        self._setup_motor_spin(self.Rd, minv=-1e12, maxv=1e12, decimals=2, step=50.0)

        self.dir_offset = FlexibleDoubleSpinBox()
        self._setup_motor_spin(self.dir_offset, minv=0.0, maxv=20000.0, decimals=1, step=25.0)

        # Bitren Rp2
        self.x_rp2_rel = FlexibleDoubleSpinBox()
        self._setup_motor_spin(self.x_rp2_rel, minv=0.0, maxv=1e12, decimals=2, step=50.0)

        self.Rp2 = FlexibleDoubleSpinBox()
        self._setup_motor_spin(self.Rp2, minv=0.0, maxv=1e12, decimals=2, step=50.0)

        # Labels según tab
        if self.is_acoplado:
            lbl_x = "Posición de tren delantero [mm]:"
            lbl_r = "Reacción en tren delantero [Kg] (UP+):"
        else:
            lbl_x = "Posición de perno rey [mm]:"
            lbl_r = "Reacción en perno rey [Kg] (UP+):"

        # Layout
        form.addRow("Largo carrozable [mm]:", self.Lc)

        if (not self.is_acoplado) and (not self.is_bitren):
            form.addRow("Tipo de semirremolque:", self.cmb_semi_tipo)

        form.addRow("Configuración de ejes:", self.cmb_config)
        form.addRow("Simulación de levante:", self.chk_axis_lift)

        form.addRow(lbl_x, self.x_front_or_kp)
        form.addRow(lbl_r, self.R_front_or_kp)
        form.addRow("Reacción en tándem [Kg] (UP+):", self.Rt)

        if not self.is_acoplado:
            form.addRow("Reacción en direccional [Kg] (UP+):", self.Rd)
            form.addRow("Offset direccional (x_t - offset) [mm]:", self.dir_offset)

        if self.is_bitren:
            form.addRow("x_Rp2 relativo a L [mm]:", self.x_rp2_rel)
            form.addRow("Rp2 [Kg] (DOWN+):", self.Rp2)

        btns = QHBoxLayout()
        self.btn_solve = QPushButton("Resolver equilibrio")
        self.btn_clear = QPushButton("Volver a entradas")
        self.btn_reset = QPushButton("Reset")
        btns.addWidget(self.btn_solve)
        btns.addWidget(self.btn_clear)
        btns.addWidget(self.btn_reset)
        btns.addStretch(1)
        form.addRow(btns)

        motor_v.addWidget(formw)

        # Inicializar opciones de config
        self._populate_configs()

        load_box = QGroupBox("Cargas")
        load_lay = QVBoxLayout(load_box)
        load_lay.addWidget(QLabel("Magnitud: kg para puntuales y distribuidas (P total), kg·mm para momentos."))
        self.tbl = QTableWidget(0, 4)
        self.tbl.setHorizontalHeaderLabels(["Tipo", "Magnitud", "Posición / centro [mm]", "Longitud [mm]"])
        self.tbl.horizontalHeader().setStretchLastSection(True)
        apply_table_readability_style(self.tbl)
        self.tbl.setEditTriggers(
            QAbstractItemView.DoubleClicked
            | QAbstractItemView.EditKeyPressed
            | QAbstractItemView.AnyKeyPressed
            | QAbstractItemView.SelectedClicked
        )
        self.tbl.setItemDelegateForColumn(self.COL_MAG, NullableFloatDelegate(self, decimals=2, minv=-1e18, maxv=1e18))
        self.tbl.setItemDelegateForColumn(self.COL_POS, NullableFloatDelegate(self, decimals=2, minv=-1e12, maxv=1e12))
        self.tbl.setItemDelegateForColumn(self.COL_LEN, NullableFloatDelegate(self, decimals=2, minv=0.0, maxv=1e12))
        load_lay.addWidget(self.tbl)

        load_btns = QHBoxLayout()
        self.btn_add_load = QPushButton("Agregar carga")
        self.btn_del_load = QPushButton("Eliminar seleccionadas")
        load_btns.addWidget(self.btn_add_load)
        load_btns.addWidget(self.btn_del_load)
        load_btns.addStretch(1)
        load_lay.addLayout(load_btns)
        root.addWidget(load_box)

        # ==========================
        # Collapsible: Verificador
        # ==========================
        s_box = CollapsibleBox("Verificación de sección a flexión")
        self._all_boxes.append(s_box)
        root.addWidget(s_box)
        s_v = s_box.content_layout()

        self.section_panel = SectionCheckPanel()
        s_v.addWidget(self.section_panel)

        # ==========================
        # Collapsible: Deformada
        # ==========================
        d_box = CollapsibleBox("Deformada")
        self._all_boxes.append(d_box)
        root.addWidget(d_box)
        d_v = d_box.content_layout()

        defl_form_w = QWidget()
        defl_form = QFormLayout(defl_form_w)
        defl_form.setRowWrapPolicy(QFormLayout.WrapAllRows)
        defl_form.setFieldGrowthPolicy(QFormLayout.ExpandingFieldsGrow)

        self.chk_show_deflection = QCheckBox("Mostrar deformada")
        self.chk_show_deflection.setChecked(True)
        self.lbl_defl_e = QLabel("21000 kg/mm²")
        self.lbl_defl_e.setTextInteractionFlags(Qt.TextSelectableByMouse | Qt.TextSelectableByKeyboard)
        self.lbl_deflection = QLabel("Convexidad L/2: +30 mm\nvmin total: -\nUtilizado: - / 60 mm\nEstado: -")
        self.lbl_deflection.setWordWrap(True)
        self.lbl_deflection.setTextInteractionFlags(Qt.TextSelectableByMouse | Qt.TextSelectableByKeyboard)

        defl_form.addRow(self.chk_show_deflection)
        defl_form.addRow("E [kg/mm²]:", self.lbl_defl_e)
        d_v.addWidget(defl_form_w)
        d_v.addWidget(self.lbl_deflection)

        # ==========================
        # Collapsible: Notas
        # ==========================
        n_box = CollapsibleBox("Notas")
        self._all_boxes.append(n_box)
        root.addWidget(n_box)
        n_v = n_box.content_layout()

        self.note_label = QLabel("(sin notas)")
        self.note_label.setWordWrap(True)
        self.note_label.setTextInteractionFlags(Qt.TextSelectableByMouse | Qt.TextSelectableByKeyboard)
        n_v.addWidget(self.note_label)

        root.addStretch(1)

        # Señales tabla de cargas
        self.btn_add_load.clicked.connect(self._add_load_row)
        self.btn_del_load.clicked.connect(self._remove_selected_rows)
        self.tbl.cellChanged.connect(lambda *_: None)

        # Señales config (autocompletar reacciones)
        self.cmb_config.currentIndexChanged.connect(lambda _i: self._apply_config_defaults())
        self.cmb_config.activated.connect(lambda _i: self._apply_config_defaults())
        self.cmb_semi_tipo.currentIndexChanged.connect(lambda _i: self._populate_configs())

        # aplicar defaults iniciales
        self._clear_motor_inputs()
        self.x_front_or_kp.setValue(0.0)
        self.R_front_or_kp.setValue(0.0)

        # ✅ Asegurar que TODOS los CollapsibleBox del tab arranquen contraídos
        for b in self._all_boxes:
            b.set_collapsed(True)

    # -------------------
    # Helpers de inputs
    # -------------------
    def _setup_motor_spin(self, sp: QDoubleSpinBox, *, minv: float, maxv: float, decimals: int, step: float):
        sp.setRange(float(minv), float(maxv))
        sp.setDecimals(int(decimals))
        sp.setSingleStep(float(step))
        sp.setKeyboardTracking(False)
        sp.setSpecialValueText("")
        sp.setValue(float(minv))
        sp.editingFinished.connect(lambda s=sp: self._normalize_spin_editor_text(s))

    def _normalize_spin_editor_text(self, sp: QDoubleSpinBox):
        value = _spin_value_or_none(sp)
        if value is None:
            self._set_spin_blank(sp)
            return
        sp.setValue(value)

    def _set_spin_blank(self, sp: QDoubleSpinBox):
        sp.setValue(float(sp.minimum()))
        try:
            le = sp.lineEdit()
            if le is not None:
                le.clear()
        except Exception:
            pass

    def _clear_motor_inputs(self):
        for sp in (self.Lc, self.x_front_or_kp, self.R_front_or_kp, self.Rt, self.Rd, self.dir_offset, self.x_rp2_rel, self.Rp2):
            self._set_spin_blank(sp)
        self._apply_config_defaults()

    def _config_uses_directional(self) -> bool:
        cfg = (self.cmb_config.currentText() or "").strip()
        compact = cfg.replace(" ", "")
        return (not self.is_acoplado) and any(marker in compact for marker in ("1+2", "1+3", "1+1+1"))

    def _axis_lift_mode(self) -> Optional[Tuple[str, str, float]]:
        cfg = (self.cmb_config.currentText() or "").strip()
        compact = cfg.replace(" ", "")
        if any(marker in compact for marker in ("1+2", "1+3", "1+1+1")):
            return ("directional", "Levantar direccional", 1300.0)
        if cfg.startswith("2 ejes") or cfg.startswith("3 ejes"):
            return ("first_axle", "Levantar primer eje", 1200.0)
        return None

    def _apply_axis_lift_ui(self):
        mode = self._axis_lift_mode()
        if mode is None:
            self.chk_axis_lift.blockSignals(True)
            self.chk_axis_lift.setChecked(False)
            self.chk_axis_lift.blockSignals(False)
            self.chk_axis_lift.setText("Levante de eje no aplicable")
            self.chk_axis_lift.setEnabled(False)
            return
        _kind, label, kg = mode
        self.chk_axis_lift.setText(f"{label} ({_fmt_plain(kg, 0)} kg automaticos)")
        self.chk_axis_lift.setEnabled(True)

    def axis_lift_enabled(self) -> bool:
        return bool(self.chk_axis_lift.isEnabled() and self.chk_axis_lift.isChecked())

    def _first_axle_offset_from_tandem_center(self) -> float:
        cfg = (self.cmb_config.currentText() or "").strip()
        if cfg.startswith("2 ejes"):
            return 625.0
        if cfg.startswith("3 ejes"):
            return 1250.0
        return 0.0

    def axis_lift_load(self, *, x_t_mm: float, x_d_mm: Optional[float]) -> Optional[PointForce]:
        if not self.axis_lift_enabled():
            return None
        mode = self._axis_lift_mode()
        if mode is None:
            return None
        kind, _label, kg = mode
        if kind == "directional":
            if x_d_mm is None:
                return None
            x_mm = float(x_d_mm)
            label = "P_levante_direccional"
        elif self.is_acoplado:
            x_front = _spin_value_or_none(self.x_front_or_kp)
            if x_front is None:
                return None
            x_mm = float(x_front)
            label = "P_levante_primer_eje"
        else:
            x_mm = float(x_t_mm) - self._first_axle_offset_from_tandem_center()
            label = "P_levante_primer_eje"
        return PointForce(label=label, x_mm=x_mm, value_user=kg)

    def axis_lift_description(self, *, x_t_mm: float, x_d_mm: Optional[float]) -> Optional[str]:
        load = self.axis_lift_load(x_t_mm=x_t_mm, x_d_mm=x_d_mm)
        mode = self._axis_lift_mode()
        if load is None or mode is None:
            return None
        _kind, label, _kg = mode
        return (
            f"{label}: carga puntual descendente automatica de {_fmt_plain(load.value_user, 0)} kg "
            f"en x={_fmt_plain(load.x_mm, 0)} mm. No modifica x_t ni los apoyos; "
            "las reacciones se recalculan con apoyos fijos para evaluar FBD, V(x), M(x) "
            "y verificacion de seccion."
        )

    def _apply_motor_enablement(self):
        uses_dir = self._config_uses_directional()
        if self.is_acoplado:
            uses_dir = False

        self.Rd.setEnabled(uses_dir)
        self.dir_offset.setEnabled(uses_dir)
        if not uses_dir:
            self._set_spin_blank(self.Rd)
            self._set_spin_blank(self.dir_offset)

        if self.is_bitren:
            self.x_rp2_rel.setEnabled(True)
            self.Rp2.setEnabled(True)
        else:
            self.x_rp2_rel.setEnabled(False)
            self.Rp2.setEnabled(False)
        self._apply_axis_lift_ui()

    def _validate_required_inputs(self) -> List[str]:
        errors: List[str] = []

        required_motor: List[Tuple[str, QDoubleSpinBox]] = [
            ("Largo carrozable [mm]", self.Lc),
            ("Posición de perno/tren delantero [mm]", self.x_front_or_kp),
            ("Reacción en perno/tren delantero [Kg]", self.R_front_or_kp),
            ("Reacción en tándem Rt [Kg]", self.Rt),
        ]
        if self._config_uses_directional():
            required_motor.extend([
                ("Reacción en direccional Rd [Kg]", self.Rd),
                ("Offset direccional [mm]", self.dir_offset),
            ])
        if self.is_bitren:
            required_motor.extend([
                ("x_Rp2 relativo a L [mm]", self.x_rp2_rel),
                ("Rp2 [Kg]", self.Rp2),
            ])

        for label, sp in required_motor:
            v = _spin_value_or_none(sp)
            if v is None:
                errors.append(f"Complete el campo: {label}.")

        lc = _spin_value_or_none(self.Lc)
        if lc is not None and lc <= 0:
            errors.append("El largo carrozable debe ser mayor a 0.")

        _points, _dists, _moms, load_errors = self._build_loads_from_table()
        errors.extend(load_errors)

        return errors

    def _set_item_editable(self, row: int, col: int, editable: bool):
        it = self.tbl.item(row, col)
        if it is None:
            was_blocked = self.tbl.blockSignals(True)
            try:
                _set_item(self.tbl, row, col, "")
            finally:
                self.tbl.blockSignals(was_blocked)
            it = self.tbl.item(row, col)
        if it is None:
            return
        flags = it.flags()
        if editable:
            flags |= Qt.ItemIsEditable
        else:
            flags &= ~Qt.ItemIsEditable
        it.setFlags(flags)
        bg = TABLE_INPUT_BG if editable else TABLE_READONLY_BG
        it.setBackground(QBrush(QColor(bg)))
        it.setForeground(QBrush(QColor(TABLE_TEXT_COLOR)))

    def _type_combo(self, row: int) -> QComboBox:
        cmb = self.tbl.cellWidget(row, self.COL_TYPE)
        if not isinstance(cmb, QComboBox):
            raise RuntimeError("Fila de cargas sin combo de tipo.")
        return cmb

    def _row_of_combo(self, combo: QComboBox) -> int:
        for row in range(self.tbl.rowCount()):
            if self.tbl.cellWidget(row, self.COL_TYPE) is combo:
                return row
        return -1

    def _refresh_row_mode(self, row: int):
        load_type = self._type_combo(row).currentText()
        self._set_item_editable(row, self.COL_MAG, True)
        self._set_item_editable(row, self.COL_POS, True)
        editable_len = load_type == "Distribuida"
        self._set_item_editable(row, self.COL_LEN, editable_len)
        if not editable_len:
            self.tbl.blockSignals(True)
            _set_item(self.tbl, row, self.COL_LEN, "")
            self.tbl.blockSignals(False)

    def _on_type_changed(self, combo: QComboBox):
        row = self._row_of_combo(combo)
        if row < 0:
            return
        self._refresh_row_mode(row)

    def _add_load_row(self, *, load_type: str = "Puntual"):
        row = self.tbl.rowCount()
        self.tbl.insertRow(row)
        cmb = QComboBox()
        cmb.addItems(self.LOAD_TYPES)
        cmb.setCurrentText(load_type if load_type in self.LOAD_TYPES else "Puntual")
        cmb.setStyleSheet(combo_cell_style(TABLE_INPUT_BG))
        cmb.currentTextChanged.connect(lambda *_args, c=cmb: self._on_type_changed(c))
        self.tbl.setCellWidget(row, self.COL_TYPE, cmb)
        _set_item(self.tbl, row, self.COL_MAG, "")
        _set_item(self.tbl, row, self.COL_POS, "")
        _set_item(self.tbl, row, self.COL_LEN, "")
        self._refresh_row_mode(row)
        self.tbl.setCurrentCell(row, self.COL_MAG)

    def _remove_selected_rows(self):
        rows = sorted({idx.row() for idx in self.tbl.selectedIndexes()}, reverse=True)
        for row in rows:
            self.tbl.removeRow(row)

    def _load_row_dicts(self) -> List[Dict[str, str]]:
        rows: List[Dict[str, str]] = []
        for row in range(self.tbl.rowCount()):
            rows.append(
                {
                    "type": self._type_combo(row).currentText(),
                    "magnitude": _get_text(self.tbl, row, self.COL_MAG),
                    "position": _get_text(self.tbl, row, self.COL_POS),
                    "length": _get_text(self.tbl, row, self.COL_LEN),
                }
            )
        return rows

    def _set_load_rows(self, rows: Any) -> None:
        self.tbl.blockSignals(True)
        try:
            self.tbl.setRowCount(0)
            if not isinstance(rows, list):
                return
            for load in rows:
                if not isinstance(load, dict):
                    continue
                self._add_load_row(load_type=str(load.get("type", "Puntual")))
                row = self.tbl.rowCount() - 1
                _set_item(self.tbl, row, self.COL_MAG, str(load.get("magnitude", "")))
                _set_item(self.tbl, row, self.COL_POS, str(load.get("position", "")))
                _set_item(self.tbl, row, self.COL_LEN, str(load.get("length", "")))
                self._refresh_row_mode(row)
        finally:
            self.tbl.blockSignals(False)

    def reset_tab_inputs(self):
        self.tbl.setRowCount(0)
        if (not self.is_acoplado) and (not self.is_bitren):
            self.cmb_semi_tipo.setCurrentIndex(0)
        self._populate_configs()
        self._clear_motor_inputs()
        self.set_note("(sin notas)")
        self.clear_deflection_summary()
        self.set_view_mode("inputs")
        self.set_cache(None)
        self.set_diag(None)

    # -------------------
    # Configs por tab
    # -------------------
    def _populate_configs(self):
        self.cmb_config.blockSignals(True)
        self.cmb_config.clear()

        if self.is_acoplado:
            self.cmb_config.addItems([
                "2 ejes — 9200 / 9200",
                "3 ejes — 9200 / 15800",
                "4 ejes conv — 15800 / 15800",
                "4 ejes neum — 16700 / 16700",
            ])
            self.cmb_config.setCurrentIndex(0)

        elif self.is_bitren:
            self.cmb_config.addItems([
                "3 ejes — Rt 22200",
            ])
            self.cmb_config.setCurrentIndex(0)

        else:
            # Semirremolque: Escalado o Estándar
            tipo = self.cmb_semi_tipo.currentText().strip()
            if tipo == "Escalado":
                self.cmb_config.addItems([
                    "2 ejes — Rt 15800",
                    "3 ejes conv — Rt 22200",
                    "3 ejes neum — Rt 23475",
                    "1 + 2 ejes — Rd 9200 (offset 3075) + Rt 15800",
                    "1 + 1 + 1 ejes — Rd 9200 (offset 2450) + Rt 18800",
                    "1 + 3 ejes — Rd 9200 (offset 3700) + Rt 22200",
                ])
                self.cmb_config.setCurrentIndex(0)
            else:
                self.cmb_config.addItems([
                    "1 eje — Rt 9400",
                    "2 ejes — Rt 15800",
                    "3 ejes conv — Rt 22200",
                    "3 ejes neum — Rt 23475",
                    "1 + 2 ejes — Rd 9200 (offset 3075) + Rt 15800",
                    "1 + 1 + 1 ejes — Rd 9200 (offset 2450) + Rt 18800",
                ])
                self.cmb_config.setCurrentIndex(0)

        self.cmb_config.blockSignals(False)
        self._apply_config_defaults()

    def _apply_config_defaults(self):
        cfg = self.cmb_config.currentText()

        if self.is_acoplado:
            if cfg.startswith("2 ejes"):
                self.R_front_or_kp.setValue(9200.0)
                self.Rt.setValue(9200.0)
            elif cfg.startswith("3 ejes"):
                self.R_front_or_kp.setValue(9200.0)
                self.Rt.setValue(15800.0)
            elif cfg.startswith("4 ejes conv"):
                self.R_front_or_kp.setValue(15800.0)
                self.Rt.setValue(15800.0)
            elif cfg.startswith("4 ejes neum"):
                self.R_front_or_kp.setValue(16700.0)
                self.Rt.setValue(16700.0)
            self.Rd.setValue(0.0)

        elif self.is_bitren:
            self.R_front_or_kp.setValue(14500.0)
            self.Rt.setValue(22200.0)
            self.Rd.setValue(0.0)

        else:
            tipo = self.cmb_semi_tipo.currentText().strip()
            compact_cfg = cfg.replace(" ", "")
            self.R_front_or_kp.setValue(15000.0 if tipo == "Escalado" else 9000.0)

            if "1+2" in compact_cfg or "1+1+1" in compact_cfg:
                if "1+1+1" in compact_cfg:
                    self.Rd.setValue(9200.0)
                    self.dir_offset.setValue(2450.0)
                    self.Rt.setValue(18800.0)
                else:
                    self.Rd.setValue(9200.0)
                    self.dir_offset.setValue(3075.0)
                    self.Rt.setValue(15800.0)
            elif "1+3" in compact_cfg:
                self.Rd.setValue(9200.0)
                self.dir_offset.setValue(3700.0)
                self.Rt.setValue(22200.0)
            else:
                self.Rd.setValue(0.0)
                if "Rt 23475" in cfg:
                    self.Rt.setValue(23475.0)
                elif "Rt 22200" in cfg:
                    self.Rt.setValue(22200.0)
                elif "Rt 15800" in cfg:
                    self.Rt.setValue(15800.0)
                elif "Rt 9400" in cfg:
                    self.Rt.setValue(9400.0)
                elif "Rt 9200" in cfg:
                    self.Rt.setValue(9200.0)

        self._apply_motor_enablement()

    # ---- parsing entradas
    def _build_loads_from_table(self) -> Tuple[List[PointForce], List[DistUniform], List[PointMoment], List[str]]:
        points: List[PointForce] = []
        dists: List[DistUniform] = []
        moms: List[PointMoment] = []
        errors: List[str] = []
        dist_rows: List[int] = []
        dist_intervals: List[Tuple[float, float]] = []
        Lc = _spin_value_or_none(self.Lc)
        beam_L = float(Lc) if Lc is not None else None
        p_count = 0
        d_count = 0
        m_count = 0

        for row in range(self.tbl.rowCount()):
            load_type = self._type_combo(row).currentText()
            mag = _try_float(_get_text(self.tbl, row, self.COL_MAG))
            pos = _try_float(_get_text(self.tbl, row, self.COL_POS))
            length = _try_float(_get_text(self.tbl, row, self.COL_LEN))

            if mag is None or pos is None:
                errors.append(f"Cargas fila {row + 1}: complete magnitud y posición.")
                continue
            if beam_L is not None and not (0.0 <= float(pos) <= beam_L):
                errors.append(f"Cargas fila {row + 1}: la posición debe estar dentro de [0, L].")
                continue

            if load_type == "Puntual":
                p_count += 1
                points.append(PointForce(label=f"P{p_count}", x_mm=float(pos), value_user=float(mag)))
            elif load_type == "Momento":
                m_count += 1
                moms.append(PointMoment(label=f"M{m_count}", x_mm=float(pos), M_user_kgmm=float(mag)))
            elif load_type == "Distribuida":
                if length is None or length <= 0.0:
                    errors.append(f"Cargas fila {row + 1}: la distribuida requiere longitud > 0.")
                    continue
                try:
                    x1, x2 = dist_interval(float(pos), float(length))
                except Exception as exc:
                    errors.append(f"Cargas fila {row + 1}: {exc}")
                    continue
                if beam_L is not None and (x1 < 0.0 or x2 > beam_L):
                    errors.append(f"Cargas fila {row + 1}: el tramo distribuido debe quedar dentro de [0, L].")
                    continue
                d_count += 1
                dist_rows.append(row)
                dist_intervals.append((x1, x2))
                dists.append(DistUniform(label=f"q{d_count}", x0_mm=x1, Lq_mm=float(length), q_user=float(mag) / float(length)))

        ok, pairs = check_no_overlap(dist_intervals)
        if not ok:
            for idx_i, idx_j in pairs:
                r1 = dist_rows[idx_i] + 1 if idx_i < len(dist_rows) else "?"
                r2 = dist_rows[idx_j] + 1 if idx_j < len(dist_rows) else "?"
                errors.append(f"Cargas distribuidas solapadas entre filas {r1} y {r2}.")

        return points, dists, moms, errors

    def parse_inputs(self) -> Tuple[Beam, List[PointForce], List[DistUniform], List[PointMoment], List[str]]:
        notes: List[str] = []
        Lc = _spin_value_or_none(self.Lc)
        if Lc is None:
            raise ValueError("VALIDACION: complete 'Largo carrozable [mm]'.")
        beam = Beam(L_mm=Lc)
        points, dists, moms, load_errors = self._build_loads_from_table()
        if load_errors:
            notes.extend(load_errors)
        return beam, points, dists, moms, notes

    def set_cache(self, cache: Optional[SessionCache]):
        self._cached = cache

    def get_cache(self) -> Optional[SessionCache]:
        return self._cached

    def set_note(self, text: str):
        self.note_label.setText(text)

    def set_diag(self, diag):
        self._last_diag = diag
        self.section_panel.set_beam_context(
            largo_viga_mm=_spin_value_or_none(self.Lc),
            posicion_perno_mm=_spin_value_or_none(self.x_front_or_kp),
        )
        if diag is None:
            self.section_panel.set_moment_provider(None)
            self.section_panel.set_shear_provider(None)
            self.section_panel.set_deflection_context(None)
            self.section_panel.clear_results_only(clear_moments_if_no_provider=True)
            self.clear_deflection_summary()
        else:
            self.section_panel.set_moment_provider(lambda x_mm: float(diag.eval_M(float(x_mm))) / 10.0)
            self.section_panel.set_shear_provider(lambda x_mm: float(diag.eval_V(float(x_mm))))
            self.section_panel.clear_results_only()
            self.section_panel.refresh_results_from_context()

    def get_diag(self):
        return self._last_diag

    def set_view_mode(self, mode: str):
        self._view_mode = "solved" if str(mode).strip().lower() == "solved" else "inputs"

    def view_mode(self) -> str:
        return self._view_mode

    def deflection_enabled(self) -> bool:
        return bool(self.chk_show_deflection.isChecked())

    def deflection_params(self) -> Optional[float]:
        return 2.1e4

    def deflection_supports(self) -> Optional[Tuple[float, float]]:
        cache = self.get_cache()
        if cache is None:
            return None
        return cache.deflection_supports

    def set_deflection_summary(self, text: str, *, ok: Optional[bool] = None):
        color = "#1F1F1F"
        if ok is True:
            color = "#0A7F2E"
        elif ok is False:
            color = "#B00020"
        self.lbl_deflection.setStyleSheet(f"color: {color};")
        self.lbl_deflection.setText(text)

    def clear_deflection_summary(self, text: str = "Convexidad L/2: +30 mm\nvmin total: -\nUtilizado: - / 60 mm\nEstado: -"):
        self.set_deflection_summary(text, ok=None)

    def export_state(self) -> Dict[str, Any]:
        load_rows = self._load_row_dicts()

        legacy_points: List[List[str]] = []
        legacy_dists: List[List[str]] = []
        legacy_moms: List[List[str]] = []
        p_count = 0
        q_count = 0
        m_count = 0
        for load in load_rows:
            load_type = load.get("type", "Puntual")
            mag = load.get("magnitude", "")
            pos = load.get("position", "")
            length = load.get("length", "")
            if load_type == "Puntual":
                p_count += 1
                legacy_points.append([f"P{p_count}", pos, mag])
            elif load_type == "Distribuida":
                q_count += 1
                total = _try_float(mag)
                Lq = _try_float(length)
                center = _try_float(pos)
                q_text = ""
                x0_text = pos
                if total is not None and Lq is not None and Lq > 0.0:
                    q_text = _fmt_plain(float(total) / float(Lq), 6)
                if center is not None and Lq is not None:
                    x0_text = _fmt_plain(float(center) - (float(Lq) / 2.0), 2)
                legacy_dists.append([f"q{q_count}", x0_text, length, q_text])
            elif load_type == "Momento":
                m_count += 1
                legacy_moms.append([f"M{m_count}", pos, mag])

        return {
            "semi_tipo": self.cmb_semi_tipo.currentText() if hasattr(self, "cmb_semi_tipo") else None,
            "config_text": self.cmb_config.currentText(),
            "motor_inputs": {
                "Lc": _spin_text(self.Lc),
                "x_front_or_kp": _spin_text(self.x_front_or_kp),
                "R_front_or_kp": _spin_text(self.R_front_or_kp),
                "Rt": _spin_text(self.Rt),
                "Rd": _spin_text(self.Rd),
                "dir_offset": _spin_text(self.dir_offset),
                "x_rp2_rel": _spin_text(self.x_rp2_rel),
                "Rp2": _spin_text(self.Rp2),
            },
            "show_deflection": bool(self.chk_show_deflection.isChecked()),
            "axis_lift_enabled": self.axis_lift_enabled(),
            "view_mode": self.view_mode(),
            "loads": load_rows,
            "tbl_points": legacy_points,
            "tbl_dists": legacy_dists,
            "tbl_moms": legacy_moms,
            "section_panel": self.section_panel.export_state(),
        }

    def import_state(self, state: Optional[Dict[str, Any]]) -> None:
        if not isinstance(state, dict):
            return

        def _set_combo_text(cmb: QComboBox, text: Any) -> None:
            if text is None:
                return
            idx = cmb.findText(str(text))
            if idx >= 0:
                cmb.setCurrentIndex(idx)

        def _set_spin_from_text(sp: QDoubleSpinBox, text: Any) -> None:
            raw = normalize_decimal_text("" if text is None else str(text), allow_negative=float(sp.minimum()) < 0.0)
            if raw in {"", "-", ".", "-."}:
                sp.setValue(float(sp.minimum()))
                line_edit = sp.lineEdit()
                if line_edit is not None:
                    line_edit.clear()
                return
            value = try_parse_user_float(raw, allow_negative=float(sp.minimum()) < 0.0)
            if value is None:
                line_edit = sp.lineEdit()
                if line_edit is not None:
                    line_edit.setText(raw)
                return
            try:
                sp.setValue(float(value))
            except Exception:
                pass

        def _legacy_load_rows(state_in: Dict[str, Any]) -> List[Dict[str, str]]:
            rows: List[Dict[str, str]] = []
            for values in state_in.get("tbl_points") or []:
                if not isinstance(values, list):
                    continue
                rows.append({
                    "type": "Puntual",
                    "magnitude": str(values[2]) if len(values) > 2 else "",
                    "position": str(values[1]) if len(values) > 1 else "",
                    "length": "",
                })
            for values in state_in.get("tbl_dists") or []:
                if not isinstance(values, list):
                    continue
                x0 = _try_float(str(values[1])) if len(values) > 1 else None
                Lq = _try_float(str(values[2])) if len(values) > 2 else None
                q = _try_float(str(values[3])) if len(values) > 3 else None
                position = str(values[1]) if len(values) > 1 else ""
                magnitude = str(values[3]) if len(values) > 3 else ""
                length = str(values[2]) if len(values) > 2 else ""
                if x0 is not None and Lq is not None:
                    position = _fmt_plain(float(x0) + (float(Lq) / 2.0), 2)
                if q is not None and Lq is not None:
                    magnitude = _fmt_plain(float(q) * float(Lq), 2)
                rows.append({
                    "type": "Distribuida",
                    "magnitude": magnitude,
                    "position": position,
                    "length": length,
                })
            for values in state_in.get("tbl_moms") or []:
                if not isinstance(values, list):
                    continue
                rows.append({
                    "type": "Momento",
                    "magnitude": str(values[2]) if len(values) > 2 else "",
                    "position": str(values[1]) if len(values) > 1 else "",
                    "length": "",
                })
            return rows

        semi_tipo = state.get("semi_tipo")
        if hasattr(self, "cmb_semi_tipo") and semi_tipo is not None:
            _set_combo_text(self.cmb_semi_tipo, semi_tipo)
            self._populate_configs()

        _set_combo_text(self.cmb_config, state.get("config_text"))

        motor_inputs = state.get("motor_inputs")
        if isinstance(motor_inputs, dict):
            for key, spin in (
                ("Lc", self.Lc),
                ("x_front_or_kp", self.x_front_or_kp),
                ("R_front_or_kp", self.R_front_or_kp),
                ("Rt", self.Rt),
                ("Rd", self.Rd),
                ("dir_offset", self.dir_offset),
                ("x_rp2_rel", self.x_rp2_rel),
                ("Rp2", self.Rp2),
            ):
                _set_spin_from_text(spin, motor_inputs.get(key))

        self.chk_show_deflection.setChecked(bool(state.get("show_deflection", True)))
        self.chk_axis_lift.setChecked(bool(state.get("axis_lift_enabled", False)) and self.chk_axis_lift.isEnabled())
        loads = state.get("loads")
        self._set_load_rows(loads if isinstance(loads, list) else _legacy_load_rows(state))
        self.section_panel.import_state(state.get("section_panel"))
        self.set_view_mode(str(state.get("view_mode", "inputs")))
        self.set_cache(None)
        self.set_diag(None)
        self.set_note("(sin notas)")


# ============================================================
# MAIN WINDOW
# ============================================================
class FBDApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self._current_study_path: Optional[str] = None
        self._diagram_hover: Optional[DiagramHoverInspector] = None
        self._update_window_title()
        self.resize(1500, 850)
        try:
            self.setWindowIcon(QIcon(ensure_calculeitor_icon()))
        except Exception:
            pass
        self._build_about_menu()

        root = QWidget()
        self.setCentralWidget(root)
        main = QHBoxLayout(root)
        main.setContentsMargins(8, 8, 8, 8)

        splitter = QSplitter(Qt.Horizontal)
        splitter.setChildrenCollapsible(False)
        splitter.setHandleWidth(8)
        main.addWidget(splitter)

        # LEFT
        left_host = QWidget()
        left_lay = QVBoxLayout(left_host)
        left_lay.setContentsMargins(0, 0, 0, 0)

        self.tabs = QTabWidget()
        self.tab_acoplado = UnitTab("Acoplado", is_acoplado=True, is_bitren=False)
        self.tab_semi = UnitTab("Semirremolque", is_acoplado=False, is_bitren=False)
        self.tab_bitren = UnitTab("Bitren - primera especie", is_acoplado=False, is_bitren=True)
        self.tab_reactions = SemiTrailerReactionsTab()

        self.tabs.addTab(self.tab_acoplado, "Acoplado")
        self.tabs.addTab(self.tab_semi, "Semirremolque")
        self.tabs.addTab(self.tab_bitren, "Bitren")
        self.tabs.addTab(self.tab_reactions, "Cálculo y verificación")

        left_lay.addWidget(self.tabs)
        left_host.setMinimumWidth(380)
        splitter.addWidget(left_host)

        # RIGHT plots
        right_container = QWidget()
        right = QVBoxLayout(right_container)
        right.setContentsMargins(0, 0, 0, 0)

        self.fig = plt.Figure()
        gs = self.fig.add_gridspec(4, 1, height_ratios=[1.35, 1.0, 1.0, 1.0], hspace=0.60)
        self.ax_fbd = self.fig.add_subplot(gs[0, 0])
        self.ax_V = self.fig.add_subplot(gs[1, 0], sharex=self.ax_fbd)
        self.ax_M = self.fig.add_subplot(gs[2, 0], sharex=self.ax_fbd)
        self.ax_defl = self.fig.add_subplot(gs[3, 0], sharex=self.ax_fbd)
        self.canvas = FigureCanvas(self.fig)
        self.canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.canvas.setMinimumHeight(980)

        self.plot_scroll = QScrollArea()
        self.plot_scroll.setWidgetResizable(True)
        self.plot_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.plot_scroll.setWidget(self.canvas)

        btn_row = QHBoxLayout()
        self.btn_save_study = QPushButton("Guardar estudio")
        self.btn_load_study = QPushButton("Cargar estudio")
        self.btn_export_plots = QPushButton("Exportar gráficos (FBD, V(x), M(x), deformada)")
        self.btn_export_memoria_docx = QPushButton("Exportar memoria de cálculo")
        btn_row.addWidget(self.btn_save_study)
        btn_row.addWidget(self.btn_load_study)
        btn_row.addWidget(self.btn_export_plots)
        btn_row.addWidget(self.btn_export_memoria_docx)
        btn_row.addStretch(1)

        right.addLayout(btn_row)
        right.addWidget(self.plot_scroll)
        splitter.addWidget(right_container)

        splitter.setSizes([380, 1120])
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)

        # timers / signals
        self._redraw_timer = QTimer(self)
        self._redraw_timer.setSingleShot(True)
        self._redraw_timer.timeout.connect(self._replot_active_tab)

        self.canvas.installEventFilter(self)
        for tab in [self.tab_acoplado, self.tab_semi, self.tab_bitren]:
            tab.btn_solve.clicked.connect(lambda _=False, t=tab: self._solve_for_tab(t))
            tab.btn_clear.clicked.connect(lambda _=False, t=tab: self._plot_inputs_for_tab(t))
            tab.btn_reset.clicked.connect(lambda _=False, t=tab: self._reset_tab(t))

            def wire(sp, t=tab):
                sp.valueChanged.connect(lambda *_: self._schedule_replot_tab(t, reset_solution=True))

            wire(tab.Lc)
            wire(tab.x_front_or_kp)
            wire(tab.R_front_or_kp)
            wire(tab.Rt)

            if not tab.is_acoplado:
                wire(tab.Rd)
                wire(tab.dir_offset)
            if tab.is_bitren:
                wire(tab.x_rp2_rel)
                wire(tab.Rp2)

            tab.cmb_config.currentIndexChanged.connect(lambda *_, t=tab: self._schedule_replot_tab(t, reset_solution=True))
            if hasattr(tab, "cmb_semi_tipo"):
                tab.cmb_semi_tipo.currentIndexChanged.connect(lambda *_, t=tab: self._schedule_replot_tab(t, reset_solution=True))

            tab.tbl.cellChanged.connect(lambda *_, t=tab: self._schedule_replot_tab(t, reset_solution=True))
            tab.chk_axis_lift.toggled.connect(lambda *_, t=tab: self._schedule_axis_lift_replot(t))
            tab.chk_show_deflection.toggled.connect(lambda *_, t=tab: self._schedule_replot_tab(t, reset_solution=False))
            tab.section_panel.inertia_inputs_changed.connect(lambda t=tab: self._schedule_replot_tab(t, reset_solution=False))

        self.tab_reactions.plot_data_changed.connect(self._schedule_active_replot)
        self.tab_reactions.section_panel.inertia_inputs_changed.connect(self._schedule_active_replot)

        self.tabs.currentChanged.connect(lambda _i: self._on_tab_changed())
        self.btn_save_study.clicked.connect(self._save_study)
        self.btn_load_study.clicked.connect(self._load_study)
        self.btn_export_plots.clicked.connect(self._export_plots_jpg_1200)
        self.btn_export_memoria_docx.clicked.connect(self._export_memoria_docx)

        self.active_tab().set_note("Complete las entradas para visualizar y resolver.")
        self._clear_plot_canvas("Sin resultados. Complete datos y presione Resolver equilibrio.")
        self._update_export_buttons()

        self._resize_timer = QTimer(self)
        self._resize_timer.setSingleShot(True)
        self._resize_timer.timeout.connect(self._replot_active_tab)
        self.canvas.mpl_connect("resize_event", lambda evt: self._resize_timer.start(80))

    def _build_about_menu(self) -> None:
        help_menu = self.menuBar().addMenu("Ayuda")
        about_action = QAction("Acerca de Calculeitor", self)
        about_action.triggered.connect(self._show_about)
        help_menu.addAction(about_action)

    def _show_about(self) -> None:
        QMessageBox.about(
            self,
            "Acerca de Calculeitor",
            (
                f"Calculeitor\nVersión {APP_VERSION}\n\n"
                "Aplicación de escritorio para cálculo de vigas de acoplados, "
                "semirremolques y bitrenes.\n\n"
                "Incluye equilibrio, reacciones, diagramas V/M, deformada, "
                "verificación de sección, estudios .sbeam y memoria de cálculo DOCX."
            ),
        )

    def _update_window_title(self) -> None:
        if self._current_study_path:
            study_name = Path(self._current_study_path).stem
            self.setWindowTitle(f"{APP_TITLE_BASE} — {study_name}")
        else:
            self.setWindowTitle(APP_TITLE_BASE)

    def _disconnect_diagram_hover(self) -> None:
        if self._diagram_hover is not None:
            self._diagram_hover.disconnect()
        self._diagram_hover = None

    def _hide_diagram_hover(self) -> None:
        if self._diagram_hover is not None:
            self._diagram_hover.hide()

    def _setup_diagram_hover(self, line_v=None, line_m=None, line_def=None) -> None:
        self._disconnect_diagram_hover()
        curves = []
        if line_v is not None:
            curves.append(
                HoverCurve(
                    ax=self.ax_V,
                    line=line_v,
                    label="V",
                    x_unit="mm",
                    y_unit="kg",
                    y_display_scale=1.0,
                    x_decimals=0,
                    y_decimals=0,
                )
            )
        if line_m is not None:
            curves.append(
                HoverCurve(
                    ax=self.ax_M,
                    line=line_m,
                    label="M",
                    x_unit="mm",
                    y_unit="kg·cm",
                    y_display_scale=0.1,
                    x_decimals=0,
                    y_decimals=0,
                )
            )
        if line_def is not None:
            curves.append(
                HoverCurve(
                    ax=self.ax_defl,
                    line=line_def,
                    label="δ",
                    x_unit="mm",
                    y_unit="mm",
                    y_display_scale=1.0,
                    x_decimals=0,
                    y_decimals=1,
                )
            )
        if curves:
            self._diagram_hover = DiagramHoverInspector(self.canvas, curves)

    def active_tab(self):
        return self.tabs.currentWidget()

    def _on_tab_changed(self):
        self._update_export_buttons()
        self._replot_active_tab()

    def _update_export_buttons(self):
        self.btn_export_memoria_docx.setEnabled(True)

    def _export_study_state(self) -> Dict[str, Any]:
        return {
            "version": 1,
            "saved_at": datetime.now().isoformat(timespec="seconds"),
            "active_tab_index": int(self.tabs.currentIndex()),
            "tabs": {
                "acoplado": self.tab_acoplado.export_state(),
                "semirremolque": self.tab_semi.export_state(),
                "bitren": self.tab_bitren.export_state(),
                "reacciones": self.tab_reactions.export_state(),
            },
        }

    def _apply_study_state(self, state: Dict[str, Any]) -> None:
        tabs = state.get("tabs")
        if not isinstance(tabs, dict):
            raise ValueError("El archivo no contiene pestaÃ±as de estudio vÃ¡lidas.")

        self.tab_acoplado.import_state(tabs.get("acoplado"))
        self.tab_semi.import_state(tabs.get("semirremolque"))
        self.tab_bitren.import_state(tabs.get("bitren"))
        self.tab_reactions.import_state(tabs.get("reacciones"))

        active_tab_index = state.get("active_tab_index", 0)
        try:
            active_tab_index = int(active_tab_index)
        except Exception:
            active_tab_index = 0
        active_tab_index = max(0, min(active_tab_index, self.tabs.count() - 1))
        self.tabs.setCurrentIndex(active_tab_index)
        self._replot_active_tab()

    def _save_study(self):
        default_name = f"estudio_{datetime.now().strftime('%Y%m%d_%H%M%S')}.sbeam"
        path, _ = QFileDialog.getSaveFileName(self, "Guardar estudio", default_name, "Estudio Semi Beam (*.sbeam)")
        if not path:
            return
        if not path.lower().endswith(".sbeam"):
            path += ".sbeam"
        try:
            save_study_file(path, self._export_study_state())
            self._current_study_path = path
            self._update_window_title()
            QMessageBox.information(self, "Guardar estudio", f"Estudio guardado en:\n{path}")
        except Exception as e:
            QMessageBox.critical(self, "Guardar estudio", f"No se pudo guardar el estudio: {e}")

    def _load_study(self):
        path, _ = QFileDialog.getOpenFileName(self, "Cargar estudio", "", "Estudio Semi Beam (*.sbeam)")
        if not path:
            return
        try:
            state = load_study_file(path)
            self._apply_study_state(state)
            self._current_study_path = path
            self._update_window_title()
            QMessageBox.information(self, "Cargar estudio", f"Estudio cargado desde:\n{path}")
        except Exception as e:
            QMessageBox.critical(self, "Cargar estudio", f"No se pudo cargar el estudio: {e}")

    def _schedule_active_replot(self):
        self._redraw_timer.start(60)

    def eventFilter(self, obj, event):
        if obj is self.canvas and isinstance(event, QWheelEvent):
            bar = self.plot_scroll.verticalScrollBar()
            bar.setValue(bar.value() - event.angleDelta().y())
            return True
        return super().eventFilter(obj, event)

    def _clear_plot_canvas(self, subtitle: str = ""):
        self._disconnect_diagram_hover()
        for ax in (self.ax_fbd, self.ax_V, self.ax_M, self.ax_defl):
            ax.clear()
            ax.set_axis_off()
        if subtitle:
            self.ax_fbd.text(0.5, 0.5, subtitle, ha="center", va="center", transform=self.ax_fbd.transAxes)
        self.canvas.draw_idle()

    def _reset_tab(self, tab: UnitTab):
        tab.reset_tab_inputs()
        if tab is self.active_tab():
            self._clear_plot_canvas("Tab reiniciado. Complete entradas para resolver.")

    def _schedule_replot_tab(self, tab: UnitTab, *, reset_solution: bool):
        if reset_solution:
            tab.set_view_mode("inputs")
            tab.set_cache(None)
            tab.set_diag(None)
        if tab is self.active_tab():
            self._redraw_timer.start(90)

    def _schedule_axis_lift_replot(self, tab: UnitTab):
        keep_solved = tab.view_mode() == "solved"
        tab.set_cache(None)
        tab.set_diag(None)
        tab.set_view_mode("solved" if keep_solved else "inputs")
        if tab is self.active_tab():
            self._redraw_timer.start(90)

    def _current_style(self) -> RenderStyle:
        return RenderStyle()

    # -------------------------
    # Extremos locales (M en kg·cm)
    # -------------------------
    def _find_local_extrema_kgcm(self, diag, xlim: Tuple[float, float]):
        import numpy as np

        x, _, M = diag.sample(n_per_segment=220)  # M kg·mm
        x = np.asarray(x, dtype=float)
        M = np.asarray(M, dtype=float) / 10.0     # -> kg·cm

        mask = (x >= xlim[0]) & (x <= xlim[1])
        x = x[mask]
        M = M[mask]
        if len(x) < 6:
            return [], []

        d = np.diff(M)
        eps = 1e-8 * max(1.0, float(np.max(np.abs(M))))

        s = np.sign(d)
        for i in range(len(s)):
            if abs(s[i]) < 1e-12:
                s[i] = s[i - 1] if i > 0 else 0.0

        maxs, mins = [], []
        for i in range(1, len(s)):
            if s[i - 1] > 0 and s[i] < 0:
                if abs(M[i]) > eps:
                    maxs.append((x[i], M[i]))
            if s[i - 1] < 0 and s[i] > 0:
                if abs(M[i]) > eps:
                    mins.append((x[i], M[i]))

        def dedup(lst, dx=10.0):
            out = []
            for xi, mi in lst:
                if not out or abs(xi - out[-1][0]) > dx:
                    out.append((xi, mi))
            return out

        return dedup(maxs), dedup(mins)

    def _equilibrium_diagnostics(
        self,
        *,
        point_forces: List[PointForce],
        dist_loads: List[DistUniform],
        moments: List[PointMoment],
        residual_fy: float,
        residual_m0: float,
    ) -> Dict[str, Any]:
        fy_terms: List[Tuple[str, float]] = []
        m0_terms: List[Tuple[str, float]] = []

        for pf in point_forces:
            fy_i = float(to_internal_Fy(pf.label, pf.value_user))
            fy_terms.append((pf.label, abs(fy_i)))
            m0_terms.append((f"{pf.label}·x", abs(fy_i * float(pf.x_mm))))

        for dl in dist_loads:
            w_up = float(to_internal_w_up(dl.label, dl.q_user))
            fres = w_up * float(dl.Lq_mm)
            x_cent = float(dl.x0_mm) + 0.5 * float(dl.Lq_mm)
            fy_terms.append((f"{dl.label} (res)", abs(fres)))
            m0_terms.append((f"{dl.label} (res)·x", abs(fres * x_cent)))

        for pm in moments:
            m0_terms.append((pm.label, abs(float(pm.M_user_kgmm))))

        ref_fy = max(sum(v for _, v in fy_terms), 1.0)
        ref_m0 = max(sum(v for _, v in m0_terms), 1.0)

        err_fy_pct = (abs(float(residual_fy)) / ref_fy) * 100.0
        err_m0_pct = (abs(float(residual_m0)) / ref_m0) * 100.0
        status = "OK" if max(err_fy_pct, err_m0_pct) <= 2.0 else "WARNING"

        top_terms = sorted(m0_terms, key=lambda kv: kv[1], reverse=True)[:3]
        return {
            "status": status,
            "ref_fy": ref_fy,
            "ref_m0": ref_m0,
            "err_fy_pct": err_fy_pct,
            "err_m0_pct": err_m0_pct,
            "top_terms": top_terms,
        }

    def _solve_final_equilibrium_with_axis_lift(
        self,
        *,
        tab: UnitTab,
        case: BeamCase,
        base_result: EquilibriumResult,
        beam_L_mm: float,
    ) -> Tuple[EquilibriumResult, Optional[PointForce], Optional[ReactionsResult]]:
        lift_load = tab.axis_lift_load(x_t_mm=float(base_result.x_t_mm), x_d_mm=base_result.x_d_mm)
        if lift_load is None:
            return base_result, None, None

        external_points = list(case.point_forces) + [lift_load]
        final_loads = [*external_points, *base_result.solved_dist_loads, *base_result.solved_moments]

        x_k = float(case.kingpin.x_mm)
        x_t = float(base_result.x_t_mm)
        reaction_result = solve_reactions_2support(float(beam_L_mm), (x_k, x_t), final_loads)
        support_points = [
            PointForce(label=case.kingpin.label, x_mm=x_k, value_user=float(reaction_result.reacciones["R_A"])),
            PointForce(label=case.tandem.label, x_mm=x_t, value_user=float(reaction_result.reacciones["R_B"])),
        ]

        notes = list(base_result.notes)
        notes.append(
            "Levante de eje incluido en equilibrio final: se recalcularon reacciones entre apoyo delantero "
            "y tandem con x_t conservado."
        )
        notes.extend(reaction_result.notes)
        final_result = EquilibriumResult(
            x_t_mm=float(base_result.x_t_mm),
            q_user_kg_per_mm=float(base_result.q_user_kg_per_mm),
            x_d_mm=None if base_result.x_d_mm is None else float(base_result.x_d_mm),
            residual_Fy=float(reaction_result.Fy_total_residual),
            residual_M0=float(reaction_result.M0_residual),
            notes=notes,
            solved_point_forces=[*external_points, *support_points],
            solved_dist_loads=list(base_result.solved_dist_loads),
            solved_moments=list(base_result.solved_moments),
        )
        return final_result, lift_load, reaction_result

    def _compute_deflection_result(
        self,
        *,
        diag,
        beam_L_mm: float,
        supports: Optional[Tuple[float, float]],
        params: Optional[float],
        section_panel=None,
    ):
        if supports is None or params is None:
            return None
        xa, xb = float(supports[0]), float(supports[1])
        if not xa < xb:
            return None
        x = np.linspace(0.0, float(beam_L_mm), 600, dtype=float)
        if hasattr(diag, "_eval_M_array"):
            M = diag._eval_M_array(x)
        else:
            M = np.asarray([float(diag.eval_M(float(xi))) for xi in x], dtype=float)
        e_val = float(params)
        i_input = None
        i_source = ""
        if section_panel is not None:
            try:
                i_profile = section_panel.build_inertia_profile_mm4(x)
            except Exception:
                i_profile = None
            if i_profile is not None:
                i_input = i_profile
                i_source = "tabla de secciones"
        if i_input is None:
            return None
        result = compute_total_deflection(x, M, E=e_val, I=i_input, supports=(xa, xb), camber_mid_mm=30.0)
        return result, i_source

    def _deflection_summary_text(self, result, i_source: str) -> str:
        state = "OK" if result.ok else "EXCEDE"
        return (
            "Convexidad L/2: +30 mm\n"
            f"I usado: {i_source}\n"
            f"vmin total: {_fmt_plain(result.vmin_mm, 2)} mm @ x={_fmt_plain(result.x_vmin_mm, 0)} mm\n"
            f"Utilizado: {_fmt_plain(result.utilized_mm, 2)} / {_fmt_plain(result.allowable_mm, 2)} mm\n"
            f"Estado: {state}"
        )

    def _render_deflection_axis(
        self,
        *,
        payload,
        xlim: Tuple[float, float],
        enabled: bool,
        summary_target=None,
        unavailable_text: str = "Deformada desactivada.",
    ):
        if not enabled:
            self.ax_defl.clear()
            self.ax_defl.set_axis_off()
            self.ax_defl.text(0.5, 0.5, unavailable_text, ha="center", va="center", transform=self.ax_defl.transAxes)
            if summary_target is not None:
                summary_target.clear_deflection_summary("Convexidad L/2: +30 mm\nvmin total: -\nUtilizado: - / 60 mm\nEstado: desactivada")
                if hasattr(summary_target, "section_panel"):
                    summary_target.section_panel.set_deflection_context(None)
            return None

        if payload is None:
            self.ax_defl.clear()
            self.ax_defl.set_axis_off()
            self.ax_defl.text(0.5, 0.5, "Deformada no disponible. Complete la tabla de secciones.", ha="center", va="center", transform=self.ax_defl.transAxes)
            if summary_target is not None:
                summary_target.clear_deflection_summary("Convexidad L/2: +30 mm\nvmin total: -\nUtilizado: - / 60 mm\nEstado: complete la tabla de secciones")
                if hasattr(summary_target, "section_panel"):
                    summary_target.section_panel.set_deflection_context(None)
            return None

        result, i_source = payload
        line_def = render_deflection(self.ax_defl, result, y_zoom=1.0, xlim=xlim)
        if summary_target is not None:
            summary_target.set_deflection_summary(self._deflection_summary_text(result, i_source), ok=bool(result.ok))
            if hasattr(summary_target, "section_panel"):
                summary_target.section_panel.set_deflection_context(result, i_source=i_source)
        return line_def

    def _save_axis_snapshot(self, ax, out_path: str, *, dpi: int = MEMORIA_EXPORT_IMAGE_DPI):
        fig = self.fig
        self._hide_diagram_hover()
        self.canvas.draw()
        renderer = fig.canvas.get_renderer()
        bbox = ax.get_tightbbox(renderer).transformed(fig.dpi_scale_trans.inverted())
        fig.savefig(out_path, dpi=int(dpi), bbox_inches=bbox)

    # Plotting
    def _plot_triplet(self, cache: SessionCache, *, set_diag_on_tab: Optional[UnitTab] = None):
        beam_plot, points, dists, moms = cache.beam_plot, cache.points, cache.dists, cache.moms
        xlim = _compute_x_view(beam_plot.L_mm, points, dists, moms)

        data = normalize_inputs(beam_plot, points, dists, moms)
        render_fbd(self.ax_fbd, data, self._current_style(), y_zoom=1.0, xlim=xlim)
        self.ax_fbd.set_xlabel("x [mm]")
        self.ax_fbd.tick_params(labelbottom=True)

        diag = build_V_M(
            beam_L_mm=beam_plot.L_mm,
            point_forces=points,
            dist_loads=dists,
            moments=moms,
            x_start=xlim[0],
            x_end=xlim[1],
        )
        line_v = render_shear(self.ax_V, diag, y_zoom=1.0, xlim=xlim)
        self.ax_V.set_xlabel("x [mm]")
        self.ax_V.tick_params(labelbottom=True)

        line_m = render_moment(self.ax_M, diag, y_zoom=1.0, xlim=xlim)
        self.ax_M.set_xlabel("x [mm]")
        self.ax_M.tick_params(labelbottom=True)

        defl_payload = None
        line_def = None
        if set_diag_on_tab is not None and set_diag_on_tab.deflection_enabled():
            defl_payload = self._compute_deflection_result(
                diag=diag,
                beam_L_mm=beam_plot.L_mm,
                supports=cache.deflection_supports,
                params=set_diag_on_tab.deflection_params(),
                section_panel=set_diag_on_tab.section_panel,
            )
            line_def = self._render_deflection_axis(
                payload=defl_payload,
                xlim=xlim,
                enabled=set_diag_on_tab.deflection_enabled(),
                summary_target=set_diag_on_tab,
                unavailable_text="Deformada desactivada.",
            )
        else:
            line_def = self._render_deflection_axis(
                payload=None,
                xlim=xlim,
                enabled=False,
                summary_target=None,
            )

        self._setup_diagram_hover(line_v=line_v, line_m=line_m, line_def=line_def)
        self.fig.subplots_adjust(left=0.07, right=0.985, top=0.96, bottom=0.06, hspace=0.65)
        self.canvas.draw_idle()

        if set_diag_on_tab is not None:
            set_diag_on_tab.set_diag(diag)

    def _plot_reactions_tab(self, tab: SemiTrailerReactionsTab):
        state = tab.current_plot_state()
        if state is None:
            tab.set_diag(None)
            self._clear_plot_canvas("Cálculo y verificación: complete datos válidos para calcular.")
            return

        xlim = _compute_x_view(state.beam.L_mm, state.point_forces, state.dist_loads, state.moments)
        data = normalize_inputs(state.beam, state.point_forces, state.dist_loads, state.moments)
        render_fbd(self.ax_fbd, data, self._current_style(), y_zoom=1.0, xlim=xlim)
        self.ax_fbd.set_xlabel("x [mm]")
        self.ax_fbd.tick_params(labelbottom=True)

        diag = build_V_M(
            beam_L_mm=state.beam.L_mm,
            point_forces=state.point_forces,
            dist_loads=state.dist_loads,
            moments=state.moments,
            x_start=xlim[0],
            x_end=xlim[1],
        )
        tab.set_diag(diag)

        line_v = None
        line_m = None
        if state.show_vm:
            line_v = render_shear(self.ax_V, diag, y_zoom=1.0, xlim=xlim)
            self.ax_V.set_xlabel("x [mm]")
            self.ax_V.tick_params(labelbottom=True)
            line_m = render_moment(self.ax_M, diag, y_zoom=1.0, xlim=xlim)
            self.ax_M.set_xlabel("x [mm]")
            self.ax_M.tick_params(labelbottom=True)
        else:
            for ax, title in ((self.ax_V, "V(x) desactivado"), (self.ax_M, "M(x) desactivado")):
                ax.clear()
                ax.set_axis_off()
                ax.text(0.5, 0.5, title, ha="center", va="center", transform=ax.transAxes)

        defl_payload = None
        if tab.deflection_enabled():
            defl_payload = self._compute_deflection_result(
                diag=diag,
                beam_L_mm=state.beam.L_mm,
                supports=tab.deflection_supports(),
                params=tab.deflection_params(),
                section_panel=tab.section_panel,
            )
        line_def = self._render_deflection_axis(
            payload=defl_payload,
            xlim=xlim,
            enabled=tab.deflection_enabled(),
            summary_target=tab,
            unavailable_text="Deformada desactivada.",
        )

        self._setup_diagram_hover(line_v=line_v, line_m=line_m, line_def=line_def)
        self.fig.subplots_adjust(left=0.07, right=0.985, top=0.96, bottom=0.06, hspace=0.65)
        self.canvas.draw_idle()

    def _replot_active_tab(self):
        tab = self.active_tab()
        if tab is self.tab_reactions:
            self._plot_reactions_tab(self.tab_reactions)
            return
        cache = tab.get_cache()
        if cache is None:
            if tab.view_mode() == "solved":
                self._solve_for_tab(tab)
            else:
                self._plot_inputs_for_tab(tab)
            return
        self._plot_triplet(cache, set_diag_on_tab=tab)

    def _plot_inputs_for_tab(self, tab: UnitTab):
        try:
            beam, points, dists, moms, notes = tab.parse_inputs()
            note = f"[{tab.title}] Vista: entradas (sin motor). Largo carrozable = {beam.L_mm:g} mm"
            if notes:
                note += "\nNotas:\n- " + "\n- ".join(notes)

            cache = SessionCache(beam_plot=beam, points=points, dists=dists, moms=moms, note_text=note, deflection_supports=None)
            tab.set_view_mode("inputs")
            tab.set_cache(cache)
            tab.set_note(note)
            self._plot_triplet(cache, set_diag_on_tab=tab)

        except Exception as e:
            msg = str(e)
            if msg.startswith("VALIDACION:"):
                tab.set_view_mode("inputs")
                tab.set_cache(None)
                tab.set_note(msg.replace("VALIDACION:", "").strip())
                tab.set_diag(None)
                if tab is self.active_tab():
                    self._clear_plot_canvas(msg.replace("VALIDACION:", "").strip())
                return
            QMessageBox.critical(self, "Error", f"Error al graficar entradas: {e}")
            tab.set_view_mode("inputs")
            tab.set_cache(None)
            tab.set_note(f"Error: {e}")
            tab.set_diag(None)

    # Solve equilibrium
    def _solve_for_tab(self, tab: UnitTab):
        try:
            errors = tab._validate_required_inputs()
            if errors:
                head = "No se puede resolver: hay datos requeridos incompletos.\n"
                body = "\n".join([f"- {e}" for e in errors[:12]])
                if len(errors) > 12:
                    body += f"\n- ... y {len(errors) - 12} más."
                QMessageBox.warning(self, "Validación de entradas", head + body)
                return

            beam_motor, pforces, dloads, moms, pnotes = tab.parse_inputs()
            Lc = float(_spin_value_or_none(tab.Lc))

            kingpin = FixedSupport(
                label="Rp1",
                x_mm=float(_spin_value_or_none(tab.x_front_or_kp)),
                reaction_user=float(_spin_value_or_none(tab.R_front_or_kp))
            )
            tandem = TandemSupport(label="Rt", reaction_user=float(_spin_value_or_none(tab.Rt)))

            directional = None
            if tab._config_uses_directional():
                directional = DirectionalSupport(
                    label="Rd",
                    reaction_user=float(_spin_value_or_none(tab.Rd)),
                    offset_mm=float(_spin_value_or_none(tab.dir_offset))
                )

            hitch = None
            x_rp2_abs = None
            if tab.is_bitren:
                x_rp2_abs = Lc + float(_spin_value_or_none(tab.x_rp2_rel))
                hitch = FixedSupport(label="Rp2", x_mm=x_rp2_abs, reaction_user=float(_spin_value_or_none(tab.Rp2)))

            unknown_q = UnknownUniformLoad(label="q", span_start_mm=0.0, span_len_mm=Lc)

            case = BeamCase(
                beam=beam_motor,
                point_forces=pforces,
                dist_loads=dloads,
                moments=moms,
                kingpin=kingpin,
                tandem=tandem,
                directional=directional,
                hitch=hitch,
                unknown_uniform=unknown_q,
            )

            res = solve_equilibrium(case)

            if tab.is_bitren:
                L_viga_total = float(res.x_t_mm) + 2070.0
            else:
                L_viga_total = float(Lc)

            beam_plot = Beam(L_mm=L_viga_total)
            res, lift_load, reaction_result = self._solve_final_equilibrium_with_axis_lift(
                tab=tab,
                case=case,
                base_result=res,
                beam_L_mm=float(beam_plot.L_mm),
            )

            support_positions = [float(kingpin.x_mm), float(res.x_t_mm)]
            if res.x_d_mm is not None:
                support_positions.append(float(res.x_d_mm))
            if x_rp2_abs is not None:
                support_positions.append(float(x_rp2_abs))

            cache = SessionCache(
                beam_plot=beam_plot,
                points=res.solved_point_forces,
                dists=res.solved_dist_loads,
                moms=res.solved_moments,
                note_text="",
                deflection_supports=(min(support_positions), max(support_positions)),
            )

            xlim = _compute_x_view(beam_plot.L_mm, cache.points, cache.dists, cache.moms)
            diag = build_V_M(
                beam_L_mm=beam_plot.L_mm,
                point_forces=cache.points,
                dist_loads=cache.dists,
                moments=cache.moms,
                x_start=xlim[0],
                x_end=xlim[1],
            )
            maxs, mins = self._find_local_extrema_kgcm(diag, xlim)
            eq_diag = self._equilibrium_diagnostics(
                point_forces=cache.points,
                dist_loads=cache.dists,
                moments=cache.moms,
                residual_fy=float(res.residual_Fy),
                residual_m0=float(res.residual_M0),
            )

            note_lines = [
                f"[{tab.title}] Vista: solución (motor).",
                f"Largo carrozable = {_fmt_plain(Lc, 0)} mm",
                f"Largo viga total = {_fmt_plain(L_viga_total, 0)} mm",
                f"x_t = {_fmt_plain(float(res.x_t_mm), 0)} mm",
                f"q calculada = {_fmt_plain(res.q_user_kg_per_mm, 6)} kg/mm (en [0, L_carrozable])",
            ]
            if tab.is_bitren and x_rp2_abs is not None:
                note_lines.append(f"Bitren: x_Rp2_abs = {_fmt_plain(x_rp2_abs, 0)} mm")
            if res.x_d_mm is not None:
                note_lines.append(f"x_d = {_fmt_plain(res.x_d_mm, 0)} mm")
            lift_desc = tab.axis_lift_description(x_t_mm=float(res.x_t_mm), x_d_mm=res.x_d_mm)
            if lift_desc is not None:
                note_lines.append(lift_desc)
            if reaction_result is not None:
                note_lines.append("Reacciones recalculadas con levante y apoyos fijos:")
                for pf in cache.points:
                    if pf.label in {"Rp1", "Rd", "Rt", "Rp2"}:
                        note_lines.append(
                            f"  {pf.label}: x={_fmt_plain(pf.x_mm, 0)} mm -> R={_fmt_plain(pf.value_user, 2)} kg"
                        )

            if maxs:
                note_lines.append("Máximos locales M(x) [kg·cm]:")
                for xi, mi in maxs:
                    note_lines.append(f"  x={_fmt_plain(xi, 0)} mm -> M={_fmt_plain(mi, 2)} kg·cm")
            if mins:
                note_lines.append("Mínimos locales M(x) [kg·cm]:")
                for xi, mi in mins:
                    note_lines.append(f"  x={_fmt_plain(xi, 0)} mm -> M={_fmt_plain(mi, 2)} kg·cm")

            note_lines.append(f"Residual ΣFy = {_fmt_plain(res.residual_Fy, 6)}")
            note_lines.append(f"Residual ΣM0 = {_fmt_plain(res.residual_M0, 6)}")
            note_lines.append(
                f"Chequeo equilibrio (tol 2%): {eq_diag['status']} | "
                f"err ΣFy={_fmt_plain(eq_diag['err_fy_pct'], 3)}% (ref={_fmt_plain(eq_diag['ref_fy'], 2)}), "
                f"err ΣM0={_fmt_plain(eq_diag['err_m0_pct'], 3)}% (ref={_fmt_plain(eq_diag['ref_m0'], 2)})."
            )
            if eq_diag["status"] == "WARNING":
                note_lines.append("Sugerencia: revisar signos de cargas P/q y valores de reacciones.")
                top_terms = eq_diag.get("top_terms", [])
                if top_terms:
                    note_lines.append("Aportes dominantes (magnitud):")
                    for lbl, val in top_terms:
                        note_lines.append(f"  {lbl}: {_fmt_plain(val, 2)}")

            if pnotes:
                note_lines.append("Notas:")
                for n in pnotes:
                    note_lines.append(f"- {n}")

            note = "\n".join(note_lines)
            cache.note_text = note

            tab.set_view_mode("solved")
            tab.set_cache(cache)
            tab.set_note(note)

            self._plot_triplet(cache, set_diag_on_tab=tab)
            tab.section_panel.refresh_results_from_context()

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Error al resolver equilibrio: {e}")
            tab.set_view_mode("inputs")
            tab.set_cache(None)
            tab.set_note(f"Error: {e}")
            tab.set_diag(None)

    # Exportar gráficos
    def _export_plots_jpg_1200_legacy(self):
        folder = QFileDialog.getExistingDirectory(self, "Seleccionar carpeta destino")
        if not folder:
            return
        try:
            self._hide_diagram_hover()
            self.canvas.draw()
            fig = self.fig
            renderer = fig.canvas.get_renderer()

            path_fbd = f"{folder}/FBD.jpg"
            fig.savefig(
                path_fbd,
                dpi=1200,
                bbox_inches=self.ax_fbd.get_tightbbox(renderer).transformed(fig.dpi_scale_trans.inverted())
            )

            path_v = f"{folder}/V.jpg"
            fig.savefig(
                path_v,
                dpi=1200,
                bbox_inches=self.ax_V.get_tightbbox(renderer).transformed(fig.dpi_scale_trans.inverted())
            )

            path_m = f"{folder}/M.jpg"
            fig.savefig(
                path_m,
                dpi=1200,
                bbox_inches=self.ax_M.get_tightbbox(renderer).transformed(fig.dpi_scale_trans.inverted())
            )

            QMessageBox.information(self, "Exportación", f"Exportado:\n- {path_fbd}\n- {path_v}\n- {path_m}")
        except Exception as e:
            QMessageBox.critical(self, "Exportación", f"Error al exportar: {e}")

    def _export_plots_jpg_1200(self):
        folder = QFileDialog.getExistingDirectory(self, "Seleccionar carpeta destino")
        if not folder:
            return
        try:
            path_fbd = f"{folder}/FBD.jpg"
            path_v = f"{folder}/V.jpg"
            path_m = f"{folder}/M.jpg"
            path_defl = f"{folder}/Deformada.jpg"

            self._save_axis_snapshot(self.ax_fbd, path_fbd, dpi=1200)
            self._save_axis_snapshot(self.ax_V, path_v, dpi=1200)
            self._save_axis_snapshot(self.ax_M, path_m, dpi=1200)
            self._save_axis_snapshot(self.ax_defl, path_defl, dpi=1200)

            QMessageBox.information(
                self,
                "ExportaciÃ³n",
                f"Exportado:\n- {path_fbd}\n- {path_v}\n- {path_m}\n- {path_defl}",
            )
        except Exception as e:
            QMessageBox.critical(self, "ExportaciÃ³n", f"Error al exportar: {e}")

    def _save_text_schematic(self, path: str, title: str, lines: List[str]):
        fig = plt.Figure(figsize=(8, 3.2))
        ax = fig.add_subplot(111)
        ax.axis("off")
        ax.text(0.5, 0.86, title, ha="center", va="center", fontsize=12, fontweight="bold", transform=ax.transAxes)
        y = 0.66
        for ln in lines:
            ax.text(0.08, y, f"• {ln}", ha="left", va="center", fontsize=10, transform=ax.transAxes)
            y -= 0.16
        fig.tight_layout()
        fig.savefig(path, dpi=300)

    def _export_reactions_memoria_docx(self):
        def _extremos_V(diag_obj, xlim_):
            x, V, _ = diag_obj.sample(n_per_segment=220)
            x = np.asarray(x, dtype=float)
            V = np.asarray(V, dtype=float)
            mask = (x >= xlim_[0]) & (x <= xlim_[1])
            x = x[mask]
            V = V[mask]
            if x.size < 5:
                return [], []
            d = np.diff(V)
            maxs = []
            mins = []
            for i in range(1, len(d)):
                if d[i - 1] > 0 and d[i] <= 0:
                    maxs.append((float(x[i]), float(V[i])))
                if d[i - 1] < 0 and d[i] >= 0:
                    mins.append((float(x[i]), float(V[i])))
            return maxs, mins

        def _to_float_or_none(value: Any) -> Optional[float]:
            return try_parse_user_float(str(value or ""))

        tab = self.tab_reactions
        try:
            tab.recompute_now()
            loads, load_errors, _ = tab._build_loads()
            errors = tab._validate_geometry() + load_errors
            if errors:
                head = "No se puede exportar DOCX: hay datos requeridos incompletos.\n"
                body = "\n".join([f"- {e}" for e in errors[:12]])
                if len(errors) > 12:
                    body += f"\n- ... y {len(errors) - 12} más."
                QMessageBox.warning(self, "Validación de entradas", head + body)
                return

            state = tab.current_plot_state()
            reaction_result = tab._last_result
            if state is None or reaction_result is None:
                raise ValueError("No hay una solución estructural disponible para exportar.")

            self._plot_reactions_tab(tab)
            diag = tab.get_diag()
            xlim = _compute_x_view(state.beam.L_mm, state.point_forces, state.dist_loads, state.moments)
            if diag is None:
                diag = build_V_M(
                    beam_L_mm=state.beam.L_mm,
                    point_forces=state.point_forces,
                    dist_loads=state.dist_loads,
                    moments=state.moments,
                    x_start=xlim[0],
                    x_end=xlim[1],
                )

            maxM, minM = self._find_local_extrema_kgcm(diag, xlim)
            maxV, minV = _extremos_V(diag, xlim)
            all_abs = [(x, abs(v)) for x, v in maxM] + [(x, abs(v)) for x, v in minM]
            if all_abs:
                mmax_x_mm, mmax_kgcm = max(all_abs, key=lambda kv: kv[1])
            else:
                mmax_x_mm, mmax_kgcm = 0.0, 0.0

            unidad_titulo = "Cálculo y verificación"
            default_name = f"Memoria - {unidad_titulo} - {datetime.now().strftime('%Y%m%d')}.docx"
            path, _ = QFileDialog.getSaveFileName(self, "Exportar memoria de cálculo", default_name, "Word (*.docx)")
            if not path:
                return
            if not path.lower().endswith(".docx"):
                path += ".docx"

            tmpdir = tempfile.mkdtemp(prefix="semi_beam_docx_")
            try:
                defl_payload = None
                if tab.deflection_enabled():
                    defl_payload = self._compute_deflection_result(
                        diag=diag,
                        beam_L_mm=state.beam.L_mm,
                        supports=tab.deflection_supports(),
                        params=tab.deflection_params(),
                        section_panel=tab.section_panel,
                    )

                path_fbd = os.path.join(tmpdir, "FBD.jpg")
                self._save_axis_snapshot(self.ax_fbd, path_fbd)
                path_v = os.path.join(tmpdir, "V.jpg")
                self._save_axis_snapshot(self.ax_V, path_v)
                path_m = os.path.join(tmpdir, "M.jpg")
                self._save_axis_snapshot(self.ax_M, path_m)
                path_deflection = ""
                if tab.deflection_enabled() and defl_payload is not None:
                    path_deflection = os.path.join(tmpdir, "Deflection.jpg")
                    self._save_axis_snapshot(self.ax_defl, path_deflection)

                try:
                    tab.section_panel.tbl.clearFocus()
                    tab.section_panel.set_moment_provider(lambda x_mm: float(diag.eval_M(float(x_mm))) / 10.0)
                    tab.section_panel.set_shear_provider(lambda x_mm: float(diag.eval_V(float(x_mm))))
                    if defl_payload is not None:
                        defl_result, i_source = defl_payload
                        tab.section_panel.set_deflection_context(defl_result, i_source=i_source)
                    else:
                        tab.section_panel.set_deflection_context(None)
                    tab.section_panel.clear_results_only()
                    tab.section_panel._recompute_all()
                except Exception:
                    pass
                verification = tab.section_panel.build_verification_export_payload(tmpdir, dpi=300)

                path_sec = os.path.join(tmpdir, "Secciones.jpg")
                try:
                    tab.section_panel.export_table_jpg(path_sec, dpi=MEMORIA_EXPORT_IMAGE_DPI)
                except Exception:
                    path_sec = ""

                sec_data = tab.section_panel.extract_memoria_data()
                sec_rows = list(sec_data.get("rows", []))
                sec_imgs: Dict[str, str] = {}
                for i, sec_name in enumerate(["A-A'", "B-B'", "C-C'", "D-D'", "E-E'"]):
                    pth = os.path.join(tmpdir, f"sec_{i + 1}.jpg")
                    row = sec_rows[i] if i < len(sec_rows) else {}
                    self._save_text_schematic(
                        pth,
                        f"Sección {sec_name}",
                        [
                            f"x = {row.get('x_mm', '-') or '-'} mm",
                            f"h_web = {row.get('h_web_mm', '-') or '-'} mm",
                            f"t_web = {row.get('t_web_in', '-') or '-'} in",
                        ],
                    )
                    sec_imgs[f"sec_{chr(ord('a') + i)}"] = pth

                if tab.mode.currentIndex() == 0:
                    x_a = float(tab.x_a.value())
                    x_b = float(tab.x_b.value())
                    apoyos = [
                        ("RA", f"x={_fmt_plain(x_a, 0)} mm; R={_fmt_plain(float(reaction_result.reacciones.get('R_A', 0.0)), 2)} kg (calculada)"),
                        ("RB", f"x={_fmt_plain(x_b, 0)} mm; R={_fmt_plain(float(reaction_result.reacciones.get('R_B', 0.0)), 2)} kg (calculada)"),
                    ]
                    dist_perno_mm = x_a
                    peso_eje1_kg = float(reaction_result.reacciones.get("R_A", 0.0))
                    peso_eje2_kg = float(reaction_result.reacciones.get("R_B", 0.0))
                    dist_eje1_mm = x_a
                    dist_eje2_mm = x_b
                    x_t_result = x_b
                    x_d_result = None
                    p_stab_long_lines = [
                        f"x_A = {_fmt_plain(x_a, 0)} mm",
                        f"x_B = {_fmt_plain(x_b, 0)} mm",
                        f"ΣFy residual = {_fmt_plain(float(reaction_result.Fy_total_residual), 6)}",
                    ]
                else:
                    x_k = float(tab.x_k.value())
                    x_t = float(tab.x_t.value())
                    x_d = x_t - float(tab.offset.value())
                    apoyos = [
                        ("Rk", f"x={_fmt_plain(x_k, 0)} mm; R={_fmt_plain(float(reaction_result.reacciones.get('R_k', 0.0)), 2)} kg (calculada)"),
                        ("Rd", f"x={_fmt_plain(x_d, 0)} mm; R={_fmt_plain(float(reaction_result.reacciones.get('R_d', 0.0)), 2)} kg (calculada)"),
                        ("Rt", f"x={_fmt_plain(x_t, 0)} mm; R={_fmt_plain(float(reaction_result.reacciones.get('R_t', 0.0)), 2)} kg (calculada)"),
                    ]
                    dist_perno_mm = x_k
                    peso_eje1_kg = float(reaction_result.reacciones.get("R_d", 0.0))
                    peso_eje2_kg = float(reaction_result.reacciones.get("R_t", 0.0))
                    dist_eje1_mm = x_d
                    dist_eje2_mm = x_t
                    x_t_result = x_t
                    x_d_result = x_d
                    p_stab_long_lines = [
                        f"x_k = {_fmt_plain(x_k, 0)} mm",
                        f"x_d = {_fmt_plain(x_d, 0)} mm",
                        f"x_t = {_fmt_plain(x_t, 0)} mm",
                    ]

                p_stab_long = os.path.join(tmpdir, "stab_long.jpg")
                self._save_text_schematic(p_stab_long, "Estabilidad longitudinal (esquema)", p_stab_long_lines)
                p_stab_lat = os.path.join(tmpdir, "stab_lat.jpg")
                self._save_text_schematic(
                    p_stab_lat,
                    "Estabilidad lateral (esquema)",
                    [
                        f"Modo estructural: {tab.mode.currentText()}",
                        f"Cantidad de cargas ingresadas: {len(loads)}",
                        f"ΣM0 residual = {_fmt_plain(float(reaction_result.M0_residual), 6)}",
                    ],
                )

                apoyos = [
                    ("Rp1", f"x={_fmt_plain(case.kingpin.x_mm, 0)} mm; R={_fmt_plain(_point_value(res.solved_point_forces, 'Rp1', case.kingpin.reaction_user), 2)} kg"),
                    ("Rt", f"x={_fmt_plain(res.x_t_mm, 0)} mm; R={_fmt_plain(_point_value(res.solved_point_forces, 'Rt', case.tandem.reaction_user), 2)} kg"),
                ]
                if case.hitch is not None and lift_load is None:
                    apoyos.append(("Rp2", f"x={_fmt_plain(case.hitch.x_mm, 0)} mm; R={_fmt_plain(_point_value(res.solved_point_forces, 'Rp2', case.hitch.reaction_user), 2)} kg"))
                if case.directional is not None and lift_load is None:
                    apoyos.append(("Rd", f"x={_fmt_plain(res.x_d_mm, 0) if res.x_d_mm is not None else '-'} mm; R={_fmt_plain(_point_value(res.solved_point_forces, 'Rd', case.directional.reaction_user), 2)} kg"))

                cargas = []
                for load in loads:
                    if isinstance(load, PointForce):
                        cargas.append((load.label, f"P: x={_fmt_plain(load.x_mm, 0)} mm; P={_fmt_plain(load.value_user, 2)} kg"))
                    elif isinstance(load, DistUniform):
                        cargas.append((load.label, f"q: x0={_fmt_plain(load.x0_mm, 0)} mm; L={_fmt_plain(load.Lq_mm, 0)} mm; q={_fmt_plain(load.q_user, 6)} kg/mm"))
                    elif isinstance(load, PointMoment):
                        cargas.append((load.label, f"M: x={_fmt_plain(load.x_mm, 0)} mm; M={_fmt_plain(load.M_user_kgmm, 2)} kg·mm"))

                resultados = MemoriaResultados(
                    q_user_kgmm=0.0,
                    x_t_mm=float(x_t_result),
                    x_d_mm=float(x_d_result) if x_d_result is not None else None,
                    residual_Fy=float(reaction_result.Fy_total_residual),
                    residual_M0=float(reaction_result.M0_residual),
                    extremos_V=[("MAX", float(x), float(v)) for x, v in maxV] + [("MIN", float(x), float(v)) for x, v in minV],
                    extremos_M=[("MAX", float(x), float(v)) for x, v in maxM] + [("MIN", float(x), float(v)) for x, v in minM],
                )
                if defl_payload is not None:
                    defl_result, i_source = defl_payload
                    resultados = MemoriaResultados(
                        q_user_kgmm=0.0,
                        x_t_mm=float(x_t_result),
                        x_d_mm=float(x_d_result) if x_d_result is not None else None,
                        residual_Fy=float(reaction_result.Fy_total_residual),
                        residual_M0=float(reaction_result.M0_residual),
                        extremos_V=[("MAX", float(x), float(v)) for x, v in maxV] + [("MIN", float(x), float(v)) for x, v in minV],
                        extremos_M=[("MAX", float(x), float(v)) for x, v in maxM] + [("MIN", float(x), float(v)) for x, v in minM],
                        vmin_mm=float(defl_result.vmin_mm),
                        x_vmin_mm=float(defl_result.x_vmin_mm),
                        utilized_mm=float(defl_result.utilized_mm),
                        allowable_mm=float(defl_result.allowable_mm),
                        deflection_ok=bool(defl_result.ok),
                        i_source=str(i_source),
                    )

                caso = MemoriaCaso(
                    unidad=unidad_titulo,
                    L_carrozable_mm=float(state.beam.L_mm),
                    L_viga_total_mm=float(state.beam.L_mm),
                    descripcion_config=tab.mode.currentText(),
                    apoyos=apoyos,
                    cargas=cargas,
                )

                fs_vals: List[float] = []
                flex_rows = []
                for row in sec_rows:
                    fs = _to_float_or_none(row.get("FS"))
                    if fs is not None:
                        fs_vals.append(fs)
                    flex_rows.append(
                        {
                            "sec": row.get("sec", ""),
                            "M_kgcm": _to_float_or_none(row.get("M_kgcm")) or 0.0,
                            "sigma_max": _to_float_or_none(row.get("sigma_max")) or 0.0,
                            "Wreq_cm3": _to_float_or_none(row.get("Wreq_cm3")) or 0.0,
                            "Wcrit_cm3": _to_float_or_none(row.get("Wcrit_cm3")) or 0.0,
                            "FS": _to_float_or_none(row.get("FS")) or 0.0,
                        }
                    )

                sigma_candidates = [
                    _to_float_or_none(sec_data.get("sigma_top_kgcm2")),
                    _to_float_or_none(sec_data.get("sigma_bot_kgcm2")),
                    _to_float_or_none(sec_data.get("sigma_web_kgcm2")),
                ]
                sigma_candidates = [sigma for sigma in sigma_candidates if sigma is not None]
                fy_kgcm2 = min(sigma_candidates) if sigma_candidates else 0.0

                seccion = MemoriaSeccion(
                    materiales=[
                        ("Planchuela sup", f"{sec_data.get('material_top', '')} / σadm={sec_data.get('sigma_top_kgcm2', '-')}"),
                        ("Planchuela inf", f"{sec_data.get('material_bot', '')} / σadm={sec_data.get('sigma_bot_kgcm2', '-')}"),
                        ("Alma", f"{sec_data.get('material_web', '')} / σadm={sec_data.get('sigma_web_kgcm2', '-')}"),
                    ],
                    fs_min=float(sec_data.get("fs_min") or 0.0),
                    n_vigas=int(sec_data.get("n_beams") or 2),
                    parametros=[],
                    tabla=[],
                )

                dlg = MemoriaHeaderDialog(self, defaults=dict(tab.memoria_header or {}))
                if dlg.exec() != QDialog.DialogCode.Accepted:
                    return
                hdr = dlg.values_dict()
                tab.memoria_header = hdr

                header = MemoriaHeader(
                    titulo=f"Memoria de Cálculo — {unidad_titulo}",
                    cliente_proyecto=" - ".join([x for x in [hdr.get("cliente", ""), hdr.get("proyecto", "")] if x]).strip(),
                    autor=hdr.get("autor", ""),
                    fecha=datetime.now(),
                    revision=hdr.get("revision", "A"),
                )

                imgs = {
                    "fbd": path_fbd,
                    "v": path_v,
                    "m": path_m,
                    "deflection": path_deflection,
                    "secciones": path_sec if path_sec else "",
                    **sec_imgs,
                    "stab_long": p_stab_long,
                    "stab_lat": p_stab_lat,
                }
                extras = {
                    "dist_perno_mm": dist_perno_mm,
                    "peso_eje1_kg": peso_eje1_kg,
                    "peso_eje2_kg": peso_eje2_kg,
                    "dist_eje1_mm": dist_eje1_mm,
                    "dist_eje2_mm": dist_eje2_mm,
                    "mmax_kgcm": float(mmax_kgcm),
                    "mmax_x_mm": float(mmax_x_mm),
                    "alas": f"{sec_data.get('material_top', '')} / {sec_data.get('material_bot', '')}",
                    "alma": str(sec_data.get("material_web", "")),
                    "fy_kgcm2": float(fy_kgcm2),
                    "fs_min_real": min(fs_vals) if fs_vals else 0.0,
                    "flex_rows": flex_rows[:5],
                }

                export_memoria_docx(
                    path,
                    header=header,
                    caso=caso,
                    resultados=resultados,
                    seccion=seccion,
                    imagenes=imgs,
                    extras=extras,
                    verification=verification,
                )
                QMessageBox.information(self, "Memoria de cálculo", f"DOCX generado:\n{path}")
            finally:
                try:
                    import shutil
                    shutil.rmtree(tmpdir, ignore_errors=True)
                except Exception:
                    pass
        except Exception as e:
            QMessageBox.critical(self, "Memoria de cálculo", f"Error al generar DOCX: {e}")

    def _export_memoria_docx(self):
        tab = self.active_tab()
        if tab is self.tab_reactions:
            self._export_reactions_memoria_docx()
            return
        try:
            errors = tab._validate_required_inputs()
            if errors:
                head = "No se puede exportar DOCX: hay datos requeridos incompletos.\n"
                body = "\n".join([f"- {e}" for e in errors[:12]])
                if len(errors) > 12:
                    body += f"\n- ... y {len(errors) - 12} más."
                QMessageBox.warning(self, "Validación de entradas", head + body)
                return

            config_txt = tab.cmb_config.currentText()
            Lc = float(_spin_value_or_none(tab.Lc))
            beam_motor, pforces, dloads, moms, _ = tab.parse_inputs()

            kingpin = FixedSupport(
                label="Rp1",
                x_mm=float(_spin_value_or_none(tab.x_front_or_kp)),
                reaction_user=float(_spin_value_or_none(tab.R_front_or_kp)),
            )
            tandem = TandemSupport(label="Rt", reaction_user=float(_spin_value_or_none(tab.Rt)))

            directional = None
            if tab._config_uses_directional():
                directional = DirectionalSupport(
                    label="Rd",
                    reaction_user=float(_spin_value_or_none(tab.Rd)),
                    offset_mm=float(_spin_value_or_none(tab.dir_offset)),
                )

            hitch = None
            if tab.is_bitren:
                hitch = FixedSupport(
                    label="Rp2",
                    x_mm=Lc + float(_spin_value_or_none(tab.x_rp2_rel)),
                    reaction_user=float(_spin_value_or_none(tab.Rp2)),
                )

            case = BeamCase(
                beam=beam_motor,
                point_forces=pforces,
                dist_loads=dloads,
                moments=moms,
                kingpin=kingpin,
                tandem=tandem,
                directional=directional,
                hitch=hitch,
                unknown_uniform=UnknownUniformLoad(label="q", span_start_mm=0.0, span_len_mm=Lc),
            )
            res = solve_equilibrium(case)

            L_viga_total = float(res.x_t_mm) + 2070.0 if tab.is_bitren else float(Lc)
            beam_plot = Beam(L_mm=L_viga_total)
            res, lift_load, reaction_result = self._solve_final_equilibrium_with_axis_lift(
                tab=tab,
                case=case,
                base_result=res,
                beam_L_mm=float(beam_plot.L_mm),
            )
            xlim = _compute_x_view(beam_plot.L_mm, res.solved_point_forces, res.solved_dist_loads, res.solved_moments)
            diag = build_V_M(
                beam_L_mm=beam_plot.L_mm,
                point_forces=res.solved_point_forces,
                dist_loads=res.solved_dist_loads,
                moments=res.solved_moments,
                x_start=xlim[0],
                x_end=xlim[1],
            )
            maxM, minM = self._find_local_extrema_kgcm(diag, xlim)
            all_abs = [(x, abs(v)) for x, v in maxM] + [(x, abs(v)) for x, v in minM]
            if all_abs:
                mmax_x_mm, mmax_kgcm = max(all_abs, key=lambda kv: kv[1])
            else:
                mmax_x_mm, mmax_kgcm = 0.0, 0.0

            default_name = f"Memoria - {tab.title} - {datetime.now().strftime('%Y%m%d')}.docx"
            path, _ = QFileDialog.getSaveFileName(self, "Exportar memoria de cálculo", default_name, "Word (*.docx)")
            if not path:
                return
            if not path.lower().endswith(".docx"):
                path += ".docx"

            tmpdir = tempfile.mkdtemp(prefix="semi_beam_docx_")
            try:
                defl_payload = None
                if tab.deflection_enabled():
                    defl_payload = self._compute_deflection_result(
                        diag=diag,
                        beam_L_mm=beam_plot.L_mm,
                        supports=tab.deflection_supports(),
                        params=tab.deflection_params(),
                        section_panel=tab.section_panel,
                    )
                path_fbd = os.path.join(tmpdir, "FBD.jpg")
                self._save_axis_snapshot(self.ax_fbd, path_fbd)
                path_v = os.path.join(tmpdir, "V.jpg")
                self._save_axis_snapshot(self.ax_V, path_v)
                path_m = os.path.join(tmpdir, "M.jpg")
                self._save_axis_snapshot(self.ax_M, path_m)
                path_deflection = ""
                if tab.deflection_enabled() and defl_payload is not None:
                    path_deflection = os.path.join(tmpdir, "Deflection.jpg")
                    self._save_axis_snapshot(self.ax_defl, path_deflection)

                try:
                    tab.section_panel.tbl.clearFocus()
                    tab.section_panel.set_moment_provider(lambda x_mm: float(diag.eval_M(float(x_mm))) / 10.0)
                    tab.section_panel.set_shear_provider(lambda x_mm: float(diag.eval_V(float(x_mm))))
                    if defl_payload is not None:
                        defl_result, i_source = defl_payload
                        tab.section_panel.set_deflection_context(defl_result, i_source=i_source)
                    else:
                        tab.section_panel.set_deflection_context(None)
                    tab.section_panel.clear_results_only()
                    tab.section_panel._recompute_all()
                except Exception:
                    pass
                verification = tab.section_panel.build_verification_export_payload(tmpdir, dpi=300)

                path_sec = os.path.join(tmpdir, "Secciones.jpg")
                try:
                    tab.section_panel.export_table_jpg(path_sec, dpi=MEMORIA_EXPORT_IMAGE_DPI)
                except Exception:
                    path_sec = ""

                sec_data = tab.section_panel.extract_memoria_data()
                sec_rows = sec_data.get("rows", [])
                sec_imgs: Dict[str, str] = {}
                for i, sec_name in enumerate(["A-A'", "B-B'", "C-C'", "D-D'", "E-E'"]):
                    pth = os.path.join(tmpdir, f"sec_{i+1}.jpg")
                    row = sec_rows[i] if i < len(sec_rows) else {}
                    self._save_text_schematic(
                        pth,
                        f"Sección {sec_name}",
                        [
                            f"x = {row.get('x_mm', '-') or '-'} mm",
                            f"h_web = {row.get('h_web_mm', '-') or '-'} mm",
                            f"t_web = {row.get('t_web_in', '-') or '-'} in",
                        ],
                    )
                    sec_imgs[f"sec_{chr(ord('a') + i)}"] = pth

                p_stab_long = os.path.join(tmpdir, "stab_long.jpg")
                self._save_text_schematic(
                    p_stab_long,
                    "Estabilidad longitudinal (esquema)",
                    [
                        f"x_t = {_fmt_plain(float(res.x_t_mm), 0)} mm",
                        f"x_d = {_fmt_plain(float(res.x_d_mm), 0) if res.x_d_mm is not None else '-'} mm",
                        f"q = {_fmt_plain(float(res.q_user_kg_per_mm), 6)} kg/mm",
                    ],
                )
                p_stab_lat = os.path.join(tmpdir, "stab_lat.jpg")
                self._save_text_schematic(
                    p_stab_lat,
                    "Estabilidad lateral (esquema)",
                    [
                        "Modelo simplificado por simetría",
                        "Sección tipo I verificada por flexión",
                        "FS según tabla de chequeo",
                    ],
                )

                apoyos = [
                    ("Rp1", f"x={_fmt_plain(case.kingpin.x_mm, 0)} mm; R={_fmt_plain(case.kingpin.reaction_user, 2)} kg (usuario)"),
                    ("Rt", f"x∈[{_fmt_plain(case.tandem.x_min_mm, 0)}, {_fmt_plain(case.tandem.x_max_mm, 0)}] mm; R={_fmt_plain(case.tandem.reaction_user, 2)} kg (usuario)"),
                ]
                if case.hitch is not None:
                    apoyos.append(("Rp2", f"x={_fmt_plain(case.hitch.x_mm, 0)} mm; R={_fmt_plain(case.hitch.reaction_user, 2)} kg (usuario)"))
                if case.directional is not None:
                    apoyos.append(("Rd", f"offset={_fmt_plain(case.directional.offset_mm, 0)} mm; R={_fmt_plain(case.directional.reaction_user, 2)} kg (usuario)"))

                cargas = []
                for pf in case.point_forces:
                    cargas.append((pf.label, f"P: x={_fmt_plain(pf.x_mm, 0)} mm; P={_fmt_plain(pf.value_user, 2)} kg"))
                if lift_load is not None:
                    cargas.append((
                        lift_load.label,
                        (
                            f"Levante automatico: x={_fmt_plain(lift_load.x_mm, 0)} mm; "
                            f"P={_fmt_plain(lift_load.value_user, 2)} kg descendente. "
                            "No modifica la posicion calculada del tandem; las reacciones se recalculan "
                            "con apoyos fijos para evaluar esfuerzos."
                        ),
                    ))
                for dl in case.dist_loads:
                    cargas.append((dl.label, f"q: x0={_fmt_plain(dl.x0_mm, 0)} mm; L={_fmt_plain(dl.Lq_mm, 0)} mm; q={_fmt_plain(dl.q_user, 6)} kg/mm"))
                for pm in case.moments:
                    cargas.append((pm.label, f"M: x={_fmt_plain(pm.x_mm, 0)} mm; M={_fmt_plain(pm.M_user_kgmm, 2)} kg·mm"))

                resultados = MemoriaResultados(
                    q_user_kgmm=float(res.q_user_kg_per_mm),
                    x_t_mm=float(res.x_t_mm),
                    x_d_mm=float(res.x_d_mm) if res.x_d_mm is not None else None,
                    residual_Fy=float(res.residual_Fy),
                    residual_M0=float(res.residual_M0),
                    extremos_V=[],
                    extremos_M=[("MAX", float(x), float(v)) for x, v in maxM] + [("MIN", float(x), float(v)) for x, v in minM],
                )
                if defl_payload is not None:
                    defl_result, i_source = defl_payload
                    resultados = MemoriaResultados(
                        q_user_kgmm=float(res.q_user_kg_per_mm),
                        x_t_mm=float(res.x_t_mm),
                        x_d_mm=float(res.x_d_mm) if res.x_d_mm is not None else None,
                        residual_Fy=float(res.residual_Fy),
                        residual_M0=float(res.residual_M0),
                        extremos_V=[],
                        extremos_M=[("MAX", float(x), float(v)) for x, v in maxM] + [("MIN", float(x), float(v)) for x, v in minM],
                        vmin_mm=float(defl_result.vmin_mm),
                        x_vmin_mm=float(defl_result.x_vmin_mm),
                        utilized_mm=float(defl_result.utilized_mm),
                        allowable_mm=float(defl_result.allowable_mm),
                        deflection_ok=bool(defl_result.ok),
                        i_source=str(i_source),
                    )
                caso = MemoriaCaso(
                    unidad=tab.title,
                    L_carrozable_mm=float(Lc),
                    L_viga_total_mm=float(L_viga_total),
                    descripcion_config=config_txt,
                    apoyos=apoyos,
                    cargas=cargas,
                )

                fs_vals: List[float] = []
                flex_rows = []
                for r in sec_rows:
                    def _fv(key: str) -> Optional[float]:
                        return try_parse_user_float(str(r.get(key, "") or ""))
                    fs = _fv("FS")
                    if fs is not None:
                        fs_vals.append(fs)
                    flex_rows.append({
                        "sec": r.get("sec", ""),
                        "M_kgcm": _fv("M_kgcm") or 0.0,
                        "sigma_max": _fv("sigma_max") or 0.0,
                        "Wreq_cm3": _fv("Wreq_cm3") or 0.0,
                        "Wcrit_cm3": _fv("Wcrit_cm3") or 0.0,
                        "FS": _fv("FS") or 0.0,
                    })

                def _f_or_none(v: Any) -> Optional[float]:
                    return try_parse_user_float(str(v))

                sigma_candidates = [
                    _f_or_none(sec_data.get("sigma_top_kgcm2")),
                    _f_or_none(sec_data.get("sigma_bot_kgcm2")),
                    _f_or_none(sec_data.get("sigma_web_kgcm2")),
                ]
                sigma_candidates = [s for s in sigma_candidates if s is not None]
                fy_kgcm2 = min(sigma_candidates) if sigma_candidates else 0.0

                seccion = MemoriaSeccion(
                    materiales=[
                        ("Planchuela sup", f"{sec_data.get('material_top', '')} / σadm={sec_data.get('sigma_top_kgcm2', '-')}"),
                        ("Planchuela inf", f"{sec_data.get('material_bot', '')} / σadm={sec_data.get('sigma_bot_kgcm2', '-')}"),
                        ("Alma", f"{sec_data.get('material_web', '')} / σadm={sec_data.get('sigma_web_kgcm2', '-')}"),
                    ],
                    fs_min=float(sec_data.get("fs_min") or 0.0),
                    n_vigas=int(sec_data.get("n_beams") or 2),
                    parametros=[],
                    tabla=[],
                )

                cache = tab.get_cache()
                dlg = MemoriaHeaderDialog(self, defaults=cache.memoria_header if cache else {})
                if dlg.exec() != QDialog.DialogCode.Accepted:
                    return
                hdr = dlg.values_dict()
                if cache is not None:
                    cache.memoria_header = hdr

                header = MemoriaHeader(
                    titulo=f"Memoria de Cálculo — {tab.title}",
                    cliente_proyecto=" - ".join([x for x in [hdr.get("cliente", ""), hdr.get("proyecto", "")] if x]).strip(),
                    autor=hdr.get("autor", ""),
                    fecha=datetime.now(),
                    revision=hdr.get("revision", "A"),
                )

                imgs = {
                    "fbd": path_fbd,
                    "v": path_v,
                    "m": path_m,
                    "deflection": path_deflection,
                    "secciones": path_sec if path_sec else "",
                    **sec_imgs,
                    "stab_long": p_stab_long,
                    "stab_lat": p_stab_lat,
                }

                peso_eje1 = float(_spin_value_or_none(tab.Rd)) if tab._config_uses_directional() else float(_spin_value_or_none(tab.Rt))
                peso_eje2 = float(_spin_value_or_none(tab.Rt)) if tab._config_uses_directional() else 0.0
                extras = {
                    "dist_perno_mm": float(_spin_value_or_none(tab.x_front_or_kp)),
                    "peso_eje1_kg": peso_eje1,
                    "peso_eje2_kg": peso_eje2,
                    "dist_eje1_mm": float(res.x_d_mm) if res.x_d_mm is not None else float(res.x_t_mm),
                    "dist_eje2_mm": float(res.x_t_mm),
                    "mmax_kgcm": float(mmax_kgcm),
                    "mmax_x_mm": float(mmax_x_mm),
                    "alas": f"{sec_data.get('material_top','')} / {sec_data.get('material_bot','')}",
                    "alma": str(sec_data.get("material_web", "")),
                    "fy_kgcm2": float(fy_kgcm2),
                    "fs_min_real": min(fs_vals) if fs_vals else 0.0,
                    "flex_rows": flex_rows[:5],
                }

                export_memoria_docx(
                    path,
                    header=header,
                    caso=caso,
                    resultados=resultados,
                    seccion=seccion,
                    imagenes=imgs,
                    extras=extras,
                    verification=verification,
                )
                QMessageBox.information(self, "Memoria de cálculo", f"DOCX generado:\n{path}")
            finally:
                try:
                    import shutil
                    shutil.rmtree(tmpdir, ignore_errors=True)
                except Exception:
                    pass
        except Exception as e:
            QMessageBox.critical(self, "Memoria de cálculo", f"Error al generar DOCX: {e}")

def main():
    app = QApplication(sys.argv)
    app.setApplicationName("Calculeitor")
    app.setApplicationVersion(APP_VERSION)
    try:
        app.setWindowIcon(QIcon(ensure_calculeitor_icon()))
    except Exception:
        pass
    w = FBDApp()
    w.show()
    sys.exit(app.exec())
