# path: src/semi_beam/ui/main_window.py
from __future__ import annotations

import sys
import os
import tempfile
from datetime import datetime
from dataclasses import dataclass, field, fields as dc_fields, is_dataclass
from typing import List, Optional, Tuple, Dict, Any

import matplotlib
matplotlib.use("QtAgg")
import matplotlib.pyplot as plt
import numpy as np

from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QWheelEvent, QIcon, QColor, QBrush
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QFormLayout,
    QLabel, QDoubleSpinBox, QPushButton, QTableWidget, QTableWidgetItem,
    QSizePolicy, QSplitter, QScrollArea, QTabWidget, QMessageBox, QFileDialog,
    QToolButton, QFrame, QComboBox, QDialog, QCheckBox
)
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas

# Delegados numéricos (solo números / admite vacío)
from semi_beam.ui.numeric_delegate import (
    NullableFloatDelegate,
    FlexibleDoubleSpinBox,
    apply_table_readability_style,
    TABLE_INPUT_BG,
    TABLE_READONLY_BG,
    TABLE_TEXT_COLOR,
)

# ---- Dominio / motor / view ----
from semi_beam.domain.beam import Beam
from semi_beam.domain.loads import PointForce, DistUniform, PointMoment
from semi_beam.domain.labels import p_index, next_free_p_index, to_internal_Fy, to_internal_w_up
from semi_beam.engine.normalize import normalize_inputs
from semi_beam.view.style import RenderStyle
from semi_beam.view.renderer_fbd import render_fbd

from semi_beam.domain.supports import FixedSupport, TandemSupport, DirectionalSupport
from semi_beam.domain.unknowns import UnknownUniformLoad
from semi_beam.domain.cases import BeamCase
from semi_beam.engine.equilibrium import solve_equilibrium
from semi_beam.engine.deflection import compute_total_deflection
from semi_beam.engine.diagrams import build_V_M
from semi_beam.view.renderer_vm import render_shear, render_moment, render_deflection
from semi_beam.services.memoria_calculo_pdf import (
    export_memoria_pdf, MemoriaHeader, MemoriaCaso, MemoriaResultados, MemoriaSeccion
)
from semi_beam.services.memoria_calculo_docx import export_memoria_docx, ensure_memoria_template, default_template_path
from semi_beam.services.branding import ensure_calculeitor_icon

# ---- Verificador (TU UI anterior) ----
from semi_beam.ui.section_check_panel import SectionCheckPanel
from semi_beam.ui.memoria_header_dialog import MemoriaHeaderDialog
from semi_beam.ui.reactions_tab import SemiTrailerReactionsTab


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
    t = (s or "").strip().replace(",", ".")
    if t == "":
        return None
    try:
        return float(t)
    except Exception:
        return None


def _spin_text(sp: QDoubleSpinBox) -> str:
    try:
        le = sp.lineEdit()
        if le is not None:
            return (le.text() or "").strip()
    except Exception:
        pass
    return (sp.text() or "").strip()


def _spin_value_or_none(sp: QDoubleSpinBox) -> Optional[float]:
    t = _spin_text(sp).replace(",", ".")
    if t == "":
        return None
    try:
        return float(t)
    except Exception:
        return None


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


def _dc_field_names(cls) -> set[str]:
    try:
        return {f.name for f in dc_fields(cls)}
    except Exception:
        return set()


def _dc_make(cls, data: Dict[str, Any]):
    """
    Construye dataclass tolerante a cambios:
    - filtra keys inexistentes
    - si el cls no es dataclass, intenta llamar igual con kwargs filtrados por signature (best effort)
    """
    fset = _dc_field_names(cls)
    if fset:
        kwargs = {k: v for k, v in data.items() if k in fset}
        return cls(**kwargs)
    # fallback no-dataclass (o dataclass sin introspección)
    try:
        return cls(**data)
    except Exception:
        # último recurso: intenta con keys comunes
        common = {}
        for k in ("titulo", "cliente", "proyecto", "cliente_proyecto", "autor", "revision", "fecha"):
            if k in data:
                common[k] = data[k]
        return cls(**common)


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
    def __init__(self, title: str, *, is_bitren: bool = False, is_acoplado: bool = False):
        super().__init__()
        self.title = title
        self.is_bitren = is_bitren
        self.is_acoplado = is_acoplado

        self._cached: Optional[SessionCache] = None
        self._last_diag = None

        self._all_boxes: List[CollapsibleBox] = []

        root = QVBoxLayout(self)
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

        # --- Entradas motor comunes ---
        self.Lc = FlexibleDoubleSpinBox()
        self._setup_motor_spin(self.Lc, minv=1.0, maxv=1e12, decimals=2, step=50.0)

        self.x_front_or_kp = FlexibleDoubleSpinBox()
        self._setup_motor_spin(self.x_front_or_kp, minv=-1e12, maxv=1e12, decimals=2, step=50.0)

        self.R_front_or_kp = FlexibleDoubleSpinBox()
        self._setup_motor_spin(self.R_front_or_kp, minv=-1e12, maxv=1e12, decimals=2, step=50.0)

        self.Rt = FlexibleDoubleSpinBox()
        self._setup_motor_spin(self.Rt, minv=-1e12, maxv=1e12, decimals=2, step=50.0)

        # Direccional (semi/bitren)
        self.Rd = FlexibleDoubleSpinBox()
        self._setup_motor_spin(self.Rd, minv=-1e12, maxv=1e12, decimals=2, step=50.0)

        self.dir_offset = FlexibleDoubleSpinBox()
        self._setup_motor_spin(self.dir_offset, minv=0.0, maxv=20000.0, decimals=1, step=25.0)

        # Bitren Rp2
        self.x_rp2_rel = FlexibleDoubleSpinBox()
        self._setup_motor_spin(self.x_rp2_rel, minv=-1e12, maxv=1e12, decimals=2, step=50.0)

        self.Rp2 = FlexibleDoubleSpinBox()
        self._setup_motor_spin(self.Rp2, minv=-1e12, maxv=1e12, decimals=2, step=50.0)

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

        # ==========================
        # Collapsible: Puntuales
        # ==========================
        p_box = CollapsibleBox("Fuerzas puntuales conocidas (P1, P2, ...)")
        self._all_boxes.append(p_box)
        root.addWidget(p_box)
        p_v = p_box.content_layout()

        self.tbl_points = QTableWidget(0, 3)
        self.tbl_points.setHorizontalHeaderLabels(["label", "x_mm", "valor_kg"])
        self.tbl_points.horizontalHeader().setStretchLastSection(True)
        apply_table_readability_style(self.tbl_points)
        p_v.addWidget(self.tbl_points)

        p_btns = QHBoxLayout()
        self.btn_add_p = QPushButton("Agregar fuerza")
        self.btn_del_p = QPushButton("Eliminar seleccionadas")
        p_btns.addWidget(self.btn_add_p)
        p_btns.addWidget(self.btn_del_p)
        p_btns.addStretch(1)
        p_v.addLayout(p_btns)

        self.tbl_points.setItemDelegateForColumn(1, NullableFloatDelegate(self, decimals=2, minv=-1e12, maxv=1e12))
        self.tbl_points.setItemDelegateForColumn(2, NullableFloatDelegate(self, decimals=2, minv=-1e12, maxv=1e12))

        # ==========================
        # Collapsible: Distribuidas
        # ==========================
        q_box = CollapsibleBox("Cargas distribuidas conocidas (kg/mm)")
        self._all_boxes.append(q_box)
        root.addWidget(q_box)
        q_v = q_box.content_layout()

        self.tbl_dists = QTableWidget(0, 4)
        self.tbl_dists.setHorizontalHeaderLabels(["label", "x0_mm", "Lq_mm", "q_kg_per_mm"])
        self.tbl_dists.horizontalHeader().setStretchLastSection(True)
        apply_table_readability_style(self.tbl_dists)
        q_v.addWidget(self.tbl_dists)

        q_btns = QHBoxLayout()
        self.btn_add_q = QPushButton("Agregar distribuida")
        self.btn_del_q = QPushButton("Eliminar seleccionadas")
        q_btns.addWidget(self.btn_add_q)
        q_btns.addWidget(self.btn_del_q)
        q_btns.addStretch(1)
        q_v.addLayout(q_btns)

        self.tbl_dists.setItemDelegateForColumn(1, NullableFloatDelegate(self, decimals=2, minv=-1e12, maxv=1e12))
        self.tbl_dists.setItemDelegateForColumn(2, NullableFloatDelegate(self, decimals=2, minv=-1e12, maxv=1e12))
        self.tbl_dists.setItemDelegateForColumn(3, NullableFloatDelegate(self, decimals=6, minv=-1e12, maxv=1e12))

        # ==========================
        # Collapsible: Momentos
        # ==========================
        m_box = CollapsibleBox("Momentos puntuales (kg·mm, CCW+)")
        self._all_boxes.append(m_box)
        root.addWidget(m_box)
        m_v = m_box.content_layout()

        self.tbl_moms = QTableWidget(0, 3)
        self.tbl_moms.setHorizontalHeaderLabels(["label", "x_mm", "M_kgmm"])
        self.tbl_moms.horizontalHeader().setStretchLastSection(True)
        apply_table_readability_style(self.tbl_moms)
        m_v.addWidget(self.tbl_moms)

        m_btns = QHBoxLayout()
        self.btn_add_m = QPushButton("Agregar momento")
        self.btn_del_m = QPushButton("Eliminar seleccionadas")
        m_btns.addWidget(self.btn_add_m)
        m_btns.addWidget(self.btn_del_m)
        m_btns.addStretch(1)
        m_v.addLayout(m_btns)

        self.tbl_moms.setItemDelegateForColumn(1, NullableFloatDelegate(self, decimals=2, minv=-1e12, maxv=1e12))
        self.tbl_moms.setItemDelegateForColumn(2, NullableFloatDelegate(self, decimals=2, minv=-1e18, maxv=1e18))

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

        # Señales tablas
        self.btn_add_p.clicked.connect(self._add_point_row)
        self.btn_del_p.clicked.connect(lambda: self._remove_selected_rows(self.tbl_points))
        self.btn_add_q.clicked.connect(self._add_dist_row)
        self.btn_del_q.clicked.connect(lambda: self._remove_selected_rows(self.tbl_dists))
        self.btn_add_m.clicked.connect(self._add_mom_row)
        self.btn_del_m.clicked.connect(lambda: self._remove_selected_rows(self.tbl_moms))
        self.tbl_points.itemChanged.connect(lambda *_: self._refresh_table_edit_locks(self.tbl_points))
        self.tbl_dists.itemChanged.connect(lambda *_: self._refresh_table_edit_locks(self.tbl_dists))
        self.tbl_moms.itemChanged.connect(lambda *_: self._refresh_table_edit_locks(self.tbl_moms))

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
        t = _spin_text(sp).replace(",", ".")
        if t == "":
            self._set_spin_blank(sp)
            return
        try:
            v = float(t)
        except Exception:
            self._set_spin_blank(sp)
            return
        sp.setValue(v)

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
        return (not self.is_acoplado) and (("1 + 2 ejes" in cfg) or ("1 + 3 ejes" in cfg))

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

        for r in range(self.tbl_points.rowCount()):
            x = _try_float(_get_text(self.tbl_points, r, 1))
            p = _try_float(_get_text(self.tbl_points, r, 2))
            if x is None or p is None:
                errors.append(f"Puntuales fila {r+1}: complete x_mm y valor_kg.")

        for r in range(self.tbl_dists.rowCount()):
            x0 = _try_float(_get_text(self.tbl_dists, r, 1))
            lq = _try_float(_get_text(self.tbl_dists, r, 2))
            q = _try_float(_get_text(self.tbl_dists, r, 3))
            if x0 is None or lq is None or q is None:
                errors.append(f"Distribuidas fila {r+1}: complete x0_mm, Lq_mm y q_kg_per_mm.")

        for r in range(self.tbl_moms.rowCount()):
            x = _try_float(_get_text(self.tbl_moms, r, 1))
            m = _try_float(_get_text(self.tbl_moms, r, 2))
            if x is None or m is None:
                errors.append(f"Momentos fila {r+1}: complete x_mm y M_kgmm.")

        return errors

    def _set_item_editable(self, tbl: QTableWidget, r: int, c: int, editable: bool):
        it = tbl.item(r, c)
        if it is None:
            was_blocked = tbl.blockSignals(True)
            try:
                _set_item(tbl, r, c, "")
            finally:
                tbl.blockSignals(was_blocked)
            it = tbl.item(r, c)
        if it is None:
            return
        flags = it.flags()
        if editable:
            flags |= Qt.ItemIsEditable
        else:
            flags &= ~Qt.ItemIsEditable
        it.setFlags(flags)
        self._style_table_item(tbl, r, c, editable=editable)

    def _style_table_item(self, tbl: QTableWidget, r: int, c: int, *, editable: bool):
        it = tbl.item(r, c)
        if it is None:
            return
        bg = TABLE_INPUT_BG if editable else TABLE_READONLY_BG
        it.setBackground(QBrush(QColor(bg)))
        it.setForeground(QBrush(QColor(TABLE_TEXT_COLOR)))

    def _refresh_table_edit_locks(self, tbl: QTableWidget):
        was_blocked = tbl.blockSignals(True)
        try:
            for r in range(tbl.rowCount()):
                self._set_item_editable(tbl, r, 0, False)
                if tbl is self.tbl_points:
                    has_x = _try_float(_get_text(tbl, r, 1)) is not None
                    self._set_item_editable(tbl, r, 1, True)
                    self._set_item_editable(tbl, r, 2, has_x)
                elif tbl is self.tbl_dists:
                    has_x0 = _try_float(_get_text(tbl, r, 1)) is not None
                    has_lq = _try_float(_get_text(tbl, r, 2)) is not None
                    self._set_item_editable(tbl, r, 1, True)
                    self._set_item_editable(tbl, r, 2, has_x0)
                    self._set_item_editable(tbl, r, 3, has_x0 and has_lq)
                elif tbl is self.tbl_moms:
                    has_x = _try_float(_get_text(tbl, r, 1)) is not None
                    self._set_item_editable(tbl, r, 1, True)
                    self._set_item_editable(tbl, r, 2, has_x)
        finally:
            tbl.blockSignals(was_blocked)

    def reset_tab_inputs(self):
        self.tbl_points.setRowCount(0)
        self.tbl_dists.setRowCount(0)
        self.tbl_moms.setRowCount(0)
        if (not self.is_acoplado) and (not self.is_bitren):
            self.cmb_semi_tipo.setCurrentIndex(0)
        self._populate_configs()
        self._clear_motor_inputs()
        self.set_note("(sin notas)")
        self.clear_deflection_summary()
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
            self.R_front_or_kp.setValue(15000.0 if tipo == "Escalado" else 9000.0)

            if "1 + 2 ejes" in cfg:
                self.Rd.setValue(9200.0)
                self.dir_offset.setValue(3075.0)
                self.Rt.setValue(15800.0)
            elif "1 + 3 ejes" in cfg:
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

    # ---- helpers UI tablas
    def _add_row(self, tbl: QTableWidget, values: List[str]):
        r = tbl.rowCount()
        tbl.insertRow(r)
        for c, v in enumerate(values):
            _set_item(tbl, r, c, v)
        self._refresh_table_edit_locks(tbl)

    def _remove_selected_rows(self, tbl: QTableWidget):
        rows = sorted(set([i.row() for i in tbl.selectedItems()]), reverse=True)
        for rr in rows:
            tbl.removeRow(rr)
        if tbl is self.tbl_dists:
            for r in range(tbl.rowCount()):
                _set_item(tbl, r, 0, f"q{r + 1}")
        elif tbl is self.tbl_moms:
            for r in range(tbl.rowCount()):
                _set_item(tbl, r, 0, f"M{r + 1}")
        self._refresh_table_edit_locks(tbl)

    def _next_p_label(self) -> str:
        used: set[int] = set()
        for r in range(self.tbl_points.rowCount()):
            idx = p_index(_get_text(self.tbl_points, r, 0).strip().upper())
            if idx is not None:
                used.add(idx)
        k = next_free_p_index(used)
        return f"P{k}"

    def _add_point_row(self):
        r = self.tbl_points.rowCount()
        self.tbl_points.insertRow(r)
        _set_item(self.tbl_points, r, 0, self._next_p_label())
        _set_item(self.tbl_points, r, 1, "")
        _set_item(self.tbl_points, r, 2, "")
        self._refresh_table_edit_locks(self.tbl_points)

    def _add_dist_row(self):
        r = self.tbl_dists.rowCount()
        self.tbl_dists.insertRow(r)
        _set_item(self.tbl_dists, r, 0, f"q{r + 1}")
        _set_item(self.tbl_dists, r, 1, "")
        _set_item(self.tbl_dists, r, 2, "")
        _set_item(self.tbl_dists, r, 3, "")
        self._refresh_table_edit_locks(self.tbl_dists)

    def _add_mom_row(self):
        r = self.tbl_moms.rowCount()
        self.tbl_moms.insertRow(r)
        _set_item(self.tbl_moms, r, 0, f"M{r + 1}")
        _set_item(self.tbl_moms, r, 1, "")
        _set_item(self.tbl_moms, r, 2, "")
        self._refresh_table_edit_locks(self.tbl_moms)

    # ---- parsing entradas
    def parse_inputs(self) -> Tuple[Beam, List[PointForce], List[DistUniform], List[PointMoment], List[str]]:
        notes: List[str] = []
        Lc = _spin_value_or_none(self.Lc)
        if Lc is None:
            raise ValueError("VALIDACION: complete 'Largo carrozable [mm]'.")
        beam = Beam(L_mm=Lc)

        # Puntuales
        raw: List[Tuple[str, float, float]] = []
        used_idx: set[int] = set()
        for r in range(self.tbl_points.rowCount()):
            label = _get_text(self.tbl_points, r, 0).strip() or "P"
            x = _try_float(_get_text(self.tbl_points, r, 1))
            v = _try_float(_get_text(self.tbl_points, r, 2))
            if x is None or v is None:
                continue
            if _is_reaction_label(label):
                notes.append(f'Se ignoró "{label}" en puntuales (reacciones van en motor).')
                continue
            idx = p_index(label.upper())
            if idx is not None:
                used_idx.add(idx)
            raw.append((label, x, v))

        points: List[PointForce] = []
        for label, x, v in raw:
            if label.strip().upper() == "P":
                k = next_free_p_index(used_idx)
                used_idx.add(k)
                label = f"P{k}"
            points.append(PointForce(label=label, x_mm=x, value_user=v))

        # Distribuidas
        dists: List[DistUniform] = []
        for r in range(self.tbl_dists.rowCount()):
            label = (_get_text(self.tbl_dists, r, 0) or "q").strip()
            x0 = _try_float(_get_text(self.tbl_dists, r, 1))
            Lq = _try_float(_get_text(self.tbl_dists, r, 2))
            q = _try_float(_get_text(self.tbl_dists, r, 3))
            if x0 is None or Lq is None or q is None:
                continue
            dists.append(DistUniform(label=label, x0_mm=x0, Lq_mm=Lq, q_user=q))

        # Momentos
        moms: List[PointMoment] = []
        for r in range(self.tbl_moms.rowCount()):
            label = (_get_text(self.tbl_moms, r, 0) or "M").strip()
            x = _try_float(_get_text(self.tbl_moms, r, 1))
            m = _try_float(_get_text(self.tbl_moms, r, 2))
            if x is None or m is None:
                continue
            moms.append(PointMoment(label=label, x_mm=x, M_user_kgmm=m))

        return beam, points, dists, moms, notes

    def set_cache(self, cache: Optional[SessionCache]):
        self._cached = cache

    def get_cache(self) -> Optional[SessionCache]:
        return self._cached

    def set_note(self, text: str):
        self.note_label.setText(text)

    def set_diag(self, diag):
        self._last_diag = diag
        if diag is None:
            self.section_panel.set_moment_provider(None)
            self.section_panel.clear_results_only(clear_moments_if_no_provider=True)
            self.clear_deflection_summary()
        else:
            self.section_panel.set_moment_provider(lambda x_mm: float(diag.eval_M(float(x_mm))) / 10.0)
            self.section_panel.clear_results_only()

    def get_diag(self):
        return self._last_diag

    def deflection_enabled(self) -> bool:
        return bool(self.chk_show_deflection.isChecked())

    def deflection_params(self) -> Optional[float]:
        return 2.1e4

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


# ============================================================
# MAIN WINDOW
# ============================================================
class FBDApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("calculeitor — Acoplado / Semirremolque / Bitren")
        self.resize(1500, 850)
        try:
            self.setWindowIcon(QIcon(ensure_calculeitor_icon()))
        except Exception:
            pass

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
        self.tabs.addTab(self.tab_reactions, "Calculo y verificación")

        left_lay.addWidget(self.tabs)

        left_scroll = QScrollArea()
        left_scroll.setWidgetResizable(True)
        left_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        left_scroll.setWidget(left_host)
        left_scroll.setMinimumWidth(380)
        splitter.addWidget(left_scroll)

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
        self.btn_export_plots = QPushButton("Exportar gráficos (FBD, V(x), M(x), deformada)")
        self.btn_export_memoria = QPushButton("Exportar memoria de cálculo (PDF)")
        self.btn_export_memoria_docx = QPushButton("Exportar Memoria (DOCX)")
        btn_row.addWidget(self.btn_export_plots)
        btn_row.addWidget(self.btn_export_memoria)
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

            tab.tbl_points.cellChanged.connect(lambda *_, t=tab: self._schedule_replot_tab(t, reset_solution=True))
            tab.tbl_dists.cellChanged.connect(lambda *_, t=tab: self._schedule_replot_tab(t, reset_solution=True))
            tab.tbl_moms.cellChanged.connect(lambda *_, t=tab: self._schedule_replot_tab(t, reset_solution=True))
            tab.chk_show_deflection.toggled.connect(lambda *_, t=tab: self._schedule_replot_tab(t, reset_solution=False))
            tab.section_panel.inertia_inputs_changed.connect(lambda t=tab: self._schedule_replot_tab(t, reset_solution=False))

        self.tab_reactions.plot_data_changed.connect(self._schedule_active_replot)
        self.tab_reactions.section_panel.inertia_inputs_changed.connect(self._schedule_active_replot)

        self.tabs.currentChanged.connect(lambda _i: self._on_tab_changed())
        self.btn_export_plots.clicked.connect(self._export_plots_jpg_1200)
        self.btn_export_memoria.clicked.connect(self._export_memoria_pdf)
        self.btn_export_memoria_docx.clicked.connect(self._export_memoria_docx)

        self.active_tab().set_note("Complete las entradas para visualizar y resolver.")
        self._clear_plot_canvas("Sin resultados. Complete datos y presione Resolver equilibrio.")
        self._update_export_buttons()

        self._resize_timer = QTimer(self)
        self._resize_timer.setSingleShot(True)
        self._resize_timer.timeout.connect(self._replot_active_tab)
        self.canvas.mpl_connect("resize_event", lambda evt: self._resize_timer.start(80))

    def active_tab(self):
        return self.tabs.currentWidget()

    def _on_tab_changed(self):
        self._update_export_buttons()
        self._replot_active_tab()

    def _update_export_buttons(self):
        is_reactions = self.active_tab() is self.tab_reactions
        self.btn_export_memoria.setEnabled(not is_reactions)
        self.btn_export_memoria_docx.setEnabled(not is_reactions)

    def _schedule_active_replot(self):
        self._redraw_timer.start(60)

    def eventFilter(self, obj, event):
        if obj is self.canvas and isinstance(event, QWheelEvent):
            bar = self.plot_scroll.verticalScrollBar()
            bar.setValue(bar.value() - event.angleDelta().y())
            return True
        return super().eventFilter(obj, event)

    def _clear_plot_canvas(self, subtitle: str = ""):
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
            tab.set_cache(None)
            tab.set_diag(None)
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
            return

        if payload is None:
            self.ax_defl.clear()
            self.ax_defl.set_axis_off()
            self.ax_defl.text(0.5, 0.5, "Deformada no disponible. Complete la tabla de secciones.", ha="center", va="center", transform=self.ax_defl.transAxes)
            if summary_target is not None:
                summary_target.clear_deflection_summary("Convexidad L/2: +30 mm\nvmin total: -\nUtilizado: - / 60 mm\nEstado: complete la tabla de secciones")
            return

        result, i_source = payload
        render_deflection(self.ax_defl, result, y_zoom=1.0, xlim=xlim)
        if summary_target is not None:
            summary_target.set_deflection_summary(self._deflection_summary_text(result, i_source), ok=bool(result.ok))

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
        render_shear(self.ax_V, diag, y_zoom=1.0, xlim=xlim)
        self.ax_V.set_xlabel("x [mm]")
        self.ax_V.tick_params(labelbottom=True)

        render_moment(self.ax_M, diag, y_zoom=1.0, xlim=xlim)
        self.ax_M.set_xlabel("x [mm]")
        self.ax_M.tick_params(labelbottom=True)

        defl_payload = None
        if set_diag_on_tab is not None:
            defl_payload = self._compute_deflection_result(
                diag=diag,
                beam_L_mm=beam_plot.L_mm,
                supports=cache.deflection_supports,
                params=set_diag_on_tab.deflection_params(),
                section_panel=set_diag_on_tab.section_panel,
            )
            self._render_deflection_axis(
                payload=defl_payload,
                xlim=xlim,
                enabled=set_diag_on_tab.deflection_enabled(),
                summary_target=set_diag_on_tab,
                unavailable_text="Deformada desactivada.",
            )
        else:
            self._render_deflection_axis(
                payload=None,
                xlim=xlim,
                enabled=False,
                summary_target=None,
            )

        self.fig.subplots_adjust(left=0.07, right=0.985, top=0.96, bottom=0.06, hspace=0.65)
        self.canvas.draw_idle()

        if set_diag_on_tab is not None:
            set_diag_on_tab.set_diag(diag)

    def _plot_reactions_tab(self, tab: SemiTrailerReactionsTab):
        state = tab.current_plot_state()
        if state is None:
            tab.set_diag(None)
            self._clear_plot_canvas("Calculo y verificación: complete datos válidos para calcular.")
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

        if state.show_vm:
            render_shear(self.ax_V, diag, y_zoom=1.0, xlim=xlim)
            self.ax_V.set_xlabel("x [mm]")
            self.ax_V.tick_params(labelbottom=True)
            render_moment(self.ax_M, diag, y_zoom=1.0, xlim=xlim)
            self.ax_M.set_xlabel("x [mm]")
            self.ax_M.tick_params(labelbottom=True)
        else:
            for ax, title in ((self.ax_V, "V(x) desactivado"), (self.ax_M, "M(x) desactivado")):
                ax.clear()
                ax.set_axis_off()
                ax.text(0.5, 0.5, title, ha="center", va="center", transform=ax.transAxes)

        defl_payload = self._compute_deflection_result(
            diag=diag,
            beam_L_mm=state.beam.L_mm,
            supports=tab.deflection_supports(),
            params=tab.deflection_params(),
            section_panel=tab.section_panel,
        )
        self._render_deflection_axis(
            payload=defl_payload,
            xlim=xlim,
            enabled=tab.deflection_enabled(),
            summary_target=tab,
            unavailable_text="Deformada desactivada.",
        )

        self.fig.subplots_adjust(left=0.07, right=0.985, top=0.96, bottom=0.06, hspace=0.65)
        self.canvas.draw_idle()

    def _replot_active_tab(self):
        tab = self.active_tab()
        if tab is self.tab_reactions:
            self._plot_reactions_tab(self.tab_reactions)
            return
        cache = tab.get_cache()
        if cache is None:
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
            tab.set_cache(cache)
            tab.set_note(note)
            self._plot_triplet(cache, set_diag_on_tab=tab)

        except Exception as e:
            msg = str(e)
            if msg.startswith("VALIDACION:"):
                tab.set_cache(None)
                tab.set_note(msg.replace("VALIDACION:", "").strip())
                tab.set_diag(None)
                if tab is self.active_tab():
                    self._clear_plot_canvas(msg.replace("VALIDACION:", "").strip())
                return
            QMessageBox.critical(self, "Error", f"Error al graficar entradas: {e}")
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

            tab.set_cache(cache)
            tab.set_note(note)

            self._plot_triplet(cache, set_diag_on_tab=tab)
            tab.section_panel.clear_results_only()

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Error al resolver equilibrio: {e}")
            tab.set_cache(None)
            tab.set_note(f"Error: {e}")
            tab.set_diag(None)

    # Export plots
    def _export_plots_jpg_1200(self):
        folder = QFileDialog.getExistingDirectory(self, "Seleccionar carpeta destino")
        if not folder:
            return
        try:
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

    def _export_memoria_docx(self):
        tab = self.active_tab()
        if tab is self.tab_reactions:
            QMessageBox.information(self, "Memoria DOCX", "La pestaña Semirremolque - Reacciones no exporta memoria DOCX.")
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
            path, _ = QFileDialog.getSaveFileName(self, "Exportar Memoria (DOCX)", default_name, "Word (*.docx)")
            if not path:
                return
            if not path.lower().endswith(".docx"):
                path += ".docx"

            tmpdir = tempfile.mkdtemp(prefix="semi_beam_docx_")
            try:
                self.canvas.draw()
                fig = self.fig
                renderer = fig.canvas.get_renderer()

                path_fbd = os.path.join(tmpdir, "FBD.jpg")
                fig.savefig(path_fbd, dpi=300, bbox_inches=self.ax_fbd.get_tightbbox(renderer).transformed(fig.dpi_scale_trans.inverted()))
                path_v = os.path.join(tmpdir, "V.jpg")
                fig.savefig(path_v, dpi=300, bbox_inches=self.ax_V.get_tightbbox(renderer).transformed(fig.dpi_scale_trans.inverted()))
                path_m = os.path.join(tmpdir, "M.jpg")
                fig.savefig(path_m, dpi=300, bbox_inches=self.ax_M.get_tightbbox(renderer).transformed(fig.dpi_scale_trans.inverted()))

                path_sec = os.path.join(tmpdir, "Secciones.jpg")
                try:
                    tab.section_panel.export_table_jpg(path_sec, dpi=300)
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
                        t = str(r.get(key, "") or "").strip().replace(",", ".")
                        if t == "":
                            return None
                        try:
                            return float(t)
                        except Exception:
                            return None
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
                    try:
                        t = str(v).strip().replace(",", ".")
                        return None if t == "" else float(t)
                    except Exception:
                        return None

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

                template_path = ensure_memoria_template(default_template_path())
                export_memoria_docx(
                    path,
                    template_path=template_path,
                    header=header,
                    caso=caso,
                    resultados=resultados,
                    seccion=seccion,
                    imagenes=imgs,
                    extras=extras,
                )
                QMessageBox.information(self, "Memoria DOCX", f"DOCX generado:\n{path}")
            finally:
                try:
                    import shutil
                    shutil.rmtree(tmpdir, ignore_errors=True)
                except Exception:
                    pass
        except Exception as e:
            QMessageBox.critical(self, "Memoria DOCX", f"Error al generar DOCX: {e}")

    def _export_memoria_pdf(self):
        """Exporta una Memoria de Cálculo en PDF (A4) con base teórica + resultados + figuras."""
        tab = self.active_tab()
        if tab is self.tab_reactions:
            QMessageBox.information(self, "Memoria PDF", "La pestaña Semirremolque - Reacciones no exporta memoria PDF.")
            return

        try:
            errors = tab._validate_required_inputs()
            if errors:
                head = "No se puede exportar PDF: hay datos requeridos incompletos.\n"
                body = "\n".join([f"- {e}" for e in errors[:12]])
                if len(errors) > 12:
                    body += f"\n- ... y {len(errors) - 12} más."
                QMessageBox.warning(self, "Validación de entradas", head + body)
                return

            config_txt = tab.cmb_config.currentText()
            if config_txt == "":
                raise ValueError("Configuración vacía.")

            Lc = float(_spin_value_or_none(tab.Lc))
            beam_motor = Beam(L_mm=Lc)

            # Cargas puntuales (P) - usuario
            pforces: List[PointForce] = []
            point_forces_user = []
            for r in range(tab.tbl_points.rowCount()):
                lab = _get_text(tab.tbl_points, r, 0).strip()
                x = _try_float(_get_text(tab.tbl_points, r, 1))
                v = _try_float(_get_text(tab.tbl_points, r, 2))
                if lab and x is not None and v is not None:
                    pforces.append(PointForce(label=lab, x_mm=x, value_user=v))
                    point_forces_user.append((lab, float(x), float(v)))

            # Distribuidas conocidas - usuario
            dloads: List[DistUniform] = []
            dist_loads_user = []
            for r in range(tab.tbl_dists.rowCount()):
                lab = _get_text(tab.tbl_dists, r, 0).strip()
                x0 = _try_float(_get_text(tab.tbl_dists, r, 1))
                Lq = _try_float(_get_text(tab.tbl_dists, r, 2))
                q = _try_float(_get_text(tab.tbl_dists, r, 3))
                if lab and x0 is not None and Lq is not None and q is not None:
                    dloads.append(DistUniform(label=lab, x0_mm=x0, Lq_mm=Lq, q_user=q))
                    dist_loads_user.append((lab, float(x0), float(Lq), float(q)))

            # Momentos puntuales - usuario
            moms: List[PointMoment] = []
            point_moments_user = []
            for r in range(tab.tbl_moms.rowCount()):
                lab = _get_text(tab.tbl_moms, r, 0).strip()
                x = _try_float(_get_text(tab.tbl_moms, r, 1))
                m = _try_float(_get_text(tab.tbl_moms, r, 2))
                if lab and x is not None and m is not None:
                    moms.append(PointMoment(label=lab, x_mm=x, M_user_kgmm=m))
                    point_moments_user.append((lab, float(x), float(m)))

            # Apoyos
            kingpin = FixedSupport(
                label="Rp1",
                x_mm=float(_spin_value_or_none(tab.x_front_or_kp)),
                reaction_user=float(_spin_value_or_none(tab.R_front_or_kp))
            )

            tandem = TandemSupport(
                label="Rt",
                reaction_user=float(_spin_value_or_none(tab.Rt)),
            )

            directional = None
            if tab._config_uses_directional():
                directional = DirectionalSupport(
                    label="Rd",
                    reaction_user=float(_spin_value_or_none(tab.Rd)),
                    offset_mm=float(_spin_value_or_none(tab.dir_offset)),
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
                hitch=hitch,
                tandem=tandem,
                directional=directional,
                unknown_uniform=unknown_q,
            )

            # Resolver
            res = solve_equilibrium(case)

            if tab.is_bitren:
                L_viga_total = float(res.x_t_mm) + 2070.0
            else:
                L_viga_total = float(Lc)

            beam_plot = Beam(L_mm=L_viga_total)

            cache = tab.get_cache()
            if cache is None:
                cache = SessionCache(beam_plot=beam_plot, points=[], dists=[], moms=[], note_text="")
                tab.set_cache(cache)

            # Diagramas
            solved_points = res.solved_point_forces
            solved_dists = res.solved_dist_loads
            solved_moms = res.solved_moments

            xlim = _compute_x_view(beam_plot.L_mm, solved_points, solved_dists, solved_moms)
            diag = build_V_M(
                beam_L_mm=beam_plot.L_mm,
                point_forces=solved_points,
                dist_loads=solved_dists,
                moments=solved_moms,
                x_start=xlim[0],
                x_end=xlim[1],
            )

            maxM, minM = self._find_local_extrema_kgcm(diag, xlim)

            def _extremos_V(diag_, xlim_):
                import numpy as np
                x, V, _ = diag_.sample(n_per_segment=220)
                x = np.asarray(x, dtype=float)
                V = np.asarray(V, dtype=float)
                mask = (x >= xlim_[0]) & (x <= xlim_[1])
                x = x[mask]; V = V[mask]
                if x.size < 5:
                    return [], []
                d = np.diff(V)
                maxs = []
                mins = []
                for i in range(1, len(d)):
                    if d[i-1] > 0 and d[i] <= 0:
                        maxs.append((float(x[i]), float(V[i])))
                    if d[i-1] < 0 and d[i] >= 0:
                        mins.append((float(x[i]), float(V[i])))
                return maxs, mins

            maxV, minV = _extremos_V(diag, xlim)

            # Guardar PDF
            default_name = f"Memoria_calculo_{tab.title.replace(' ', '_')}.pdf"
            path, _ = QFileDialog.getSaveFileName(self, "Exportar Memoria de Cálculo (PDF)", default_name, "PDF (*.pdf)")
            if not path:
                return
            if not path.lower().endswith(".pdf"):
                path += ".pdf"

            # Imágenes temporales (desde los ejes actuales)
            tmpdir = tempfile.mkdtemp(prefix="semi_beam_memoria_")
            try:
                fig = self.fig
                renderer = fig.canvas.get_renderer()

                path_fbd = os.path.join(tmpdir, "FBD.jpg")
                fig.savefig(path_fbd, dpi=300, bbox_inches=self.ax_fbd.get_tightbbox(renderer).transformed(fig.dpi_scale_trans.inverted()))

                path_v = os.path.join(tmpdir, "V.jpg")
                fig.savefig(path_v, dpi=300, bbox_inches=self.ax_V.get_tightbbox(renderer).transformed(fig.dpi_scale_trans.inverted()))

                path_m = os.path.join(tmpdir, "M.jpg")
                fig.savefig(path_m, dpi=300, bbox_inches=self.ax_M.get_tightbbox(renderer).transformed(fig.dpi_scale_trans.inverted()))

                # Sincronizar verificación de sección con el diagrama recién resuelto.
                try:
                    tab.section_panel.tbl.clearFocus()
                    tab.section_panel.set_moment_provider(lambda x_mm: float(diag.eval_M(float(x_mm))) / 10.0)
                    tab.section_panel.clear_results_only()
                    tab.section_panel._recompute_all()
                except Exception:
                    pass

                path_sec = os.path.join(tmpdir, "Secciones.jpg")
                try:
                    tab.section_panel.export_table_jpg(path_sec, dpi=300)
                except Exception:
                    path_sec = ""

                # Apoyos y cargas (texto)
                apoyos = []
                apoyos.append(("Rp1", f"x={_fmt_plain(case.kingpin.x_mm, 0)} mm; R={_fmt_plain(case.kingpin.reaction_user, 2)} kg (usuario)"))
                if case.hitch is not None:
                    apoyos.append(("Rp2", f"x={_fmt_plain(case.hitch.x_mm, 0)} mm; R={_fmt_plain(case.hitch.reaction_user, 2)} kg (usuario)"))
                apoyos.append(("Rt", f"x∈[{_fmt_plain(case.tandem.x_min_mm, 0)}, {_fmt_plain(case.tandem.x_max_mm, 0)}] mm; R={_fmt_plain(case.tandem.reaction_user, 2)} kg (usuario)"))
                if case.directional is not None:
                    apoyos.append(("Rd", f"offset={_fmt_plain(case.directional.offset_mm, 0)} mm; R={_fmt_plain(case.directional.reaction_user, 2)} kg (usuario)"))

                cargas = []
                for pf in case.point_forces:
                    cargas.append((pf.label, f"P: x={_fmt_plain(pf.x_mm, 0)} mm; P={_fmt_plain(pf.value_user, 2)} kg (usuario, down+)"))
                for dl in case.dist_loads:
                    cargas.append((dl.label, f"q: x0={_fmt_plain(dl.x0_mm, 0)} mm; L={_fmt_plain(dl.Lq_mm, 0)} mm; q={_fmt_plain(dl.q_user, 6)} kg/mm (usuario, down+)"))
                for pm in case.moments:
                    # ✅ FIX: PointMoment no tiene value_user; es M_user_kgmm
                    cargas.append((pm.label, f"M: x={_fmt_plain(pm.x_mm, 0)} mm; M={_fmt_plain(pm.M_user_kgmm, 2)} kg·mm (usuario)"))
                cargas.append(("q (resuelta)", f"Tramo [0, {_fmt_plain(Lc,0)}] mm; q={_fmt_plain(res.q_user_kg_per_mm, 6)} kg/mm (usuario, down+)"))

                extremos_V = [("MAX", x, v) for x, v in maxV] + [("MIN", x, v) for x, v in minV]
                extremos_M = [("MAX", x, m) for x, m in maxM] + [("MIN", x, m) for x, m in minM]

                # Header UI
                dlg = MemoriaHeaderDialog(self, defaults=cache.memoria_header if cache else {})
                if dlg.exec() != QDialog.DialogCode.Accepted:
                    return
                hdr = dlg.values_dict()
                if cache:
                    cache.memoria_header = hdr

                # ----------------------------
                # Construir MemoriaHeader tolerante
                # ----------------------------
                h_fields = _dc_field_names(MemoriaHeader)
                header_data: Dict[str, Any] = {
                    "titulo": f"Memoria de Cálculo — {tab.title}",
                    "autor": hdr.get("autor", ""),
                    "revision": hdr.get("revision", "A"),
                }
                # Variantes: (cliente, proyecto) vs cliente_proyecto
                cliente = (hdr.get("cliente", "") or "").strip()
                proyecto = (hdr.get("proyecto", "") or "").strip()
                extra_linea = (hdr.get("extra_linea", "") or "").strip()
                if "cliente_proyecto" in h_fields:
                    cp = " - ".join([x for x in [cliente, proyecto] if x]) if (cliente or proyecto) else ""
                    if extra_linea:
                        cp = (cp + " | " + extra_linea).strip(" |")
                    header_data["cliente_proyecto"] = cp
                else:
                    # si existen campos separados, pásalos
                    if "cliente" in h_fields:
                        header_data["cliente"] = cliente
                    if "proyecto" in h_fields:
                        header_data["proyecto"] = proyecto
                    if "extra_linea" in h_fields:
                        header_data["extra_linea"] = extra_linea

                # fecha opcional
                if "fecha" in h_fields:
                    header_data["fecha"] = datetime.now()

                header = _dc_make(MemoriaHeader, header_data)

                # ----------------------------
                # Construir MemoriaCaso tolerante
                # ----------------------------
                c_fields = _dc_field_names(MemoriaCaso)
                if "unidad" in c_fields:
                    caso = MemoriaCaso(
                        unidad=tab.title,
                        L_carrozable_mm=float(Lc),
                        L_viga_total_mm=float(L_viga_total),
                        descripcion_config=config_txt,
                        apoyos=apoyos,
                        cargas=cargas,
                    )
                else:
                    # variante alternativa (si existiera)
                    caso_alt = {
                        "configuracion": tab.title,
                        "Lc_mm": float(Lc),
                        "L_total_mm": float(L_viga_total),
                        "point_forces": point_forces_user,
                        "dist_loads": dist_loads_user,
                        "point_moments": point_moments_user,
                        "descripcion_config": config_txt,
                        "apoyos": apoyos,
                        "cargas": cargas,
                        "unidad": tab.title,
                        "L_carrozable_mm": float(Lc),
                        "L_viga_total_mm": float(L_viga_total),
                    }
                    caso = _dc_make(MemoriaCaso, caso_alt)

                # ----------------------------
                # Construir MemoriaResultados tolerante
                # ----------------------------
                r_fields = _dc_field_names(MemoriaResultados)
                resultados_data = {
                    "q_user_kgmm": float(res.q_user_kg_per_mm),
                    "x_t_mm": float(res.x_t_mm) if res.x_t_mm is not None else 0.0,
                    "x_d_mm": float(res.x_d_mm) if res.x_d_mm is not None else None,
                    "residual_Fy": float(res.residual_Fy),
                    "residual_M0": float(res.residual_M0),
                    "extremos_V": extremos_V,
                    "extremos_M": extremos_M,
                    "extremes_V": extremos_V,
                    "extremes_M": extremos_M,
                }
                resultados = _dc_make(MemoriaResultados, resultados_data)

                # ----------------------------
                # Construir MemoriaSeccion tolerante
                # ----------------------------
                sec_data = tab.section_panel.extract_memoria_data()
                sec_rows = [
                    r for r in sec_data.get("rows", [])
                    if any(str(r.get(k, "") or "").strip() for k in (
                        "x_mm", "h_web_mm", "M_kgcm", "FS", "Jx_cm4", "Wcrit_cm3", "Wreq_cm3", "sigma_max"
                    ))
                ]

                s_fields = _dc_field_names(MemoriaSeccion)
                seccion_obj = None
                if s_fields:
                    if "materiales" in s_fields:
                        seccion_obj = MemoriaSeccion(
                            materiales=[
                                ("Planchuela sup", f"{sec_data.get('material_top','')} / σadm={sec_data.get('sigma_top_kgcm2','')}"),
                                ("Planchuela inf", f"{sec_data.get('material_bot','')} / σadm={sec_data.get('sigma_bot_kgcm2','')}"),
                                ("Alma", f"{sec_data.get('material_web','')} / σadm={sec_data.get('sigma_web_kgcm2','')}"),
                            ],
                            fs_min=float(sec_data.get("fs_min") or 0.0),
                            n_vigas=int(sec_data.get("n_beams") or 2),
                            parametros=[
                                ("t_top [in]", str(sec_data.get("t_top_in",""))),
                                ("t_bot [in]", str(sec_data.get("t_bot_in",""))),
                                ("b_f", str(sec_data.get("bf_text",""))),
                            ],
                            tabla=[
                                ["Sec", "x", "h_web", "t_web", "M", "FS", "Jx", "ybar", "cmax", "Wcrit", "Wreq", "σmax"],
                                *[
                                    [
                                        r.get("sec",""),
                                        r.get("x_mm",""),
                                        r.get("h_web_mm",""),
                                        r.get("t_web_in",""),
                                        r.get("M_kgcm",""),
                                        r.get("FS",""),
                                        r.get("Jx_cm4",""),
                                        r.get("ybar_cm",""),
                                        r.get("cmax_cm",""),
                                        r.get("Wcrit_cm3",""),
                                        r.get("Wreq_cm3",""),
                                        r.get("sigma_max",""),
                                    ]
                                    for r in sec_rows
                                ]
                            ],
                        )
                    else:
                        # otra variante posible: material consolidado + sigma + imagen tabla
                        mat_top = str(sec_data.get("material_top", "")).strip()
                        mat_bot = str(sec_data.get("material_bot", "")).strip()
                        mat_web = str(sec_data.get("material_web", "")).strip()
                        material_txt = " / ".join([m for m in [mat_top, mat_bot, mat_web] if m]) or "—"

                        def _to_float_or_none(v):
                            if v is None:
                                return None
                            try:
                                s = str(v).strip()
                                if s == "":
                                    return None
                                return float(s)
                            except Exception:
                                return None

                        sigmas = [
                            _to_float_or_none(sec_data.get("sigma_top_kgcm2")),
                            _to_float_or_none(sec_data.get("sigma_bot_kgcm2")),
                            _to_float_or_none(sec_data.get("sigma_web_kgcm2")),
                        ]
                        sigmas = [s for s in sigmas if s is not None]
                        sigma_adm = min(sigmas) if sigmas else None

                        tabla_img_path = None
                        try:
                            tabla_img_path = os.path.join(tempfile.gettempdir(), f"memoria_seccion_{tab.title}.jpg")
                            tab.section_panel.export_table_jpg(tabla_img_path, dpi=300)
                        except Exception:
                            tabla_img_path = None

                        seccion_obj = _dc_make(MemoriaSeccion, {
                            "material": material_txt,
                            "sigma_adm_kgcm2": sigma_adm,
                            "n_vigas": int(sec_data.get("n_beams", 2) or 2),
                            "fs_min": _to_float_or_none(sec_data.get("fs_min")),
                            "tabla_imagen_path": tabla_img_path,
                        })

                # ----------------------------
                # Exportar PDF (tolerante a firma del exportador)
                # ----------------------------
                imgs = {
                    "fbd": path_fbd,
                    "v": path_v,
                    "m": path_m,
                    "secciones": path_sec if path_sec else "",
                }

                try:
                    export_memoria_pdf(
                        path,
                        header=header,
                        caso=caso,
                        resultados=resultados,
                        seccion=seccion_obj,
                        imagenes=imgs,
                    )
                except TypeError:
                    try:
                        export_memoria_pdf(
                            path,
                            header=header,
                            caso=caso,
                            resultados=resultados,
                            seccion=seccion_obj,
                            images=imgs,  # posible variante
                        )
                    except TypeError:
                        try:
                            # posible variante posicional
                            export_memoria_pdf(path, header, caso, resultados, seccion_obj, imgs)
                        except TypeError:
                            # último recurso: sin imágenes
                            export_memoria_pdf(path, header=header, caso=caso, resultados=resultados, seccion=seccion_obj)

                QMessageBox.information(self, "Memoria de cálculo", f"PDF generado:\n{path}")

            finally:
                try:
                    import shutil
                    shutil.rmtree(tmpdir, ignore_errors=True)
                except Exception:
                    pass

        except Exception as e:
            import logging
            import traceback

            logger = logging.getLogger("semi_beam")
            try:
                logger.error("Error al generar la memoria (PDF).\n%s", traceback.format_exc())
            except Exception:
                print(traceback.format_exc())

            QMessageBox.critical(self, "Memoria de cálculo", f"Error al generar la memoria: {e}")


def main():
    app = QApplication(sys.argv)
    app.setApplicationName("calculeitor")
    try:
        app.setWindowIcon(QIcon(ensure_calculeitor_icon()))
    except Exception:
        pass
    w = FBDApp()
    w.show()
    sys.exit(app.exec())
