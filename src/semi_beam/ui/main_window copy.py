from __future__ import annotations

import sys
import os
import tempfile
from datetime import datetime
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import matplotlib
matplotlib.use("QtAgg")
import matplotlib.pyplot as plt

from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QWheelEvent
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QFormLayout,
    QLabel, QDoubleSpinBox, QPushButton, QTableWidget, QTableWidgetItem,
    QSizePolicy, QSplitter, QScrollArea, QTabWidget, QMessageBox, QFileDialog,
    QToolButton, QFrame, QComboBox, QDialog
)
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas

# Delegados numéricos (solo números / admite vacío)
from semi_beam.ui.numeric_delegate import NullableFloatDelegate

# ---- Dominio / motor / view ----
from semi_beam.domain.beam import Beam
from semi_beam.domain.loads import PointForce, DistUniform, PointMoment
from semi_beam.domain.labels import p_index, next_free_p_index
from semi_beam.engine.normalize import normalize_inputs
from semi_beam.view.style import RenderStyle
from semi_beam.view.renderer_fbd import render_fbd

from semi_beam.domain.supports import FixedSupport, TandemSupport, DirectionalSupport
from semi_beam.domain.unknowns import UnknownUniformLoad
from semi_beam.domain.cases import BeamCase
from semi_beam.engine.equilibrium import solve_equilibrium
from semi_beam.engine.diagrams import build_V_M
from semi_beam.view.renderer_vm import render_shear, render_moment
from semi_beam.services.memoria_calculo_pdf import (
    export_memoria_pdf, MemoriaHeader, MemoriaCaso, MemoriaResultados, MemoriaSeccion
)

# ---- Verificador (TU UI anterior) ----
from semi_beam.ui.section_check_panel import SectionCheckPanel
from semi_beam.ui.memoria_header_dialog import MemoriaHeaderDialog



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


# path: src/semi_beam/ui/main_window.py
def _fmt_plain(v, decimals: int = 2) -> str:
    """Formatea números para UI/texto. Tolerante a None (devuelve '-')."""
    if v is None:
        return "-"
    try:
        s = f"{float(v):.{decimals}f}"
    except Exception:
        # Fallback defensivo: si no es convertible a float, devolver string directo
        return str(v)
    if "." in s:
        s = s.rstrip("0").rstrip(".")
    return s



@dataclass
class SessionCache:
    beam_plot: Beam
    points: List[PointForce]
    dists: List[DistUniform]
    moms: List[PointMoment]
    note_text: str
    memoria_header: dict = field(default_factory=dict)



# ============================================================
# Un TAB completo (estado independiente)
# ============================================================
class UnitTab(QWidget):
    """
    Cada tab es una sesión independiente:
    - Motor con selector de configuración de ejes (único)
    - Tablas de cargas
    - Verificador de sección (TU UI)
    - Notas
    """
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
        self.Lc = QDoubleSpinBox()
        self.Lc.setRange(1.0, 1e12)
        self.Lc.setDecimals(2)
        self.Lc.setSingleStep(50.0)
        self.Lc.setValue(10365.0)

        self.x_front_or_kp = QDoubleSpinBox()
        self.x_front_or_kp.setRange(-1e12, 1e12)
        self.x_front_or_kp.setDecimals(2)
        self.x_front_or_kp.setSingleStep(50.0)
        self.x_front_or_kp.setValue(2200.0)

        self.R_front_or_kp = QDoubleSpinBox()
        self.R_front_or_kp.setRange(-1e12, 1e12)
        self.R_front_or_kp.setDecimals(2)
        self.R_front_or_kp.setSingleStep(50.0)
        self.R_front_or_kp.setValue(14500.0 if self.is_bitren else 9000.0)

        self.Rt = QDoubleSpinBox()
        self.Rt.setRange(-1e12, 1e12)
        self.Rt.setDecimals(2)
        self.Rt.setSingleStep(50.0)
        self.Rt.setValue(22200.0)

        # Direccional (semi/bitren)
        self.Rd = QDoubleSpinBox()
        self.Rd.setRange(-1e12, 1e12)
        self.Rd.setDecimals(2)
        self.Rd.setSingleStep(50.0)
        self.Rd.setValue(0.0)

        self.dir_offset = QDoubleSpinBox()
        self.dir_offset.setRange(0.0, 20000.0)
        self.dir_offset.setDecimals(1)
        self.dir_offset.setSingleStep(25.0)
        self.dir_offset.setValue(3075.0)

        # Bitren Rp2
        self.x_rp2_rel = QDoubleSpinBox()
        self.x_rp2_rel.setRange(-1e12, 1e12)
        self.x_rp2_rel.setDecimals(2)
        self.x_rp2_rel.setSingleStep(50.0)
        self.x_rp2_rel.setValue(2100.0)

        self.Rp2 = QDoubleSpinBox()
        self.Rp2.setRange(-1e12, 1e12)
        self.Rp2.setDecimals(2)
        self.Rp2.setSingleStep(50.0)
        self.Rp2.setValue(13200.0)

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
        btns.addWidget(self.btn_solve)
        btns.addWidget(self.btn_clear)
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
        self.btn_add_q.clicked.connect(lambda: self._add_row(self.tbl_dists, ["q", "", "", ""]))
        self.btn_del_q.clicked.connect(lambda: self._remove_selected_rows(self.tbl_dists))
        self.btn_add_m.clicked.connect(lambda: self._add_row(self.tbl_moms, ["M", "", ""]))
        self.btn_del_m.clicked.connect(lambda: self._remove_selected_rows(self.tbl_moms))

        # Señales config (autocompletar reacciones)
        self.cmb_config.currentIndexChanged.connect(lambda _i: self._apply_config_defaults())
        self.cmb_semi_tipo.currentIndexChanged.connect(lambda _i: self._populate_configs())

        # aplicar defaults iniciales
        self._apply_config_defaults()

        # ✅ Asegurar que TODOS los CollapsibleBox del tab arranquen contraídos
        for b in self._all_boxes:
            b.set_collapsed(True)

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
                    "3 ejes conv — Rt 22200",
                    "3 ejes neum — Rt 23475",
                    "1 + 2 ejes — Rd 9200 (offset 3075) + Rt 15800",
                    "1 + 3 ejes — Rd 9200 (offset 3700) + Rt 22200",
                ])
                self.cmb_config.setCurrentIndex(0)
            else:
                self.cmb_config.addItems([
                    "1 eje — Rt 9200",
                    "2 ejes — Rt 15800",
                    "3 ejes conv — Rt 22200",
                    "3 ejes neum — Rt 23475",
                    "1 + 2 ejes — Rd 9200 (offset 3075) + Rt 15800",
                ])
                self.cmb_config.setCurrentIndex(0)

        self.cmb_config.blockSignals(False)
        self._apply_config_defaults()

    def _apply_config_defaults(self):
        """
        Setea reacciones según configuración seleccionada.
        Los spinbox quedan editables (el usuario puede ajustar).
        """
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
                elif "Rt 9200" in cfg:
                    self.Rt.setValue(9200.0)

    # ---- helpers UI tablas
    def _add_row(self, tbl: QTableWidget, values: List[str]):
        r = tbl.rowCount()
        tbl.insertRow(r)
        for c, v in enumerate(values):
            _set_item(tbl, r, c, v)

    def _remove_selected_rows(self, tbl: QTableWidget):
        rows = sorted(set([i.row() for i in tbl.selectedItems()]), reverse=True)
        for rr in rows:
            tbl.removeRow(rr)

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

    # ---- parsing entradas
    def parse_inputs(self) -> Tuple[Beam, List[PointForce], List[DistUniform], List[PointMoment], List[str]]:
        notes: List[str] = []
        Lc = float(self.Lc.value())
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

    # ---- cache / diag
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
        else:
            self.section_panel.set_moment_provider(lambda x_mm: float(diag.eval_M(float(x_mm))) / 10.0)
            self.section_panel.clear_results_only()

    def get_diag(self):
        return self._last_diag


# ============================================================
# MAIN WINDOW
# ============================================================
class FBDApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("semi_beam — Acoplado / Semirremolque / Bitren")
        self.resize(1500, 850)

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

        self.tabs.addTab(self.tab_acoplado, "Acoplado")
        self.tabs.addTab(self.tab_semi, "Semirremolque")
        self.tabs.addTab(self.tab_bitren, "Bitren")

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
        gs = self.fig.add_gridspec(3, 1, height_ratios=[1.35, 1.0, 1.0], hspace=0.60)
        self.ax_fbd = self.fig.add_subplot(gs[0, 0])
        self.ax_V = self.fig.add_subplot(gs[1, 0], sharex=self.ax_fbd)
        self.ax_M = self.fig.add_subplot(gs[2, 0], sharex=self.ax_fbd)

        self.canvas = FigureCanvas(self.fig)
        self.canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.canvas.setMinimumHeight(800)

        self.plot_scroll = QScrollArea()
        self.plot_scroll.setWidgetResizable(True)
        self.plot_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.plot_scroll.setWidget(self.canvas)

        btn_row = QHBoxLayout()
        self.btn_export_plots = QPushButton("Exportar gráficos (FBD, V(x), M(x))")
        self.btn_export_memoria = QPushButton("Exportar memoria de cálculo (PDF)")
        btn_row.addWidget(self.btn_export_plots)
        btn_row.addWidget(self.btn_export_memoria)
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
        # ✅ conexiones por-tab con invalidación de solución + auto-refresh FBD
        for tab in [self.tab_acoplado, self.tab_semi, self.tab_bitren]:
            tab.btn_solve.clicked.connect(lambda _=False, t=tab: self._solve_for_tab(t))
            tab.btn_clear.clicked.connect(lambda _=False, t=tab: self._plot_inputs_for_tab(t))

            # cualquier cambio en inputs => invalida solución y repinta entradas
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

            tab.cmb_config.currentIndexChanged.connect(lambda *_ , t=tab: self._schedule_replot_tab(t, reset_solution=True))
            if hasattr(tab, "cmb_semi_tipo"):
                tab.cmb_semi_tipo.currentIndexChanged.connect(lambda *_ , t=tab: self._schedule_replot_tab(t, reset_solution=True))

            # ✅ tablas: usar cellChanged (más confiable con delegates)
            tab.tbl_points.cellChanged.connect(lambda *_ , t=tab: self._schedule_replot_tab(t, reset_solution=True))
            tab.tbl_dists.cellChanged.connect(lambda *_ , t=tab: self._schedule_replot_tab(t, reset_solution=True))
            tab.tbl_moms.cellChanged.connect(lambda *_ , t=tab: self._schedule_replot_tab(t, reset_solution=True))

        self.tabs.currentChanged.connect(lambda _i: self._replot_active_tab())
        self.btn_export_plots.clicked.connect(self._export_plots_jpg_1200)
        self.btn_export_memoria.clicked.connect(self._export_memoria_pdf)

        self._plot_inputs_for_tab(self.active_tab())

        self._resize_timer = QTimer(self)
        self._resize_timer.setSingleShot(True)
        self._resize_timer.timeout.connect(self._replot_active_tab)
        self.canvas.mpl_connect("resize_event", lambda evt: self._resize_timer.start(80))

    def active_tab(self) -> UnitTab:
        return self.tabs.currentWidget()

    def eventFilter(self, obj, event):
        if obj is self.canvas and isinstance(event, QWheelEvent):
            bar = self.plot_scroll.verticalScrollBar()
            bar.setValue(bar.value() - event.angleDelta().y())
            return True
        return super().eventFilter(obj, event)

    def _schedule_replot_tab(self, tab: UnitTab, *, reset_solution: bool):
        if reset_solution:
            tab.set_cache(None)
            tab.set_diag(None)  # limpia provider/tabla si hacía falta
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
        # “relleno” de ceros para no inventar extremos en tramos planos
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


        self.fig.subplots_adjust(left=0.07, right=0.985, top=0.96, bottom=0.06, hspace=0.65)
        self.canvas.draw()

        if set_diag_on_tab is not None:
            set_diag_on_tab.set_diag(diag)

    def _replot_active_tab(self):
        tab = self.active_tab()
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

            cache = SessionCache(beam_plot=beam, points=points, dists=dists, moms=moms, note_text=note)
            tab.set_cache(cache)
            tab.set_note(note)
            self._plot_triplet(cache, set_diag_on_tab=tab)

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Error al graficar entradas: {e}")
            tab.set_cache(None)
            tab.set_note(f"Error: {e}")
            tab.set_diag(None)

    # Solve equilibrium
    def _solve_for_tab(self, tab: UnitTab):
        try:
            beam_motor, pforces, dloads, moms, pnotes = tab.parse_inputs()
            Lc = float(tab.Lc.value())

            kingpin = FixedSupport(
                label="Rp1",
                x_mm=float(tab.x_front_or_kp.value()),
                reaction_user=float(tab.R_front_or_kp.value())
            )
            tandem = TandemSupport(label="Rt", reaction_user=float(tab.Rt.value()))

            directional = None
            if (not tab.is_acoplado) and abs(float(tab.Rd.value())) > 0.0:
                directional = DirectionalSupport(
                    label="Rd",
                    reaction_user=float(tab.Rd.value()),
                    offset_mm=float(tab.dir_offset.value())
                )

            hitch = None
            x_rp2_abs = None
            if tab.is_bitren:
                x_rp2_abs = Lc + float(tab.x_rp2_rel.value())
                hitch = FixedSupport(label="Rp2", x_mm=x_rp2_abs, reaction_user=float(tab.Rp2.value()))

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

            # Largo de viga:
            # bitren: L = x_t + 2070
            # otros: L = Lc
            if tab.is_bitren:
                L_viga_total = float(res.x_t_mm) + 2070.0
            else:
                L_viga_total = float(Lc)

            beam_plot = Beam(L_mm=L_viga_total)

            # armar cache solución
            cache = SessionCache(
                beam_plot=beam_plot,
                points=res.solved_point_forces,
                dists=res.solved_dist_loads,
                moms=res.solved_moments,
                note_text="",
            )

            # construir diag para extremos y notas
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

            if pnotes:
                note_lines.append("Notas:")
                for n in pnotes:
                    note_lines.append(f"- {n}")

            note = "\n".join(note_lines)
            cache.note_text = note

            tab.set_cache(cache)
            tab.set_note(note)

            # ploteo (y set moment provider en panel)
            self._plot_triplet(cache, set_diag_on_tab=tab)

            # ✅ al resolver, recalcula panel (si ya hay datos cargados)
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


    def _export_memoria_pdf(self):
        """Exporta una Memoria de Cálculo en PDF (A4) con base teórica + resultados + figuras."""
        tab = self.active_tab()

        # Recalcular solución para asegurar consistencia (no depende del estado del gráfico)
        try:
            # 1) Construir el caso exactamente como en _solve_for_tab (motor)
            config = tab.cmb_config.currentText()
            if config == "":
                raise ValueError("Configuración vacía.")

            Lc = float(tab.Lc.value())

            # Beam motor (carrozable). La viga total se define después de resolver (según reglas actuales).
            beam_motor = Beam(L_mm=Lc)


            # Cargas puntuales (P)
            pforces: List[PointForce] = []
            for r in range(tab.tbl_points.rowCount()):
                lab = _get_text(tab.tbl_points, r, 0).strip()
                x = _try_float(_get_text(tab.tbl_points, r, 1))
                v = _try_float(_get_text(tab.tbl_points, r, 2))
                if lab and x is not None and v is not None:
                    pforces.append(PointForce(label=lab, x_mm=x, value_user=v))

            # Cargas distribuidas conocidas
            dloads: List[DistUniform] = []
            for r in range(tab.tbl_dists.rowCount()):
                lab = _get_text(tab.tbl_dists, r, 0).strip()
                x0 = _try_float(_get_text(tab.tbl_dists, r, 1))
                Lq = _try_float(_get_text(tab.tbl_dists, r, 2))
                q = _try_float(_get_text(tab.tbl_dists, r, 3))
                if lab and x0 is not None and Lq is not None and q is not None:
                    dloads.append(DistUniform(label=lab, x0_mm=x0, Lq_mm=Lq, q_user=q))

            # Momentos puntuales
            moms: List[PointMoment] = []
            for r in range(tab.tbl_moms.rowCount()):
                lab = _get_text(tab.tbl_moms, r, 0).strip()
                x = _try_float(_get_text(tab.tbl_moms, r, 1))
                m = _try_float(_get_text(tab.tbl_moms, r, 2))
                if lab and x is not None and m is not None:
                    # PointMoment define el campo de valor como M_user_kgmm
                    moms.append(PointMoment(label=lab, x_mm=x, M_user_kgmm=m))

            # Apoyos: kingpin/frente, tándem (Rt), direccional (Rd) opcional, hitch (Rp2) en bitren
            kingpin = FixedSupport(
                label="Rp1" if tab.is_acoplado else "Rp1",
                x_mm=float(tab.x_front_or_kp.value()),
                reaction_user=float(tab.R_front_or_kp.value())
            )

            tandem = TandemSupport(
                label="Rt",
                reaction_user=float(tab.Rt.value()),
            )

            directional = None
            if (not tab.is_acoplado) and abs(float(tab.Rd.value())) > 0.0:
                directional = DirectionalSupport(
                    label="Rd",
                    reaction_user=float(tab.Rd.value()),
                    offset_mm=float(tab.dir_offset.value()),
                )

            hitch = None
            x_rp2_abs = None
            if tab.is_bitren:
                x_rp2_abs = Lc + float(tab.x_rp2_rel.value())
                hitch = FixedSupport(label="Rp2", x_mm=x_rp2_abs, reaction_user=float(tab.Rp2.value()))

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

            # 2) Resolver equilibrio y construir diagramas
            res = solve_equilibrium(case)

            # Largo de viga total (misma regla que _solve_for_tab)
            # bitren: L = x_t + 2070
            # otros: L = Lc
            if tab.is_bitren:
                L_viga_total = float(res.x_t_mm) + 2070.0
            else:
                L_viga_total = float(Lc)

            beam_plot = Beam(L_mm=L_viga_total)

            cache = SessionCache(
                beam_plot=beam_plot,
                points=res.solved_point_forces,
                dists=res.solved_dist_loads,
                moms=res.solved_moments,
                note_text="",
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

            # Extremos M (ya existe helper)
            maxM, minM = self._find_local_extrema_kgcm(diag, xlim)

            # Extremos V (cálculo local por muestreo)
            def _extremos_V(diag_, xlim_):
                import numpy as np
                x, V, _ = diag_.sample(n_per_segment=220)  # V en kg (interno)
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

            # 3) Pedir destino
            default_name = f"Memoria_calculo_{tab.title.replace(' ', '_')}.pdf"
            path, _ = QFileDialog.getSaveFileName(self, "Exportar Memoria de Cálculo (PDF)", default_name, "PDF (*.pdf)")
            if not path:
                return
            if not path.lower().endswith(".pdf"):
                path += ".pdf"

            # 4) Generar imágenes temporales (FBD, V, M, tabla secciones)
            tmpdir = tempfile.mkdtemp(prefix="semi_beam_memoria_")
            try:
                # Guardar cada subplot como imagen independiente
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

                # 5) Armar datos para el PDF
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
                    cargas.append((pm.label, f"M: x={_fmt_plain(pm.x_mm, 0)} mm; M={_fmt_plain(pm.value_user, 2)} kg·mm (usuario)"))
                cargas.append(("q (resuelta)", f"Tramo [0, {_fmt_plain(Lc,0)}] mm; q={_fmt_plain(res.q_user_kg_per_mm, 6)} kg/mm (usuario, down+)"))

                extremos_V = [("MAX", x, v) for x, v in maxV] + [("MIN", x, v) for x, v in minV]
                extremos_M = [("MAX", x, m) for x, m in maxM] + [("MIN", x, m) for x, m in minM]

                dlg = MemoriaHeaderDialog(self, defaults=cache.memoria_header if cache else {})
                if dlg.exec() != QDialog.DialogCode.Accepted:
                    return
                hdr = dlg.values_dict()
                if cache:
                    cache.memoria_header = hdr

                # --- HEADER (OK como lo tenés) ---
                header = MemoriaHeader(
                    titulo=f"Memoria de Cálculo — {tab.title}",
                    cliente=hdr.get("cliente", ""),
                    proyecto=hdr.get("proyecto", ""),
                    autor=hdr.get("autor", ""),
                    revision=hdr.get("revision", "A"),
                    extra_linea=hdr.get("extra_linea", ""),
                )

                # Helpers locales (para evitar float(None) / strings vacíos)
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

                # -------------------------------------------------------------------------
                # MEMORIA CASO (ajustado a dataclass real)
                # Deben existir estas listas (formato usuario):
                #   point_forces_user: List[(label, x_mm, P_user)]
                #   dist_loads_user:   List[(label, x0_mm, L_mm, q_user)]
                #   point_moments_user:List[(label, x_mm, M_user)]
                # Si hoy no las tenés, armálas igual que cuando construís `case` para solver.
                # -------------------------------------------------------------------------
                caso = MemoriaCaso(
                    configuracion=tab.title,                 # o `config` si preferís describir la config textual
                    Lc_mm=float(Lc),
                    L_total_mm=float(L_viga_total),
                    point_forces=point_forces_user,
                    dist_loads=dist_loads_user,
                    point_moments=point_moments_user,
                )

                # -------------------------------------------------------------------------
                # MEMORIA RESULTADOS (ajustado a dataclass real)
                # OJO: x_t_mm puede ser None en algunos casos → NO hacer float(None)
                # -------------------------------------------------------------------------
                resultados = MemoriaResultados(
                    q_user_kgmm=float(res.q_user_kg_per_mm),
                    x_t_mm=_to_float_or_none(res.x_t_mm),
                    x_d_mm=_to_float_or_none(res.x_d_mm),
                    residual_Fy=float(res.residual_Fy),
                    residual_M0=float(res.residual_M0),
                    extremes_V=extremos_V,   # tus variables existentes, solo cambia el nombre del kw
                    extremes_M=extremos_M,
                )

                # -------------------------------------------------------------------------
                # MEMORIA SECCIÓN (ajustado a dataclass real)
                # MemoriaSeccion en el exportador es SIMPLE:
                #   material: str
                #   sigma_adm_kgcm2: Optional[float]
                #   n_vigas: int
                #   fs_min: Optional[float]
                #   tabla_imagen_path: Optional[str]
                # -------------------------------------------------------------------------
                sec_data = tab.section_panel.extract_memoria_data()

                # material consolidado (si tenés 3 materiales)
                mat_top = str(sec_data.get("material_top", "")).strip()
                mat_bot = str(sec_data.get("material_bot", "")).strip()
                mat_web = str(sec_data.get("material_web", "")).strip()
                material_txt = " / ".join([m for m in [mat_top, mat_bot, mat_web] if m])

                # sigma adm: si tu extract devuelve 3, tomamos la mínima disponible
                sigmas = [
                    _to_float_or_none(sec_data.get("sigma_top_kgcm2")),
                    _to_float_or_none(sec_data.get("sigma_bot_kgcm2")),
                    _to_float_or_none(sec_data.get("sigma_web_kgcm2")),
                ]
                sigmas = [s for s in sigmas if s is not None]
                sigma_adm = min(sigmas) if sigmas else None

                # Export de tabla a imagen (si tu panel lo soporta). Si falla, seguimos sin tabla.
                tabla_img_path = None
                try:
                    import os
                    import tempfile
                    tabla_img_path = os.path.join(tempfile.gettempdir(), f"memoria_seccion_{tab.title}.jpg")
                    tab.section_panel.export_table_jpg(tabla_img_path, dpi=300)
                except Exception:
                    tabla_img_path = None

                seccion = MemoriaSeccion(
                    material=material_txt or "—",
                    sigma_adm_kgcm2=sigma_adm,
                    n_vigas=int(sec_data.get("n_beams", 2) or 2),
                    fs_min=_to_float_or_none(sec_data.get("fs_min")),
                    tabla_imagen_path=tabla_img_path,
                )


                export_memoria_pdf(
                    path,
                    header=header,
                    caso=caso,
                    resultados=resultados,
                    seccion=seccion,
                    imagenes={
                        "fbd": path_fbd,
                        "v": path_v,
                        "m": path_m,
                        "secciones": path_sec if path_sec else "",
                    },
                )

                QMessageBox.information(self, "Memoria de cálculo", f"PDF generado:\n{path}")
            finally:
                try:
                    # limpieza best-effort
                    import shutil
                    shutil.rmtree(tmpdir, ignore_errors=True)
                except Exception:
                    pass

        except Exception as e:
            # Importante: este error suele estar atrapado (no dispara sys.excepthook),
            # por lo que registramos traceback completo en el logger de la app.
            import logging
            import traceback

            logger = logging.getLogger("semi_beam")
            try:
                logger.error("Error al generar la memoria (PDF).\n%s", traceback.format_exc())
            except Exception:
                # fallback si el logger no está inicializado
                print(traceback.format_exc())

            QMessageBox.critical(self, "Memoria de cálculo", f"Error al generar la memoria: {e}")


def main():
    app = QApplication(sys.argv)
    w = FBDApp()
    w.show()
    sys.exit(app.exec())
