from __future__ import annotations

import math
import os
import tempfile
from typing import Any, Callable, Dict, Optional, List

from PySide6.QtCore import Qt, QTimer, Signal
from PySide6.QtGui import QColor, QBrush
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QFormLayout, QLabel,
    QComboBox, QPushButton, QTableWidget, QTableWidgetItem, QSizePolicy,
    QFileDialog, QMessageBox, QStyle, QCheckBox, QAbstractItemView
)

import matplotlib
matplotlib.use("QtAgg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backend_bases import MouseButton
from matplotlib.patches import FancyArrowPatch

from semi_beam.sections.i_section import CompositeSection, ISection, IN_TO_MM, SectionRect
from semi_beam.sections.flex_check import M_ZERO_TOL_KGCM, compute_flex_row
from semi_beam.services.memoria_calculo_docx import (
    MemoriaHeader,
    MemoriaSeccion,
    export_memoria_docx,
)
from semi_beam.ui.numeric_delegate import (
    NullableFloatDelegate,
    SpinBoxDelegate,
    FlexibleDoubleSpinBox,
    apply_table_readability_style,
    combo_cell_style,
    TABLE_ERROR_BG,
    TABLE_INPUT_BG,
    TABLE_OK_BG,
    TABLE_READONLY_BG,
    TABLE_TEXT_COLOR,
)
from semi_beam.ui.number_parsing import try_parse_user_float

from semi_beam.materials.material_db import MaterialDB, default_materials_path


BASTIDOR_LATERAL_DISTANCIA_MM = 1250.0
BASTIDOR_LATERAL_ESPESOR_MM = 4.7625
BASTIDOR_LATERAL_ALA_MM = 45.0
BASTIDOR_LATERAL_ALTURA_DEFAULT_MM = 170.0
BASTIDOR_LATERAL_ALTURA_MIN_MM = 130.0
BASTIDOR_LATERAL_ALTURA_MAX_MM = 170.0
PISO_ANCHO_MM = 2430.0
PISO_ESPESOR_OPTIONS_MM = (
    ("2 mm", 2.0),
    ("3 mm", 3.0),
    ("4 mm", 4.0),
    ('1/8" - 3.175 mm', 3.175),
    ('3/16" - 4.7625 mm', 4.7625),
)
CHAPON_MATERIAL_ID = "SAE1010"
CHAPON_ANCHO_MM = 1050.0
CHAPON_LENGTH_DEFAULT_MM = 2000.0
CHAPON_ESPESOR_OPTIONS_MM = (
    ('1/4" - 6.35 mm', 6.35),
    ('5/16" - 7.9375 mm', 7.9375),
    ('3/8" - 9.525 mm', 9.525),
)
VIGA_LEFT_FACE_INTERIOR_MM = -389.0
VIGA_LEFT_FACE_EXTERIOR_MM = -516.0
VIGA_RIGHT_FACE_INTERIOR_MM = 389.0
VIGA_RIGHT_FACE_EXTERIOR_MM = 516.0
FLEX_NO_DEMAND_TEXT = "Sin demanda a flexión"


def _in_to_mm(v_in: float) -> float:
    return float(v_in) * IN_TO_MM


def _parse_frac_in(text: str) -> float:
    t = (text or "").strip()
    if "/" in t:
        a, b = t.split("/")
        return float(a) / float(b)
    return float(t)


def _set_item(tbl: QTableWidget, r: int, c: int, value: str):
    it = QTableWidgetItem(value)
    it.setTextAlignment(Qt.AlignCenter)
    tbl.setItem(r, c, it)


def _set_item_editable(tbl: QTableWidget, r: int, c: int, editable: bool):
    it = tbl.item(r, c)
    if it is None:
        return
    flags = it.flags()
    if editable:
        it.setFlags(flags | Qt.ItemIsEditable)
    else:
        it.setFlags(flags & ~Qt.ItemIsEditable)


def _get_text(tbl: QTableWidget, r: int, c: int) -> str:
    it = tbl.item(r, c)
    return "" if it is None else (it.text() or "").strip()


def _try_float(s: str) -> Optional[float]:
    return try_parse_user_float(s)


def _fmt2(v: float) -> str:
    s = f"{float(v):.2f}"
    if "." in s:
        s = s.rstrip("0").rstrip(".")
    return s


def _fmt_int(v: float) -> str:
    return str(int(round(float(v), 0)))


def _fmt_in_mm(text_in: str) -> str:
    v_in = _parse_frac_in(text_in)
    return f"{text_in} - {_fmt2(_in_to_mm(v_in))} mm"


def _is_no_flex_demand(M_kgcm: Optional[float]) -> bool:
    return M_kgcm is not None and abs(float(M_kgcm)) <= M_ZERO_TOL_KGCM


class DoubleWebGeometryError(ValueError):
    pass


def build_web_rects(
    *,
    b_f_mm: float,
    t_web_mm: float,
    h_web_mm: float,
    y0_mm: float,
    double_web_enabled: bool,
    double_web_inner_face_offset_mm: Optional[float],
    x_center_mm: float = 0.0,
    label_prefix: str = "alma",
) -> list[SectionRect]:
    b_f = float(b_f_mm)
    t = float(t_web_mm)
    h = float(h_web_mm)
    y0 = float(y0_mm)
    xc = float(x_center_mm)

    if not double_web_enabled:
        return [SectionRect(xc - t / 2.0, y0, t, h, label_prefix)]

    if double_web_inner_face_offset_mm is None:
        raise DoubleWebGeometryError("Falta distancia de doble alma.")
    d = float(double_web_inner_face_offset_mm)
    if d < 0.0:
        raise DoubleWebGeometryError("La distancia de doble alma no puede ser negativa.")
    if d + t > b_f / 2.0:
        raise DoubleWebGeometryError("La doble alma excede el ancho de la planchuela.")

    return [
        SectionRect(xc - d - t, y0, t, h, f"{label_prefix}_izq"),
        SectionRect(xc + d, y0, t, h, f"{label_prefix}_der"),
    ]


def _configure_combo_for_contents(cmb: QComboBox) -> None:
    cmb.setSizeAdjustPolicy(QComboBox.AdjustToContents)
    cmb.setMinimumContentsLength(0)
    cmb.setMinimumWidth(0)
    cmb.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)

    def _apply_width() -> None:
        text = cmb.currentText() or ""
        fm = cmb.fontMetrics()
        text_w = fm.horizontalAdvance(text)
        frame_w = max(cmb.style().pixelMetric(QStyle.PixelMetric.PM_DefaultFrameWidth, None, cmb), 1)
        arrow_w = 28
        padding_w = 18
        cmb.setFixedWidth(max(56, text_w + (2 * frame_w) + arrow_w + padding_w))

    cmb.currentTextChanged.connect(lambda _t: _apply_width())
    _apply_width()


def _configure_table_combo_for_cell(cmb: QComboBox) -> None:
    cmb.setSizeAdjustPolicy(QComboBox.AdjustToMinimumContentsLengthWithIcon)
    cmb.setMinimumContentsLength(0)
    cmb.setMinimumWidth(0)
    cmb.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)


class SectionCheckPanel(QWidget):
    inertia_inputs_changed = Signal()

    COL_SEC = 0
    COL_X = 1
    COL_HWEB = 2
    COL_TWEB = 3
    COL_DOUBLE_WEB = 4
    COL_DOUBLE_WEB_OFFSET = 5
    COL_FS = 6
    COL_M = 7

    COL_JX = 8
    COL_YBAR = 9
    COL_CMAX = 10
    COL_WCRIT = 11
    COL_WREQ = 12
    COL_SIGMAX = 13

    INPUT_ITEM_COLUMNS = (COL_X, COL_HWEB, COL_DOUBLE_WEB_OFFSET)
    RESULT_COLUMNS = (COL_FS, COL_M, COL_JX, COL_YBAR, COL_CMAX, COL_WCRIT, COL_WREQ, COL_SIGMAX)

    TFLANGE_OPTIONS = ["5/16", "3/8", "7/16", "1/2", "5/8", "3/4"]
    TWEB_OPTIONS = ["3/16", "1/4", "5/16", "3/8", "7/16", "1/2", "5/8"]

    def __init__(self, parent=None):
        super().__init__(parent)

        self._moment_provider: Optional[Callable[[float], float]] = None
        self._shear_provider: Optional[Callable[[float], float]] = None
        self._deflection_context: Optional[Dict[str, Any]] = None
        self.n_beams = 2
        self._base_geometry_includes_bastidor_lateral = False
        self._base_geometry_includes_piso = False
        self._pending_recompute_after_edit = False

        self.mat_db: Optional[MaterialDB] = None
        self._load_material_db()

        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(10)

        form = QFormLayout()
        form.setRowWrapPolicy(QFormLayout.WrapAllRows)
        form.setFieldGrowthPolicy(QFormLayout.ExpandingFieldsGrow)

        self.cmb_t_top = QComboBox()
        self.cmb_t_bot = QComboBox()
        self._populate_thickness_combo(self.cmb_t_top, self.TFLANGE_OPTIONS, default="1/2")
        self._populate_thickness_combo(self.cmb_t_bot, self.TFLANGE_OPTIONS, default="1/2")
        _configure_combo_for_contents(self.cmb_t_top)
        _configure_combo_for_contents(self.cmb_t_bot)

        self.lbl_bf = QLabel("5.0 in - 127 mm")
        self.lbl_info = QLabel("Cálculo: 2 vigas idénticas (I y Wcrit se duplican).")
        self.lbl_info.setWordWrap(True)

        self.cmb_mat_top = QComboBox()
        self.cmb_mat_bot = QComboBox()
        self.cmb_mat_web = QComboBox()
        self.cmb_mat_piso = QComboBox()
        _configure_combo_for_contents(self.cmb_mat_top)
        _configure_combo_for_contents(self.cmb_mat_bot)
        _configure_combo_for_contents(self.cmb_mat_web)
        _configure_combo_for_contents(self.cmb_mat_piso)

        self.n_min = FlexibleDoubleSpinBox()
        self.n_min.setRange(1.0, 100.0)
        self.n_min.setDecimals(2)
        self.n_min.setValue(2.9)
        self.n_min.setSingleStep(0.1)

        self.chk_bastidor_lateral = QCheckBox("Agregar bastidor lateral")
        self.chk_bastidor_lateral_structural = QCheckBox("Bastidor lateral estructural")
        self.chk_bastidor_lateral_structural.setChecked(True)
        self.chk_bastidor_lateral_structural.setVisible(False)
        self.chk_bastidor_lateral_structural.toggled.connect(
            lambda checked: self.chk_bastidor_lateral_structural.setChecked(True) if not checked else None
        )
        self.n_bastidor_lateral_altura = FlexibleDoubleSpinBox()
        self.n_bastidor_lateral_altura.setRange(
            BASTIDOR_LATERAL_ALTURA_MIN_MM,
            BASTIDOR_LATERAL_ALTURA_MAX_MM,
        )
        self.n_bastidor_lateral_altura.setDecimals(1)
        self.n_bastidor_lateral_altura.setSingleStep(1.0)
        self.n_bastidor_lateral_altura.setValue(BASTIDOR_LATERAL_ALTURA_DEFAULT_MM)
        self.n_bastidor_lateral_altura.setSuffix(" mm")

        self.chk_piso = QCheckBox("Agregar piso")
        self.n_piso_width = FlexibleDoubleSpinBox()
        self.n_piso_width.setRange(0.01, 100000.0)
        self.n_piso_width.setDecimals(1)
        self.n_piso_width.setSingleStep(10.0)
        self.n_piso_width.setValue(PISO_ANCHO_MM)
        self.n_piso_width.setSuffix(" mm")
        self.cmb_espesor_piso = QComboBox()
        self._populate_piso_thickness_combo(default=3.0)
        _configure_combo_for_contents(self.cmb_espesor_piso)
        self.chk_chapon = QCheckBox("Agregar chapón")
        self.n_chapon_length = FlexibleDoubleSpinBox()
        self.n_chapon_length.setRange(0.0, 100000.0)
        self.n_chapon_length.setDecimals(1)
        self.n_chapon_length.setSingleStep(100.0)
        self.n_chapon_length.setValue(CHAPON_LENGTH_DEFAULT_MM)
        self.n_chapon_length.setSuffix(" mm")
        self.n_chapon_length.setToolTip("Medido desde el inicio del largo carrozable (x = 0).")
        self.cmb_espesor_chapon = QComboBox()
        self._populate_chapon_thickness_combo(default=6.35)
        _configure_combo_for_contents(self.cmb_espesor_chapon)

        self._populate_material_combos()

        form.addRow("Material planchuela superior:", self.cmb_mat_top)
        form.addRow("Material planchuela inferior:", self.cmb_mat_bot)
        form.addRow("Material alma:", self.cmb_mat_web)
        form.addRow("Material piso:", self.cmb_mat_piso)
        form.addRow("Espesor planchuela superior - 5.0 in - 127 mm:", self.cmb_t_top)
        form.addRow("Espesor planchuela inferior - 5.0 in - 127 mm:", self.cmb_t_bot)
        form.addRow("FS mínimo:", self.n_min)
        form.addRow(self.chk_bastidor_lateral)
        form.addRow("Altura bastidor lateral:", self.n_bastidor_lateral_altura)
        form.addRow(self.chk_piso)
        form.addRow("Ancho piso [mm]:", self.n_piso_width)
        form.addRow("Espesor piso:", self.cmb_espesor_piso)
        form.addRow(self.chk_chapon)
        form.addRow("Largo chapón [mm]:", self.n_chapon_length)
        form.addRow("Espesor chapón SAE 1010:", self.cmb_espesor_chapon)

        lay.addLayout(form)
        lay.addWidget(self.lbl_info)

        # Preview
        self.fig = plt.Figure(figsize=(3.8, 2.5))
        self.ax = self.fig.add_subplot(111)
        self.canvas = FigureCanvas(self.fig)
        self.fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
        self._preview_home_xlim: Optional[tuple[float, float]] = None
        self._preview_home_ylim: Optional[tuple[float, float]] = None
        self._preview_user_view_dirty = False
        self._preview_current_xlim: Optional[tuple[float, float]] = None
        self._preview_current_ylim: Optional[tuple[float, float]] = None
        self._preview_pan_start: Optional[dict[str, tuple[float, float]]] = None
        self._connect_preview_interaction()
        lay.addWidget(self.canvas)

        lay.addWidget(QLabel("Secciones: cargar x, h_viga y Espesor. M se autocompleta desde M(x)."))

        # Tabla
        self.tbl = QTableWidget(8, 14)
        self.tbl.setHorizontalHeaderLabels([
            "Sección", "x [mm]", "h_viga [mm]", "Espesor", "Doble alma", "d alma [mm]", "FS", "M [kg·cm]",
            "Jx [cm^4]", "ȳ [cm]", "c_max [cm]", "Wcrit [cm^3]", "Wreq [cm^3]", "σmax [kg/cm²]"
        ])
        self._ensure_table_edit_policy()
        self.tbl.horizontalHeader().setStretchLastSection(True)
        apply_table_readability_style(self.tbl)

        self._tweb_widgets: List[QComboBox] = []
        self._double_web_widgets: List[QCheckBox] = []
        for r in range(self.tbl.rowCount()):
            _set_item(self.tbl, r, self.COL_SEC, str(r + 1))
            _set_item(self.tbl, r, self.COL_X, "")
            _set_item(self.tbl, r, self.COL_HWEB, "")
            _set_item(self.tbl, r, self.COL_DOUBLE_WEB, "")
            _set_item(self.tbl, r, self.COL_DOUBLE_WEB_OFFSET, "")
            _set_item(self.tbl, r, self.COL_FS, "")
            _set_item(self.tbl, r, self.COL_M, "")
            for c in [self.COL_JX, self.COL_YBAR, self.COL_CMAX, self.COL_WCRIT, self.COL_WREQ, self.COL_SIGMAX]:
                _set_item(self.tbl, r, c, "")

            _set_item_editable(self.tbl, r, self.COL_SEC, False)
            _set_item_editable(self.tbl, r, self.COL_X, True)
            _set_item_editable(self.tbl, r, self.COL_HWEB, True)
            _set_item_editable(self.tbl, r, self.COL_DOUBLE_WEB, False)
            _set_item_editable(self.tbl, r, self.COL_DOUBLE_WEB_OFFSET, False)
            _set_item_editable(self.tbl, r, self.COL_FS, False)
            _set_item_editable(self.tbl, r, self.COL_M, False)
            for c in [self.COL_JX, self.COL_YBAR, self.COL_CMAX, self.COL_WCRIT, self.COL_WREQ, self.COL_SIGMAX]:
                _set_item_editable(self.tbl, r, c, False)

            cmb = QComboBox()
            self._populate_thickness_combo(cmb, self.TWEB_OPTIONS, default="1/4")
            _configure_table_combo_for_cell(cmb)
            cmb.setStyleSheet(combo_cell_style(TABLE_INPUT_BG))
            cmb.currentTextChanged.connect(lambda _t, rr=r: self._schedule_recompute())
            cmb.currentTextChanged.connect(lambda _t: self._emit_inertia_inputs_changed())
            cmb.currentTextChanged.connect(lambda _t, rr=r: self._repaint_preview_from_selection() if rr == self.tbl.currentRow() else None)
            self.tbl.setCellWidget(r, self.COL_TWEB, cmb)
            self._tweb_widgets.append(cmb)

            chk_double_web = QCheckBox()
            chk_double_web.toggled.connect(lambda _checked, rr=r: self._on_double_web_row_changed(rr))
            self.tbl.setCellWidget(r, self.COL_DOUBLE_WEB, chk_double_web)
            self._double_web_widgets.append(chk_double_web)
            self._sync_double_web_row_controls(r)

        # Delegates numéricos
        x_delegate = NullableFloatDelegate(self, decimals=2, minv=-1e12, maxv=1e12)
        hweb_delegate = SpinBoxDelegate(self, minv=0.0, maxv=5000.0, decimals=1, step=10.0)
        double_web_offset_delegate = NullableFloatDelegate(self, decimals=2, minv=0.0, maxv=1e12)
        for delegate in (x_delegate, hweb_delegate, double_web_offset_delegate):
            delegate.closeEditor.connect(self._on_input_editor_closed)
        self.tbl.setItemDelegateForColumn(self.COL_X, x_delegate)
        self.tbl.setItemDelegateForColumn(self.COL_HWEB, hweb_delegate)
        self.tbl.setItemDelegateForColumn(self.COL_DOUBLE_WEB_OFFSET, double_web_offset_delegate)

        lay.addWidget(self.tbl)

        btns = QHBoxLayout()
        self.btn_export = QPushButton("Exportar tabla (.jpg)")
        btns.addWidget(self.btn_export)
        btns.addStretch(1)
        lay.addLayout(btns)

        # Timer
        self._timer = QTimer(self)
        self._timer.setSingleShot(True)
        self._timer.timeout.connect(self._recompute_all)

        # Señales
        self.cmb_t_top.currentTextChanged.connect(self._on_global_changed)
        self.cmb_t_bot.currentTextChanged.connect(self._on_global_changed)
        self.cmb_mat_top.currentTextChanged.connect(self._on_global_changed)
        self.cmb_mat_bot.currentTextChanged.connect(self._on_global_changed)
        self.cmb_mat_web.currentTextChanged.connect(self._on_global_changed)
        self.cmb_mat_piso.currentTextChanged.connect(self._on_global_changed)
        self.n_min.valueChanged.connect(self._schedule_recompute)
        self.chk_bastidor_lateral.toggled.connect(self._on_bastidor_lateral_changed)
        self.n_bastidor_lateral_altura.valueChanged.connect(self._on_bastidor_lateral_changed)
        self.chk_piso.toggled.connect(self._on_piso_changed)
        self.n_piso_width.valueChanged.connect(self._on_piso_changed)
        self.cmb_espesor_piso.currentTextChanged.connect(self._on_piso_changed)
        self.chk_chapon.toggled.connect(self._on_chapon_changed)
        self.n_chapon_length.valueChanged.connect(self._on_chapon_changed)
        self.cmb_espesor_chapon.currentTextChanged.connect(self._on_chapon_changed)

        self.tbl.itemSelectionChanged.connect(self._repaint_preview_from_selection)
        self.tbl.cellChanged.connect(lambda *_: self._schedule_recompute())
        self.tbl.itemChanged.connect(self._on_table_item_changed)

        self.btn_export.clicked.connect(self._export_table_jpg)

        self._update_bastidor_lateral_controls()
        self._update_piso_controls()
        self._update_chapon_controls()
        self._update_sigma_labels()
        self._repaint_preview(h_web_override_mm=200.0, t_web_in=0.25)
        self._recompute_all()

    def _emit_inertia_inputs_changed(self):
        self.inertia_inputs_changed.emit()

    def _ensure_table_edit_policy(self) -> None:
        self.tbl.setEditTriggers(
            QAbstractItemView.DoubleClicked
            | QAbstractItemView.EditKeyPressed
            | QAbstractItemView.AnyKeyPressed
            | QAbstractItemView.SelectedClicked
        )

    def _is_input_column(self, col: int) -> bool:
        return col in self.INPUT_ITEM_COLUMNS

    def _is_result_column(self, col: int) -> bool:
        return col in self.RESULT_COLUMNS

    def _is_user_editing_input_cell(self) -> bool:
        return (
            self.tbl.state() == QAbstractItemView.EditingState
            and self._is_input_column(self.tbl.currentColumn())
        )

    def _on_input_editor_closed(self, *_args) -> None:
        if self._pending_recompute_after_edit:
            QTimer.singleShot(0, self._flush_pending_recompute_after_edit)

    def _flush_pending_recompute_after_edit(self) -> None:
        if not self._pending_recompute_after_edit:
            return
        if self._is_user_editing_input_cell():
            QTimer.singleShot(50, self._flush_pending_recompute_after_edit)
            return
        self._pending_recompute_after_edit = False
        self.refresh_results_from_context()

    def _apply_row_editable_flags(self, row: int) -> None:
        was_blocked = self.tbl.blockSignals(True)
        try:
            _set_item_editable(self.tbl, row, self.COL_SEC, False)
            _set_item_editable(self.tbl, row, self.COL_X, True)
            _set_item_editable(self.tbl, row, self.COL_HWEB, True)
            _set_item_editable(self.tbl, row, self.COL_DOUBLE_WEB, False)
            _set_item_editable(self.tbl, row, self.COL_DOUBLE_WEB_OFFSET, self._row_double_web_enabled(row))
            _set_item_editable(self.tbl, row, self.COL_FS, False)
            _set_item_editable(self.tbl, row, self.COL_M, False)
            for c in [self.COL_JX, self.COL_YBAR, self.COL_CMAX, self.COL_WCRIT, self.COL_WREQ, self.COL_SIGMAX]:
                _set_item_editable(self.tbl, row, c, False)
        finally:
            self.tbl.blockSignals(was_blocked)

    # -------- Materials ----------
    def _load_material_db(self):
        try:
            self.mat_db = MaterialDB.from_txt(default_materials_path())
        except Exception as e:
            self.mat_db = None
            print(f"[SectionCheckPanel] No se pudo cargar MaterialDB: {e}")

    def _populate_thickness_combo(self, cmb: QComboBox, options: List[str], *, default: str):
        cmb.blockSignals(True)
        cmb.clear()
        for opt in options:
            cmb.addItem(_fmt_in_mm(opt), opt)
        idx = cmb.findData(default)
        if idx >= 0:
            cmb.setCurrentIndex(idx)
        cmb.blockSignals(False)

    def _populate_piso_thickness_combo(self, *, default: float):
        self.cmb_espesor_piso.blockSignals(True)
        self.cmb_espesor_piso.clear()
        for label, value_mm in PISO_ESPESOR_OPTIONS_MM:
            self.cmb_espesor_piso.addItem(label, float(value_mm))
        idx = self.cmb_espesor_piso.findData(float(default))
        if idx >= 0:
            self.cmb_espesor_piso.setCurrentIndex(idx)
        self.cmb_espesor_piso.blockSignals(False)

    def _populate_chapon_thickness_combo(self, *, default: float):
        self.cmb_espesor_chapon.blockSignals(True)
        self.cmb_espesor_chapon.clear()
        for label, value_mm in CHAPON_ESPESOR_OPTIONS_MM:
            self.cmb_espesor_chapon.addItem(label, float(value_mm))
        idx = self.cmb_espesor_chapon.findData(float(default))
        if idx >= 0:
            self.cmb_espesor_chapon.setCurrentIndex(idx)
        self.cmb_espesor_chapon.blockSignals(False)

    def _current_material_id(self, cmb: QComboBox) -> str:
        data = cmb.currentData()
        return "" if data is None else str(data)

    def _current_piso_thickness_mm(self) -> float:
        data = self.cmb_espesor_piso.currentData()
        if data is None:
            return 0.0
        return float(data)

    def _current_piso_width_mm(self) -> float:
        return float(self.n_piso_width.value())

    def _current_chapon_thickness_mm(self) -> float:
        data = self.cmb_espesor_chapon.currentData()
        if data is None:
            return 0.0
        return float(data)

    def _current_thickness_in(self, cmb: QComboBox) -> str:
        data = cmb.currentData()
        if data is None:
            return (cmb.currentText() or "").strip()
        return str(data)

    def _current_tweb_in(self, row: int) -> str:
        if 0 <= row < len(self._tweb_widgets):
            return self._current_thickness_in(self._tweb_widgets[row])
        return _get_text(self.tbl, row, self.COL_TWEB)

    def _row_double_web_enabled(self, row: int) -> bool:
        if 0 <= row < len(self._double_web_widgets):
            return bool(self._double_web_widgets[row].isChecked())
        return False

    def _row_double_web_offset_mm(self, row: int) -> Optional[float]:
        if not self._row_double_web_enabled(row):
            return None
        return _try_float(_get_text(self.tbl, row, self.COL_DOUBLE_WEB_OFFSET))

    def _double_web_payload_for_row(self, row: int) -> dict[str, Any]:
        enabled = self._row_double_web_enabled(row)
        offset = self._row_double_web_offset_mm(row) if enabled else None
        return {
            "double_web_enabled": enabled,
            "double_web_inner_face_offset_mm": offset,
            "double_web_clear_gap_mm": None if offset is None else 2.0 * float(offset),
            "web_configuration_label": "Doble" if enabled else "Simple",
        }

    def _sync_double_web_row_controls(self, row: int) -> None:
        self._apply_row_editable_flags(row)

    def _on_double_web_row_changed(self, row: int) -> None:
        self._sync_double_web_row_controls(row)
        if row == self.tbl.currentRow():
            self._repaint_preview_from_selection()
        self._schedule_recompute()
        self._emit_inertia_inputs_changed()

    def _populate_material_combos(self):
        for cmb in (self.cmb_mat_top, self.cmb_mat_bot, self.cmb_mat_web, self.cmb_mat_piso):
            cmb.blockSignals(True)
            cmb.clear()
            cmb.blockSignals(False)

        if self.mat_db is None:
            for cmb in (self.cmb_mat_top, self.cmb_mat_bot, self.cmb_mat_web, self.cmb_mat_piso):
                cmb.addItem("(sin materiales)", None)
            return

        ids = self.mat_db.ids()
        for mat_id in ids:
            mat = self.mat_db.get(mat_id)
            sigma = "-" if mat is None else _fmt2(mat.sigma_adm_kgcm2)
            label = f"{mat_id} - {sigma}"
            self.cmb_mat_top.addItem(label, mat_id)
            self.cmb_mat_bot.addItem(label, mat_id)
            self.cmb_mat_web.addItem(label, mat_id)
            self.cmb_mat_piso.addItem(label, mat_id)

        if self.mat_db.get("F36") is not None:
            idx_top = self.cmb_mat_top.findData("F36")
            idx_bot = self.cmb_mat_bot.findData("F36")
            idx_piso = self.cmb_mat_piso.findData("F36")
            if idx_top >= 0:
                self.cmb_mat_top.setCurrentIndex(idx_top)
            if idx_bot >= 0:
                self.cmb_mat_bot.setCurrentIndex(idx_bot)
            if idx_piso >= 0:
                self.cmb_mat_piso.setCurrentIndex(idx_piso)
        if self.mat_db.get("F24") is not None:
            idx_web = self.cmb_mat_web.findData("F24")
            if idx_web >= 0:
                self.cmb_mat_web.setCurrentIndex(idx_web)

    def _mat_sigma(self, mat_id: str) -> Optional[float]:
        if self.mat_db is None:
            return None
        m = self.mat_db.get((mat_id or "").strip())
        if m is None:
            return None
        return float(m.sigma_adm_kgcm2)

    def _update_sigma_labels(self):
        sigma_top = self._mat_sigma(self._current_material_id(self.cmb_mat_top))
        sigma_bot = self._mat_sigma(self._current_material_id(self.cmb_mat_bot))
        sigma_web = self._mat_sigma(self._current_material_id(self.cmb_mat_web))
        sigma_piso = self._mat_sigma(self._current_material_id(self.cmb_mat_piso))
        self.cmb_mat_top.setToolTip("-" if sigma_top is None else f"σ_adm = {_fmt2(sigma_top)} kg/cm²")
        self.cmb_mat_bot.setToolTip("-" if sigma_bot is None else f"σ_adm = {_fmt2(sigma_bot)} kg/cm²")
        self.cmb_mat_web.setToolTip("-" if sigma_web is None else f"σ_adm = {_fmt2(sigma_web)} kg/cm²")
        self.cmb_mat_piso.setToolTip("-" if sigma_piso is None else f"σ_adm = {_fmt2(sigma_piso)} kg/cm²")

    # -------- API ----------
    def set_moment_provider(self, fn: Optional[Callable[[float], float]]):
        self._moment_provider = fn
        for r in range(self.tbl.rowCount()):
            self._auto_fill_M_for_row(r)
        self._schedule_recompute()

    def set_shear_provider(self, fn: Optional[Callable[[float], float]]):
        self._shear_provider = fn

    def refresh_results_from_context(self) -> None:
        if self._is_user_editing_input_cell():
            self._pending_recompute_after_edit = True
            return
        self._ensure_table_edit_policy()
        for r in range(self.tbl.rowCount()):
            self._auto_fill_M_for_row(r)
        self._recompute_all()

    def set_deflection_context(self, result: Optional[Any], *, i_source: str = ""):
        if result is None:
            self._deflection_context = None
            return
        self._deflection_context = {
            "vmin_mm": float(result.vmin_mm),
            "x_vmin_mm": float(result.x_vmin_mm),
            "utilized_mm": float(result.utilized_mm),
            "allowable_mm": float(result.allowable_mm),
            "limit_y_mm": float(result.limit_y_mm),
            "camber_mid_mm": float(result.camber_mid_mm),
            "ok": bool(result.ok),
            "i_source": str(i_source or ""),
        }

    def set_beam_context(
        self,
        *,
        largo_viga_mm: Optional[float] = None,
        posicion_perno_mm: Optional[float] = None,
    ) -> None:
        return

    def clear_results_only(self, *, clear_moments_if_no_provider: bool = False):
        self._ensure_table_edit_policy()
        self.tbl.blockSignals(True)
        for r in range(self.tbl.rowCount()):
            for c in [self.COL_FS, self.COL_JX, self.COL_YBAR, self.COL_CMAX, self.COL_WCRIT, self.COL_WREQ, self.COL_SIGMAX]:
                _set_item(self.tbl, r, c, "")
                _set_item_editable(self.tbl, r, c, False)
            if clear_moments_if_no_provider and self._moment_provider is None:
                _set_item(self.tbl, r, self.COL_M, "")
                _set_item_editable(self.tbl, r, self.COL_M, False)
            self._set_row_color(r, ok=None, paint_widgets=False)
        self.tbl.blockSignals(False)

    # -------- Bastidor lateral ----------
    def _update_bastidor_lateral_controls(self) -> None:
        enabled = bool(self.chk_bastidor_lateral.isChecked())
        self.chk_bastidor_lateral_structural.setChecked(True)
        self.chk_bastidor_lateral_structural.setEnabled(False)
        self.n_bastidor_lateral_altura.setEnabled(enabled)

    def _include_bastidor_lateral_in_geometry(self) -> bool:
        return (
            bool(self.chk_bastidor_lateral.isChecked())
            and not bool(self._base_geometry_includes_bastidor_lateral)
        )

    def _on_bastidor_lateral_changed(self, *_args) -> None:
        self._update_bastidor_lateral_controls()
        self._repaint_preview_from_selection()
        self._schedule_recompute()
        self._emit_inertia_inputs_changed()

    # -------- Piso ----------
    def _update_piso_controls(self) -> None:
        enabled = bool(self.chk_piso.isChecked())
        self.n_piso_width.setEnabled(enabled)
        self.cmb_espesor_piso.setEnabled(enabled)
        self.cmb_mat_piso.setEnabled(enabled)

    def _include_piso_in_geometry(self) -> bool:
        return (
            bool(self.chk_piso.isChecked())
            and not bool(self._base_geometry_includes_piso)
        )

    def _on_piso_changed(self, *_args) -> None:
        self._update_piso_controls()
        self._repaint_preview_from_selection()
        self._schedule_recompute()
        self._emit_inertia_inputs_changed()

    # -------- Chapón inferior ----------
    def _update_chapon_controls(self) -> None:
        enabled = bool(self.chk_chapon.isChecked())
        self.n_chapon_length.setEnabled(enabled)
        self.cmb_espesor_chapon.setEnabled(enabled)

    def _chapon_length_mm(self) -> Optional[float]:
        try:
            value = float(self.n_chapon_length.value())
        except Exception:
            return None
        if not math.isfinite(value) or value <= 0.0:
            return None
        return value

    def _chapon_end_mm(self) -> Optional[float]:
        return self._chapon_length_mm()

    def _chapon_context_missing_fields(self) -> list[str]:
        missing: list[str] = []
        if self._chapon_length_mm() is None:
            missing.append("chapon_length_mm")
        return missing

    def _chapon_context_missing(self) -> bool:
        return bool(self.chk_chapon.isChecked()) and bool(self._chapon_context_missing_fields())

    def _include_chapon_in_geometry(self, station_mm: Optional[float]) -> bool:
        if not bool(self.chk_chapon.isChecked()):
            return False
        if station_mm is None:
            return False
        chapon_end = self._chapon_end_mm()
        if chapon_end is None:
            return False
        return 0.0 <= float(station_mm) <= float(chapon_end)

    def _on_chapon_changed(self, *_args) -> None:
        self._update_chapon_controls()
        self._repaint_preview_from_selection()
        self._schedule_recompute()
        self._emit_inertia_inputs_changed()

    def _top_sigma_for_section(
        self,
        sec: ISection | CompositeSection,
        sigma_top_kgcm2: Optional[float],
        sigma_piso_kgcm2: Optional[float],
    ) -> Optional[float]:
        if isinstance(sec, CompositeSection) and sec.includes_piso:
            return sigma_piso_kgcm2
        return sigma_top_kgcm2

    def _bottom_sigma_for_section(
        self,
        sec: ISection | CompositeSection,
        sigma_bot_kgcm2: Optional[float],
        sigma_chapon_kgcm2: Optional[float],
    ) -> Optional[float]:
        if isinstance(sec, CompositeSection) and sec.includes_chapon:
            return sigma_chapon_kgcm2
        return sigma_bot_kgcm2

    def _material_sigma_by_id(self, mat_id: str) -> Optional[float]:
        return self._mat_sigma(str(mat_id or "").strip())

    def _component_material_for_rect(self, rect_label: str) -> tuple[str, str]:
        label = str(rect_label or "")
        if label.startswith("bastidor"):
            return "Bastidor lateral", CHAPON_MATERIAL_ID
        if label.startswith("piso"):
            return "Piso", self._current_material_id(self.cmb_mat_piso)
        if label.startswith("chapon"):
            return "Chapón", CHAPON_MATERIAL_ID
        if label.endswith("ala_sup"):
            return "Viga principal - ala superior", self._current_material_id(self.cmb_mat_top)
        if label.endswith("ala_inf"):
            return "Viga principal - ala inferior", self._current_material_id(self.cmb_mat_bot)
        if label.endswith("alma") or "_alma" in label:
            return "Viga principal - alma", self._current_material_id(self.cmb_mat_web)
        return "Viga principal", self._current_material_id(self.cmb_mat_web)

    def _base_component_rects(self, sec: ISection) -> list[SectionRect]:
        b = float(sec.b_f_mm)
        tw = float(sec.t_web_mm)
        return [
            SectionRect(0.0, 0.0, b, sec.t_bot_mm, "base_ala_inf"),
            SectionRect(b / 2.0 - tw / 2.0, sec.t_bot_mm, tw, sec.h_web_mm, "base_alma"),
            SectionRect(0.0, sec.t_bot_mm + sec.h_web_mm, b, sec.t_top_mm, "base_ala_sup"),
        ]

    def _component_flex_checks(
        self,
        sec: ISection | CompositeSection,
        M_kgcm: Optional[float],
    ) -> list[dict[str, Any]]:
        if M_kgcm is None:
            return []
        no_flex_demand = _is_no_flex_demand(M_kgcm)

        props = sec.props_mm()
        ybar_global_mm = float(props.get("ybar_global_mm", props.get("ybar_mm", 0.0)))
        y_min_mm = float(props.get("y_min_mm", 0.0))
        ix_total_cm4 = (
            float(props["Ix_mm4"])
            * float(self._calculation_n_beams(sec))
            / (10.0 ** 4)
        )
        if ix_total_cm4 <= 0.0:
            return []

        rects = list(sec.rects) if isinstance(sec, CompositeSection) else self._base_component_rects(sec)
        groups: dict[tuple[str, str], dict[str, Any]] = {}
        for rect in rects:
            component, material_id = self._component_material_for_rect(rect.label)
            key = (component, material_id)
            item = groups.setdefault(
                key,
                {
                    "component": component,
                    "material": material_id,
                    "y_inf_mm": float(rect.y0_mm),
                    "y_sup_mm": float(rect.y1_mm),
                },
            )
            item["y_inf_mm"] = min(float(item["y_inf_mm"]), float(rect.y0_mm))
            item["y_sup_mm"] = max(float(item["y_sup_mm"]), float(rect.y1_mm))

        checks: list[dict[str, Any]] = []
        for item in groups.values():
            y_inf = float(item["y_inf_mm"])
            y_sup = float(item["y_sup_mm"])
            c_mm = max(abs(y_inf - ybar_global_mm), abs(y_sup - ybar_global_mm))
            c_cm = c_mm / 10.0
            sigma_calc = 0.0 if no_flex_demand else abs(float(M_kgcm)) * c_cm / max(ix_total_cm4, 1e-12)
            sigma_adm = self._material_sigma_by_id(str(item["material"]))
            fs = math.inf if no_flex_demand else (None if sigma_adm is None else float(sigma_adm) / max(sigma_calc, 1e-12))
            checks.append(
                {
                    "component": item["component"],
                    "material": item["material"],
                    "y_inf_cm": (y_inf - y_min_mm) / 10.0,
                    "y_sup_cm": (y_sup - y_min_mm) / 10.0,
                    "cmax_cm": c_cm,
                    "sigma_calc_kgcm2": sigma_calc,
                    "sigma_adm_kgcm2": sigma_adm,
                    "fs": fs,
                    "wreq_cm3": 0.0 if no_flex_demand else (None if sigma_adm is None else abs(float(M_kgcm)) / max(float(sigma_adm), 1e-12)),
                    "no_flex_demand": no_flex_demand,
                }
            )

        def _sort_key(row: dict[str, Any]) -> tuple[float, str]:
            fs = row.get("fs")
            return (float(fs) if fs is not None else float("inf"), str(row.get("component", "")))

        return sorted(checks, key=_sort_key)

    def _governing_component_check(self, checks: list[dict[str, Any]]) -> Optional[dict[str, Any]]:
        valid = [row for row in checks if row.get("fs") is not None and not row.get("no_flex_demand")]
        if not valid:
            return None
        return min(valid, key=lambda row: float(row["fs"]))

    def _missing_material_components(self, checks: list[dict[str, Any]]) -> list[dict[str, str]]:
        missing: list[dict[str, str]] = []
        for row in checks:
            if row.get("no_flex_demand"):
                continue
            if row.get("sigma_adm_kgcm2") is not None:
                continue
            missing.append(
                {
                    "component": str(row.get("component", "")),
                    "material": str(row.get("material", "")),
                }
            )
        return missing

    def _compute_section_row_result(
        self,
        *,
        hweb: float,
        tweb_in: float,
        x_value: Optional[float],
        m_value: Optional[float],
        round_up_decimals: int,
        row_index: Optional[int] = None,
    ) -> dict[str, Any]:
        double_web_enabled = self._row_double_web_enabled(row_index if row_index is not None else -1)
        double_web_offset = self._row_double_web_offset_mm(row_index if row_index is not None else -1)
        sec = self._make_section(
            hweb,
            tweb_in,
            station_mm=x_value,
            double_web_enabled=double_web_enabled,
            double_web_inner_face_offset_mm=double_web_offset,
        )
        props = sec.props_mm()
        chapon_context_error = self._chapon_context_missing()
        chapon_context_missing_fields = self._chapon_context_missing_fields() if chapon_context_error else []
        no_flex_demand = _is_no_flex_demand(m_value)

        sigma_base_top_adm = self._mat_sigma(self._current_material_id(self.cmb_mat_top))
        sigma_base_bot_adm = self._mat_sigma(self._current_material_id(self.cmb_mat_bot))
        sigma_piso_adm = self._mat_sigma(self._current_material_id(self.cmb_mat_piso))
        sigma_chapon_adm = self._mat_sigma(CHAPON_MATERIAL_ID)
        sigma_top_calc_adm = self._top_sigma_for_section(sec, sigma_base_top_adm, sigma_piso_adm)
        sigma_bot_calc_adm = self._bottom_sigma_for_section(sec, sigma_base_bot_adm, sigma_chapon_adm)

        res = None
        sigma_top_for_flex = sigma_top_calc_adm
        sigma_bot_for_flex = sigma_bot_calc_adm
        if no_flex_demand:
            sigma_top_for_flex = 1.0 if sigma_top_for_flex is None else sigma_top_for_flex
            sigma_bot_for_flex = 1.0 if sigma_bot_for_flex is None else sigma_bot_for_flex
        if m_value is not None and sigma_top_for_flex is not None and sigma_bot_for_flex is not None:
            try:
                res = compute_flex_row(
                    section=sec,
                    M_kgcm=m_value,
                    sigma_adm_kgcm2=float(min(sigma_top_for_flex, sigma_bot_for_flex)),
                    sigma_adm_top_kgcm2=float(sigma_top_for_flex),
                    sigma_adm_bot_kgcm2=float(sigma_bot_for_flex),
                    n_beams=self._calculation_n_beams(sec),
                    round_up_decimals=round_up_decimals,
                )
            except Exception:
                res = None

        calc_n_beams = self._calculation_n_beams(sec)
        ix_single_cm4 = float(props["Ix_mm4"]) / (10.0 ** 4)
        ix_total_cm4 = ix_single_cm4 * float(calc_n_beams)
        h_total_mm = float(props["H_mm"])
        ybar_cm = float(props["ybar_mm"]) / 10.0
        c_top_cm = float(props["c_top_mm"]) / 10.0
        c_bot_cm = float(props["c_bot_mm"]) / 10.0
        cmax_cm = float(props["c_max_mm"]) / 10.0
        wcrit_cm3 = ix_total_cm4 / max(cmax_cm, 1e-12)
        sigma_top = (abs(m_value) * c_top_cm / max(ix_total_cm4, 1e-12)) if m_value is not None else None
        sigma_bot = (abs(m_value) * c_bot_cm / max(ix_total_cm4, 1e-12)) if m_value is not None else None
        wreq_top = (abs(m_value) / max(float(sigma_top_calc_adm), 1e-12)) if (m_value is not None and sigma_top_calc_adm is not None) else None
        wreq_bot = (abs(m_value) / max(float(sigma_bot_calc_adm), 1e-12)) if (m_value is not None and sigma_bot_calc_adm is not None) else None
        fs_top = (float(sigma_top_calc_adm) / max(sigma_top, 1e-12)) if (sigma_top_calc_adm is not None and sigma_top is not None) else None
        fs_bot = (float(sigma_bot_calc_adm) / max(sigma_bot, 1e-12)) if (sigma_bot_calc_adm is not None and sigma_bot is not None) else None
        if no_flex_demand:
            sigma_top = 0.0
            sigma_bot = 0.0
            wreq_top = 0.0
            wreq_bot = 0.0
            fs_top = math.inf
            fs_bot = math.inf

        component_checks = self._component_flex_checks(sec, m_value)
        missing_materials = self._missing_material_components(component_checks)
        material_error = bool(missing_materials)
        governing_component = self._governing_component_check(component_checks)
        fs_final = None
        sigma_final = None
        wreq_final = None
        if no_flex_demand:
            fs_final = math.inf
            sigma_final = 0.0
            wreq_final = 0.0
            governing_component = None
        elif chapon_context_error:
            res = None
            wreq_top = None
            wreq_bot = None
            fs_top = None
            fs_bot = None
            governing_component = None
        elif material_error:
            res = None
            wreq_top = None
            wreq_bot = None
            fs_top = None
            fs_bot = None
            governing_component = None
        elif governing_component is not None:
            fs_final = float(governing_component["fs"])
            sigma_final = float(governing_component["sigma_calc_kgcm2"])
            wreq_values = [
                float(row["wreq_cm3"])
                for row in component_checks
                if row.get("wreq_cm3") is not None
            ]
            if wreq_values:
                wreq_final = max(wreq_values)
                wreq_top = wreq_final
                wreq_bot = wreq_final
        elif res is not None:
            fs_final = float(res.FS)
            sigma_final = float(res.sigma_max_kgcm2)
            wreq_final = float(res.Wreq_cm3)

        if wreq_final is None and res is not None:
            wreq_final = float(res.Wreq_cm3)
        if sigma_final is None and res is not None:
            sigma_final = float(res.sigma_max_kgcm2)
        if fs_final is None and res is not None:
            fs_final = float(res.FS)

        return {
            "sec": sec,
            "res": res,
            "no_flex_demand": no_flex_demand,
            "chapon_context_error": chapon_context_error,
            "chapon_context_missing_fields": chapon_context_missing_fields,
            "double_web_enabled": double_web_enabled,
            "double_web_inner_face_offset_mm": double_web_offset,
            "double_web_error": "",
            "sigma_top_calc_adm": sigma_top_calc_adm,
            "sigma_bot_calc_adm": sigma_bot_calc_adm,
            "sigma_base_top_adm": sigma_base_top_adm,
            "sigma_base_bot_adm": sigma_base_bot_adm,
            "sigma_piso_adm": sigma_piso_adm,
            "sigma_chapon_adm": sigma_chapon_adm,
            "component_checks": component_checks,
            "missing_materials": missing_materials,
            "material_error": material_error,
            "governing_component": governing_component,
            "fs_final": fs_final,
            "sigma_final": sigma_final,
            "wreq_final": wreq_final,
            "fs_top": fs_top,
            "fs_bot": fs_bot,
            "sigma_top": sigma_top,
            "sigma_bot": sigma_bot,
            "ix_single_cm4": ix_single_cm4,
            "ix_total_cm4": ix_total_cm4,
            "h_total_mm": h_total_mm,
            "ybar_cm": ybar_cm,
            "c_top_cm": c_top_cm,
            "c_bot_cm": c_bot_cm,
            "cmax_cm": cmax_cm,
            "wcrit_cm3": wcrit_cm3,
            "wreq_top": wreq_top,
            "wreq_bot": wreq_bot,
        }

    # -------- Preview ----------
    def _draw_dim_v_on(self, ax, y1: float, y2: float, x_dim: float, x_obj: float, text: str, *, color="blue"):
        ax.plot([x_obj, x_dim], [y1, y1], color=color, linewidth=1.0)
        ax.plot([x_obj, x_dim], [y2, y2], color=color, linewidth=1.0)

        arr = FancyArrowPatch((x_dim, y1), (x_dim, y2),
                              arrowstyle="<->", mutation_scale=12,
                              linewidth=1.0, color=color)
        arr.set_fill(False)
        ax.add_patch(arr)

        ym = 0.5 * (y1 + y2)
        ax.text(
            x_dim,
            ym,
            text,
            ha="center",
            va="center",
            fontsize=10,
            color=color,
            bbox=dict(facecolor="white", edgecolor="none", pad=1.2),
        )

    def _make_base_section(self, h_web_mm: float, t_web_in: float) -> ISection:
        b_f_in = 5.0
        t_top_in = _parse_frac_in(self._current_thickness_in(self.cmb_t_top))
        t_bot_in = _parse_frac_in(self._current_thickness_in(self.cmb_t_bot))
        return ISection(
            b_f_mm=_in_to_mm(b_f_in),
            t_top_mm=_in_to_mm(t_top_in),
            t_bot_mm=_in_to_mm(t_bot_in),
            h_web_mm=float(h_web_mm),
            t_web_mm=_in_to_mm(float(t_web_in)),
        )

    def _double_i_rects(
        self,
        sec: ISection,
        *,
        double_web_enabled: bool = False,
        double_web_inner_face_offset_mm: Optional[float] = None,
    ) -> list[SectionRect]:
        left_x0 = VIGA_LEFT_FACE_EXTERIOR_MM
        right_x0 = VIGA_RIGHT_FACE_INTERIOR_MM
        left_web_center = 0.5 * (VIGA_LEFT_FACE_EXTERIOR_MM + VIGA_LEFT_FACE_INTERIOR_MM)
        right_web_center = 0.5 * (VIGA_RIGHT_FACE_INTERIOR_MM + VIGA_RIGHT_FACE_EXTERIOR_MM)
        left_webs = build_web_rects(
            b_f_mm=sec.b_f_mm,
            t_web_mm=sec.t_web_mm,
            h_web_mm=sec.h_web_mm,
            y0_mm=sec.t_bot_mm,
            double_web_enabled=double_web_enabled,
            double_web_inner_face_offset_mm=double_web_inner_face_offset_mm,
            x_center_mm=left_web_center,
            label_prefix="viga_izq_alma",
        )
        right_webs = build_web_rects(
            b_f_mm=sec.b_f_mm,
            t_web_mm=sec.t_web_mm,
            h_web_mm=sec.h_web_mm,
            y0_mm=sec.t_bot_mm,
            double_web_enabled=double_web_enabled,
            double_web_inner_face_offset_mm=double_web_inner_face_offset_mm,
            x_center_mm=right_web_center,
            label_prefix="viga_der_alma",
        )
        return [
            SectionRect(left_x0, 0.0, sec.b_f_mm, sec.t_bot_mm, "viga_izq_ala_inf"),
            *left_webs,
            SectionRect(left_x0, sec.t_bot_mm + sec.h_web_mm, sec.b_f_mm, sec.t_top_mm, "viga_izq_ala_sup"),
            SectionRect(right_x0, 0.0, sec.b_f_mm, sec.t_bot_mm, "viga_der_ala_inf"),
            *right_webs,
            SectionRect(right_x0, sec.t_bot_mm + sec.h_web_mm, sec.b_f_mm, sec.t_top_mm, "viga_der_ala_sup"),
        ]

    def _bastidor_lateral_rects(self, top_y_mm: float) -> list[SectionRect]:
        h = float(self.n_bastidor_lateral_altura.value())
        t = BASTIDOR_LATERAL_ESPESOR_MM
        ala = BASTIDOR_LATERAL_ALA_MM
        y0 = float(top_y_mm) - h
        x_left = -BASTIDOR_LATERAL_DISTANCIA_MM
        x_right = BASTIDOR_LATERAL_DISTANCIA_MM
        return [
            SectionRect(x_left - t / 2.0, y0, t, h, "bastidor_izq_alma"),
            SectionRect(x_left + t / 2.0, y0, ala, t, "bastidor_izq_ala_inf"),
            SectionRect(x_left + t / 2.0, top_y_mm - t, ala, t, "bastidor_izq_ala_sup"),
            SectionRect(x_right - t / 2.0, y0, t, h, "bastidor_der_alma"),
            SectionRect(x_right - t / 2.0 - ala, y0, ala, t, "bastidor_der_ala_inf"),
            SectionRect(x_right - t / 2.0 - ala, top_y_mm - t, ala, t, "bastidor_der_ala_sup"),
        ]

    def _piso_rects(self, top_y_mm: float) -> list[SectionRect]:
        thickness = self._current_piso_thickness_mm()
        width = self._current_piso_width_mm()
        return [
            SectionRect(
                -width / 2.0,
                float(top_y_mm),
                width,
                thickness,
                "piso",
            )
        ]

    def _chapon_rects(self, bottom_y_mm: float) -> list[SectionRect]:
        thickness = self._current_chapon_thickness_mm()
        return [
            SectionRect(
                -CHAPON_ANCHO_MM / 2.0,
                float(bottom_y_mm) - thickness,
                CHAPON_ANCHO_MM,
                thickness,
                "chapon",
            )
        ]

    def _make_section(
        self,
        h_web_mm: float,
        t_web_in: float,
        *,
        station_mm: Optional[float] = None,
        double_web_enabled: bool = False,
        double_web_inner_face_offset_mm: Optional[float] = None,
    ) -> ISection | CompositeSection:
        base = self._make_base_section(h_web_mm, t_web_in)
        include_bastidor = self._include_bastidor_lateral_in_geometry()
        include_piso = self._include_piso_in_geometry()
        include_chapon = self._include_chapon_in_geometry(station_mm)
        if not include_bastidor and not include_piso and not include_chapon and not double_web_enabled:
            return base

        rects = self._double_i_rects(
            base,
            double_web_enabled=double_web_enabled,
            double_web_inner_face_offset_mm=double_web_inner_face_offset_mm,
        )
        chapon_end = self._chapon_end_mm()
        if include_chapon:
            rects.extend(self._chapon_rects(0.0))
        if include_bastidor:
            rects.extend(self._bastidor_lateral_rects(base.H_mm))
        if include_piso:
            rects.extend(self._piso_rects(base.H_mm))
        return CompositeSection(
            base_section=base,
            rects=tuple(rects),
            includes_bastidor_lateral=include_bastidor,
            bastidor_lateral_height_mm=float(self.n_bastidor_lateral_altura.value()) if include_bastidor else 0.0,
            includes_piso=include_piso,
            piso_thickness_mm=self._current_piso_thickness_mm() if include_piso else 0.0,
            piso_width_mm=self._current_piso_width_mm() if include_piso else 0.0,
            includes_chapon=include_chapon,
            chapon_thickness_mm=self._current_chapon_thickness_mm() if include_chapon else 0.0,
            chapon_width_mm=CHAPON_ANCHO_MM if include_chapon else 0.0,
            chapon_start_mm=0.0 if include_chapon else 0.0,
            chapon_end_mm=float(chapon_end) if include_chapon and chapon_end is not None else 0.0,
        )

    def _calculation_n_beams(self, sec: ISection | CompositeSection) -> int:
        return 1 if isinstance(sec, CompositeSection) else int(self.n_beams)

    def _draw_section_preview_on(self, ax, sec: ISection | CompositeSection):
        p = sec.props_mm()

        ax.clear()
        if isinstance(sec, CompositeSection):
            rects = list(sec.rects)
            for rect in rects:
                if rect.label.startswith("bastidor"):
                    color = "darkgreen"
                elif rect.label.startswith("piso"):
                    color = "darkorange"
                elif rect.label.startswith("chapon"):
                    color = "firebrick"
                else:
                    color = "blue"
                ax.add_patch(
                    plt.Rectangle(
                        (rect.x0_mm, rect.y0_mm),
                        rect.b_mm,
                        rect.h_mm,
                        fill=False,
                        edgecolor=color,
                        linewidth=1.2,
                    )
                )
            x_min = min(r.x0_mm for r in rects)
            x_max = max(r.x0_mm + r.b_mm for r in rects)
            y_min = float(p.get("y_min_mm", min(r.y0_mm for r in rects)))
            y_max = float(p.get("y_max_mm", max(r.y0_mm + r.h_mm for r in rects)))
            H = y_max - y_min
            self._draw_dim_v_on(ax, y_min, y_max, x_min - 0.08 * (x_max - x_min), x_min, _fmt_int(H), color="blue")
            home_xlim = (x_min - 0.16 * (x_max - x_min), x_max + 0.08 * (x_max - x_min))
            home_ylim = (y_min - 0.25 * max(H, 1.0), y_max + 0.25 * max(H, 1.0))
        else:
            H = float(p.get("H_mm", 0.0))
            b = float(sec.b_f_mm)
            t_top = float(sec.t_top_mm)
            t_bot = float(sec.t_bot_mm)
            h = float(sec.h_web_mm)
            tw = float(sec.t_web_mm)

            ax.add_patch(plt.Rectangle((0.0, 0.0), b, t_bot, fill=False, edgecolor="blue", linewidth=1.2))
            ax.add_patch(plt.Rectangle((b / 2 - tw / 2, t_bot), tw, h, fill=False, edgecolor="blue", linewidth=1.2))
            ax.add_patch(plt.Rectangle((0.0, t_bot + h), b, t_top, fill=False, edgecolor="blue", linewidth=1.2))

            x_obj_L = 0.0
            x_dim_H = -0.35 * b
            self._draw_dim_v_on(ax, 0.0, H, x_dim_H, x_obj_L, _fmt_int(H), color="blue")

            x_obj_R = b
            x_dim_h = b + 0.22 * b
            self._draw_dim_v_on(ax, t_bot, t_bot + h, x_dim_h, x_obj_R, _fmt_int(h), color="blue")

            xpad = 0.95 * b
            ypad_top = 0.35 * max(H, 1.0)
            ypad_bot = 0.25 * max(H, 1.0)
            home_xlim = (-xpad, b + xpad)
            home_ylim = (-ypad_bot, max(H, 1.0) + ypad_top)

        if ax is self.ax:
            self._preview_home_xlim = tuple(float(v) for v in home_xlim)
            self._preview_home_ylim = tuple(float(v) for v in home_ylim)
            ax.set_position([0, 0, 1, 1])
            if (
                self._preview_user_view_dirty
                and self._preview_current_xlim is not None
                and self._preview_current_ylim is not None
            ):
                ax.set_xlim(*self._preview_current_xlim)
                ax.set_ylim(*self._preview_current_ylim)
            else:
                ax.set_xlim(*self._preview_home_xlim)
                ax.set_ylim(*self._preview_home_ylim)
                self._preview_current_xlim = self._preview_home_xlim
                self._preview_current_ylim = self._preview_home_ylim
                self._preview_user_view_dirty = False
        else:
            ax.set_xlim(*home_xlim)
            ax.set_ylim(*home_ylim)

        ax.set_aspect("equal", adjustable="box")
        ax.axis("off")

    def _connect_preview_interaction(self) -> None:
        self.canvas.mpl_connect("scroll_event", self._on_preview_scroll)
        self.canvas.mpl_connect("button_press_event", self._on_preview_button_press)
        self.canvas.mpl_connect("button_release_event", self._on_preview_button_release)
        self.canvas.mpl_connect("motion_notify_event", self._on_preview_motion)

    def _store_preview_home_view(self) -> None:
        self._preview_home_xlim = tuple(float(v) for v in self.ax.get_xlim())
        self._preview_home_ylim = tuple(float(v) for v in self.ax.get_ylim())
        self._preview_current_xlim = self._preview_home_xlim
        self._preview_current_ylim = self._preview_home_ylim
        self._preview_user_view_dirty = False
        self._preview_pan_start = None

    def _reset_preview_view(self) -> None:
        if self._preview_home_xlim is None or self._preview_home_ylim is None:
            return
        self.ax.set_xlim(*self._preview_home_xlim)
        self.ax.set_ylim(*self._preview_home_ylim)
        self._preview_current_xlim = tuple(float(v) for v in self.ax.get_xlim())
        self._preview_current_ylim = tuple(float(v) for v in self.ax.get_ylim())
        self._preview_user_view_dirty = False
        self.canvas.draw_idle()

    def _on_preview_scroll(self, event) -> None:
        if event.inaxes is not self.ax or event.xdata is None or event.ydata is None:
            return

        step = float(getattr(event, "step", 0.0) or 0.0)
        if step == 0.0:
            button = getattr(event, "button", None)
            step = 1.0 if button == "up" else -1.0 if button == "down" else 0.0
        if step == 0.0:
            return

        scale = 0.85 if step > 0 else 1.0 / 0.85
        x0, x1 = self.ax.get_xlim()
        y0, y1 = self.ax.get_ylim()
        x_center = float(event.xdata)
        y_center = float(event.ydata)

        new_w = max(abs(x1 - x0) * scale, 1e-9)
        new_h = max(abs(y1 - y0) * scale, 1e-9)
        rel_x = (x_center - x0) / max(x1 - x0, 1e-9)
        rel_y = (y_center - y0) / max(y1 - y0, 1e-9)

        new_xlim = (x_center - rel_x * new_w, x_center + (1.0 - rel_x) * new_w)
        new_ylim = (y_center - rel_y * new_h, y_center + (1.0 - rel_y) * new_h)
        self.ax.set_xlim(*new_xlim)
        self.ax.set_ylim(*new_ylim)
        self._preview_current_xlim = tuple(float(v) for v in self.ax.get_xlim())
        self._preview_current_ylim = tuple(float(v) for v in self.ax.get_ylim())
        self._preview_user_view_dirty = True
        self.canvas.draw_idle()

    def _on_preview_button_press(self, event) -> None:
        if event.inaxes is not self.ax:
            return

        if getattr(event, "dblclick", False):
            self._preview_pan_start = None
            self._reset_preview_view()
            return

        if event.button != MouseButton.MIDDLE:
            return
        if event.x is None or event.y is None:
            return

        self._preview_pan_start = {
            "xy": (float(event.x), float(event.y)),
            "xlim": tuple(float(v) for v in self.ax.get_xlim()),
            "ylim": tuple(float(v) for v in self.ax.get_ylim()),
        }

    def _on_preview_motion(self, event) -> None:
        if self._preview_pan_start is None or event.x is None or event.y is None:
            return

        x_press, y_press = self._preview_pan_start["xy"]
        x_start, y_start = self.ax.transData.inverted().transform((x_press, y_press))
        x_now, y_now = self.ax.transData.inverted().transform((float(event.x), float(event.y)))
        dx = float(x_start - x_now)
        dy = float(y_start - y_now)

        x0, x1 = self._preview_pan_start["xlim"]
        y0, y1 = self._preview_pan_start["ylim"]
        new_xlim = (x0 + dx, x1 + dx)
        new_ylim = (y0 + dy, y1 + dy)
        self.ax.set_xlim(*new_xlim)
        self.ax.set_ylim(*new_ylim)
        self._preview_current_xlim = tuple(float(v) for v in self.ax.get_xlim())
        self._preview_current_ylim = tuple(float(v) for v in self.ax.get_ylim())
        self._preview_user_view_dirty = True
        self.canvas.draw_idle()

    def _on_preview_button_release(self, event) -> None:
        if event.button == MouseButton.MIDDLE:
            self._preview_pan_start = None

    def _repaint_preview(self, *, h_web_override_mm: float, t_web_in: float):
        station_mm = None
        r = self.tbl.currentRow()
        if r >= 0:
            station_mm = _try_float(_get_text(self.tbl, r, self.COL_X))
        try:
            sec = self._make_section(
                h_web_override_mm,
                t_web_in,
                station_mm=station_mm,
                double_web_enabled=self._row_double_web_enabled(r),
                double_web_inner_face_offset_mm=self._row_double_web_offset_mm(r),
            )
        except DoubleWebGeometryError:
            self.ax.clear()
            self.ax.text(0.5, 0.5, "ERR DOBLE ALMA", ha="center", va="center", transform=self.ax.transAxes)
            self.ax.axis("off")
        else:
            self._draw_section_preview_on(self.ax, sec)
        self.canvas.draw_idle()

    def _repaint_preview_from_selection(self):
        r = self.tbl.currentRow()
        if r < 0:
            return
        hweb = _try_float(_get_text(self.tbl, r, self.COL_HWEB))
        if hweb is None or hweb <= 0:
            hweb = 200.0
        tweb_in = _parse_frac_in(self._current_tweb_in(r))
        self._repaint_preview(h_web_override_mm=hweb, t_web_in=tweb_in)

    def _on_global_changed(self, *_):
        self._update_sigma_labels()
        self._repaint_preview_from_selection()
        self._schedule_recompute()
        self._emit_inertia_inputs_changed()

    # -------- Tabla / recompute ----------
    def _on_table_item_changed(self, it: QTableWidgetItem):
        if it is None:
            return
        r, c = it.row(), it.column()
        if c == self.COL_X:
            self._auto_fill_M_for_row(r)
        if c in (self.COL_HWEB, self.COL_DOUBLE_WEB_OFFSET) and r == self.tbl.currentRow():
            self._repaint_preview_from_selection()
        if c in (self.COL_X, self.COL_HWEB, self.COL_DOUBLE_WEB_OFFSET):
            self._emit_inertia_inputs_changed()

    def _auto_fill_M_for_row(self, r: int):
        x = _try_float(_get_text(self.tbl, r, self.COL_X))
        self.tbl.blockSignals(True)
        if self._moment_provider is None or x is None:
            _set_item(self.tbl, r, self.COL_M, "")
        else:
            try:
                M_kgcm = float(self._moment_provider(float(x)))
            except Exception:
                _set_item(self.tbl, r, self.COL_M, "")
            else:
                _set_item(self.tbl, r, self.COL_M, _fmt2(M_kgcm))
        _set_item_editable(self.tbl, r, self.COL_M, False)
        self.tbl.blockSignals(False)

    def _schedule_recompute(self, *args):
        if self._is_user_editing_input_cell():
            self._pending_recompute_after_edit = True
            self._timer.stop()
            return
        self._timer.start(120)

    def _set_row_color(self, r: int, ok: Optional[bool], *, paint_widgets: bool = True):
        editing_input = self._is_user_editing_input_cell()
        editing_row = self.tbl.currentRow() if editing_input else -1
        editing_col = self.tbl.currentColumn() if editing_input else -1
        col_input = QBrush(QColor(TABLE_INPUT_BG))
        col_readonly = QBrush(QColor(TABLE_READONLY_BG))
        col_ok = QBrush(QColor(TABLE_OK_BG))
        col_bad = QBrush(QColor(TABLE_ERROR_BG))
        fg = QBrush(QColor(TABLE_TEXT_COLOR))
        result_brush = col_readonly if ok is None else (col_ok if ok else col_bad)
        editable_cols = {self.COL_X, self.COL_HWEB}
        if self._row_double_web_enabled(r):
            editable_cols.add(self.COL_DOUBLE_WEB_OFFSET)
        computed_cols = {
            self.COL_FS,
            self.COL_JX,
            self.COL_YBAR,
            self.COL_CMAX,
            self.COL_WCRIT,
            self.COL_WREQ,
            self.COL_SIGMAX,
        }

        for c in range(self.tbl.columnCount()):
            it = self.tbl.item(r, c)
            if it is None:
                it = QTableWidgetItem("")
                it.setTextAlignment(Qt.AlignCenter)
                self.tbl.setItem(r, c, it)
            if r == editing_row and c == editing_col:
                continue
            if c in editable_cols:
                bg = col_input
            elif c in computed_cols:
                bg = result_brush
            else:
                bg = col_readonly
            it.setBackground(bg)
            it.setForeground(fg)

        if paint_widgets:
            w = self.tbl.cellWidget(r, self.COL_TWEB)
            if w is not None:
                w.setStyleSheet(combo_cell_style(TABLE_INPUT_BG))
            chk = self.tbl.cellWidget(r, self.COL_DOUBLE_WEB)
            if chk is not None:
                chk.setStyleSheet(combo_cell_style(TABLE_INPUT_BG))
        if r != editing_row:
            self._apply_row_editable_flags(r)

    def _set_out_cell(self, r: int, c: int, text: str):
        it = self.tbl.item(r, c)
        if it is None:
            it = QTableWidgetItem("")
            it.setTextAlignment(Qt.AlignCenter)
            self.tbl.setItem(r, c, it)
        it.setText(text)
        it.setFlags(it.flags() & ~Qt.ItemIsEditable)

    def _clear_out_cells(self, r: int):
        for c in [self.COL_FS, self.COL_JX, self.COL_YBAR, self.COL_CMAX, self.COL_WCRIT, self.COL_WREQ, self.COL_SIGMAX]:
            self._set_out_cell(r, c, "")

    def _recompute_all(self):
        if self._is_user_editing_input_cell():
            self._pending_recompute_after_edit = True
            self._timer.stop()
            return
        nmin = float(self.n_min.value())

        for r in range(self.tbl.rowCount()):
            try:
                hweb = _try_float(_get_text(self.tbl, r, self.COL_HWEB))
                x_value = _try_float(_get_text(self.tbl, r, self.COL_X))
                M = _try_float(_get_text(self.tbl, r, self.COL_M))
                tweb_in = _parse_frac_in(self._current_tweb_in(r))

                if hweb is None or hweb <= 0 or M is None:
                    self._clear_out_cells(r)
                    self._set_row_color(r, ok=None)
                    continue

                row_result = self._compute_section_row_result(
                    hweb=float(hweb),
                    tweb_in=tweb_in,
                    x_value=x_value,
                    m_value=M,
                    round_up_decimals=2,
                    row_index=r,
                )
                no_flex_demand = bool(row_result["no_flex_demand"])

                if row_result["chapon_context_error"] and not no_flex_demand:
                    self._clear_out_cells(r)
                    self._set_out_cell(r, self.COL_FS, "ERR CHAPÓN")
                    self._set_row_color(r, ok=False)
                    continue

                if row_result["missing_materials"] and not no_flex_demand:
                    self._clear_out_cells(r)
                    self._set_out_cell(r, self.COL_FS, "ERR MAT")
                    self._set_row_color(r, ok=False)
                    continue
                if (row_result["sigma_top_calc_adm"] is None or row_result["sigma_bot_calc_adm"] is None) and not no_flex_demand:
                    self._clear_out_cells(r)
                    self._set_out_cell(r, self.COL_FS, "ERR MAT")
                    self._set_row_color(r, ok=False)
                    continue

                res = row_result["res"]
                fs_final = row_result["fs_final"]
                sigma_final = row_result["sigma_final"]
                wreq_final = row_result["wreq_final"]
                if res is None or fs_final is None or sigma_final is None or wreq_final is None:
                    raise RuntimeError("No se pudo calcular la verificación de flexión.")

                self._set_out_cell(r, self.COL_JX, _fmt2(res.Jx_cm4))
                self._set_out_cell(r, self.COL_YBAR, _fmt2(res.ybar_cm))
                self._set_out_cell(r, self.COL_CMAX, _fmt2(res.cmax_cm))
                self._set_out_cell(r, self.COL_WCRIT, _fmt2(res.Wcrit_cm3))
                self._set_out_cell(r, self.COL_WREQ, _fmt2(wreq_final))
                self._set_out_cell(r, self.COL_SIGMAX, _fmt2(sigma_final))
                self._set_out_cell(r, self.COL_FS, FLEX_NO_DEMAND_TEXT if no_flex_demand else _fmt2(fs_final))

                self._set_row_color(r, ok=(fs_final >= nmin))

            except DoubleWebGeometryError:
                self._clear_out_cells(r)
                self._set_out_cell(r, self.COL_FS, "ERR DOBLE ALMA")
                self._set_row_color(r, ok=False)
            except Exception as e:
                # ✅ una fila con error no mata toda la tabla
                self._clear_out_cells(r)
                self._set_out_cell(r, self.COL_FS, "ERR")
                self._set_row_color(r, ok=False)
                print(f"[SectionCheckPanel] Error en fila {r+1}: {e}")

    # -------- Export ----------
    def _export_table_jpg(self):
        path, _ = QFileDialog.getSaveFileName(self, "Guardar tabla", "tabla_verificacion.jpg", "JPG (*.jpg)")
        if not path:
            return

        headers = [self.tbl.horizontalHeaderItem(i).text() for i in range(self.tbl.columnCount())]
        data = []
        row_ok = []
        nmin = float(self.n_min.value())

        for r in range(self.tbl.rowCount()):
            row = []
            for c in range(self.tbl.columnCount()):
                if c == self.COL_TWEB:
                    row.append(self._tweb_widgets[r].currentText())
                elif c == self.COL_DOUBLE_WEB:
                    row.append("Sí" if self._row_double_web_enabled(r) else "No")
                else:
                    row.append(_get_text(self.tbl, r, c))
            data.append(row)

            fs = _try_float(_get_text(self.tbl, r, self.COL_FS))
            row_ok.append(bool(fs is not None and fs >= nmin))

        fig = plt.Figure(figsize=(14.8, 2.2 + 0.32 * len(data)), dpi=200)
        ax = fig.add_subplot(111)
        ax.axis("off")

        tbl = ax.table(cellText=data, colLabels=headers, loc="center", cellLoc="center")
        tbl.auto_set_font_size(False)
        tbl.set_fontsize(9)
        tbl.scale(1.0, 1.5)

        for rr in range(len(data)):
            color = (0.78, 1.0, 0.78) if row_ok[rr] else (1.0, 0.82, 0.82)
            for cc in range(len(headers)):
                tbl[(rr + 1, cc)].set_facecolor(color)

        fig.tight_layout()
        fig.savefig(path, format="jpg", dpi=600)
        QMessageBox.information(self, "Exportación", "Tabla exportada a JPG (alta resolución).")

    # ------------------------------------------------------------
    # API pública (para Memoria de Cálculo)
    # ------------------------------------------------------------
    def export_table_jpg(self, path: str, *, dpi: int = 300) -> None:
        """Exporta la tabla completa (con colores) a JPG. No muestra diálogos."""
        try:
            # Sincroniza celdas calculadas antes de tomar la imagen.
            self.tbl.clearFocus()
            self._recompute_all()

            headers = [self.tbl.horizontalHeaderItem(c).text() for c in range(self.tbl.columnCount())]
            data = []
            row_ok = []
            nmin = float(self.n_min.value())
            for r in range(self.tbl.rowCount()):
                row = []
                for c in range(self.tbl.columnCount()):
                    if c == self.COL_TWEB:
                        row.append(self._tweb_widgets[r].currentText() if r < len(self._tweb_widgets) else _get_text(self.tbl, r, c))
                    elif c == self.COL_DOUBLE_WEB:
                        row.append("Sí" if self._row_double_web_enabled(r) else "No")
                    else:
                        row.append(_get_text(self.tbl, r, c))
                data.append(row)

                fs = _try_float(_get_text(self.tbl, r, self.COL_FS))
                row_ok.append(bool(fs is not None and fs >= nmin))

            fig = plt.Figure(figsize=(14.8, 2.2 + 0.32 * len(data)), dpi=200)
            ax = fig.add_subplot(111)
            ax.axis("off")

            tbl = ax.table(cellText=data, colLabels=headers, loc="center", cellLoc="center")
            tbl.auto_set_font_size(False)
            tbl.set_fontsize(9)
            tbl.scale(1.0, 1.5)

            for rr in range(len(data)):
                color = (0.78, 1.0, 0.78) if row_ok[rr] else (1.0, 0.82, 0.82)
                for cc in range(len(headers)):
                    tbl[(rr + 1, cc)].set_facecolor(color)

            fig.tight_layout()
            fig.savefig(path, format="jpg", dpi=int(dpi))
        except Exception as e:
            raise RuntimeError(f"No se pudo exportar tabla de sección a JPG: {e}")

    def _collect_table_export_data(self) -> tuple[list[str], list[list[str]], list[bool]]:
        headers = [self.tbl.horizontalHeaderItem(c).text() for c in range(self.tbl.columnCount())]
        data: list[list[str]] = []
        row_ok: list[bool] = []
        nmin = float(self.n_min.value())

        for r in range(self.tbl.rowCount()):
            row = []
            for c in range(self.tbl.columnCount()):
                if c == self.COL_TWEB:
                    row.append(self._tweb_widgets[r].currentText() if r < len(self._tweb_widgets) else _get_text(self.tbl, r, c))
                elif c == self.COL_DOUBLE_WEB:
                    row.append("Sí" if self._row_double_web_enabled(r) else "No")
                else:
                    row.append(_get_text(self.tbl, r, c))
            data.append(row)

            fs = _try_float(_get_text(self.tbl, r, self.COL_FS))
            row_ok.append(bool(fs is not None and fs >= nmin))

        return headers, data, row_ok

    def _collect_section_export_cards(self) -> list[dict[str, Any]]:
        cards: list[dict[str, Any]] = []
        nmin = float(self.n_min.value())
        sigma_web_adm = self._mat_sigma(self._current_material_id(self.cmb_mat_web))
        t_top_in = _parse_frac_in(self._current_thickness_in(self.cmb_t_top))
        t_bot_in = _parse_frac_in(self._current_thickness_in(self.cmb_t_bot))
        for r in range(self.tbl.rowCount()):
            hweb = _try_float(_get_text(self.tbl, r, self.COL_HWEB))
            x_value = _try_float(_get_text(self.tbl, r, self.COL_X))
            if hweb is None or hweb <= 0.0:
                continue
            M_value = _try_float(_get_text(self.tbl, r, self.COL_M))
            try:
                tweb_in = _parse_frac_in(self._current_tweb_in(r))
                row_result = self._compute_section_row_result(
                    hweb=float(hweb),
                    tweb_in=tweb_in,
                    x_value=x_value,
                    m_value=M_value,
                    round_up_decimals=4,
                    row_index=r,
                )
            except DoubleWebGeometryError as e:
                double_web_payload = self._double_web_payload_for_row(r)
                cards.append(
                    {
                        "sec": _get_text(self.tbl, r, self.COL_SEC) or str(r + 1),
                        "x_mm": _get_text(self.tbl, r, self.COL_X),
                        "h_web_mm": float(hweb),
                        "t_web_in": self._current_tweb_in(r),
                        "t_web_mm": _in_to_mm(float(tweb_in)),
                        "table_values": {
                            "FS": "ERR DOBLE ALMA",
                            "Jx_cm4": "",
                            "ybar_cm": "",
                            "cmax_cm": "",
                            "Wcrit_cm3": "",
                            "Wreq_cm3": "",
                            "sigma_max_kgcm2": "",
                        },
                        "fs_text": "ERR DOBLE ALMA",
                        "ok": False,
                        **double_web_payload,
                        "double_web_error": str(e),
                        "moment_kgcm": M_value,
                    }
                )
                continue
            except Exception:
                continue

            fs_text = _get_text(self.tbl, r, self.COL_FS)
            fs_value = _try_float(fs_text)
            sec = row_result["sec"]
            no_flex_demand = bool(row_result["no_flex_demand"])
            chapon_context_error = row_result["chapon_context_error"]
            chapon_context_missing_fields = row_result["chapon_context_missing_fields"]
            V_value = None
            if self._shear_provider is not None and x_value is not None:
                try:
                    V_value = float(self._shear_provider(float(x_value)))
                except Exception:
                    V_value = None

            res = row_result["res"]
            sigma_top_calc_adm = row_result["sigma_top_calc_adm"]
            sigma_bot_calc_adm = row_result["sigma_bot_calc_adm"]
            component_checks = row_result["component_checks"]
            missing_materials = row_result["missing_materials"]
            governing_component = row_result["governing_component"]
            wreq_top = row_result["wreq_top"]
            wreq_bot = row_result["wreq_bot"]
            fs_top = row_result["fs_top"]
            fs_bot = row_result["fs_bot"]
            double_web_payload = self._double_web_payload_for_row(r)
            if no_flex_demand:
                fs_value = math.inf
                fs_text = FLEX_NO_DEMAND_TEXT
            elif chapon_context_error:
                fs_value = None
                fs_text = "ERR CHAPÓN"
            elif missing_materials:
                fs_value = None
                fs_text = "ERR MAT"
            elif governing_component is not None:
                fs_value = row_result["fs_final"]
                fs_text = _fmt2(fs_value)
            shear = self._compute_shear_check_data(sec, V_value, sigma_web_adm)
            cards.append(
                {
                    "sec": _get_text(self.tbl, r, self.COL_SEC) or str(r + 1),
                    "x_mm": _get_text(self.tbl, r, self.COL_X),
                    "h_web_mm": float(hweb),
                    "t_web_in": self._current_tweb_in(r),
                    "table_values": {
                        "FS": _get_text(self.tbl, r, self.COL_FS),
                        "Jx_cm4": _get_text(self.tbl, r, self.COL_JX),
                        "ybar_cm": _get_text(self.tbl, r, self.COL_YBAR),
                        "cmax_cm": _get_text(self.tbl, r, self.COL_CMAX),
                        "Wcrit_cm3": _get_text(self.tbl, r, self.COL_WCRIT),
                        "Wreq_cm3": _get_text(self.tbl, r, self.COL_WREQ),
                        "sigma_max_kgcm2": _get_text(self.tbl, r, self.COL_SIGMAX),
                    },
                    "fs_text": fs_text or "-",
                    "ok": fs_value is not None and fs_value >= nmin,
                    "material_error": row_result["material_error"],
                    "missing_material_components": missing_materials,
                    "section": sec,
                    "include_bastidor_lateral": bool(self.chk_bastidor_lateral.isChecked()),
                    "bastidor_lateral_structural": True,
                    "bastidor_lateral_included": isinstance(sec, CompositeSection) and sec.includes_bastidor_lateral,
                    "bastidor_lateral_height_mm": float(self.n_bastidor_lateral_altura.value()),
                    "include_piso": bool(self.chk_piso.isChecked()),
                    "piso_included": isinstance(sec, CompositeSection) and sec.includes_piso,
                    "material_piso": self._current_material_id(self.cmb_mat_piso),
                    "espesor_piso": self._current_piso_thickness_mm(),
                    "piso_width_mm": self._current_piso_width_mm(),
                    "include_chapon": bool(self.chk_chapon.isChecked()),
                    "chapon_included": isinstance(sec, CompositeSection) and sec.includes_chapon,
                    "chapon_context_error": bool(chapon_context_error),
                    "chapon_context_missing_fields": chapon_context_missing_fields,
                    **double_web_payload,
                    "double_web_error": row_result["double_web_error"],
                    "material_chapon": CHAPON_MATERIAL_ID,
                    "espesor_chapon": self._current_chapon_thickness_mm(),
                    "ancho_chapon": CHAPON_ANCHO_MM,
                    "chapon_length_mm": self._chapon_length_mm(),
                    "chapon_x_start_mm": 0.0,
                    "chapon_x_end_mm": self._chapon_end_mm(),
                    "moment_kgcm": M_value,
                    "sigma_top_adm_kgcm2": sigma_top_calc_adm,
                    "sigma_base_top_adm_kgcm2": row_result["sigma_base_top_adm"],
                    "sigma_piso_adm_kgcm2": row_result["sigma_piso_adm"],
                    "sigma_bot_adm_kgcm2": sigma_bot_calc_adm,
                    "sigma_base_bot_adm_kgcm2": row_result["sigma_base_bot_adm"],
                    "sigma_chapon_adm_kgcm2": row_result["sigma_chapon_adm"],
                    "sigma_web_adm_kgcm2": sigma_web_adm,
                    "t_top_in": t_top_in,
                    "t_bot_in": t_bot_in,
                    "t_top_mm": float(sec.t_top_mm),
                    "t_bot_mm": float(sec.t_bot_mm),
                    "t_web_mm": float(sec.t_web_mm),
                    "b_f_mm": float(sec.b_f_mm),
                    "h_total_mm": row_result["h_total_mm"],
                    "ix_single_cm4": row_result["ix_single_cm4"],
                    "ix_total_cm4": row_result["ix_total_cm4"],
                    "ybar_cm": row_result["ybar_cm"],
                    "c_top_cm": row_result["c_top_cm"],
                    "c_bot_cm": row_result["c_bot_cm"],
                    "cmax_cm": row_result["cmax_cm"],
                    "wcrit_cm3": row_result["wcrit_cm3"],
                    "wreq_top_cm3": wreq_top,
                    "wreq_bot_cm3": wreq_bot,
                    "sigma_top_kgcm2": row_result["sigma_top"],
                    "sigma_bot_kgcm2": row_result["sigma_bot"],
                    "fs_top": fs_top,
                    "fs_bot": fs_bot,
                    "component_checks": component_checks,
                    "governing_component": governing_component,
                    "result": res,
                    "shear": shear,
                }
            )
        return cards

    def _render_section_report_image(self, sec: ISection | CompositeSection, path: str, *, dpi: int) -> None:
        width_mm = 68.0
        height_mm = 58.0
        fig = plt.Figure(figsize=(width_mm / 25.4, height_mm / 25.4), dpi=160)
        ax = fig.add_subplot(111)
        self._draw_section_preview_on(ax, sec)
        fig.subplots_adjust(left=0.06, right=0.94, top=0.94, bottom=0.06)
        fig.savefig(path, format="png", dpi=int(max(120, dpi)), facecolor="white")
        plt.close(fig)

    def _build_section_report_title(self, card: dict[str, Any]) -> str:
        x_text = card["x_mm"] or "-"
        return (
            f"Seccion {card['sec']} | x = {x_text} mm\n"
            f"h_viga = {_fmt_int(card['h_web_mm'])} mm | tw = {card['t_web_in']} in | FS = {card['fs_text']}"
        )

    def _compute_shear_check_data(self, sec: ISection | CompositeSection, v_kg: Optional[float], sigma_web_adm_kgcm2: Optional[float]) -> Dict[str, Any]:
        if isinstance(sec, CompositeSection):
            return self._compute_composite_shear_check_data(sec, v_kg, sigma_web_adm_kgcm2)

        props = sec.props_mm()
        ybar_mm = float(props["ybar_mm"])
        h_total_mm = float(props["H_mm"])
        layers = [
            (0.0, float(sec.t_bot_mm), float(sec.b_f_mm)),
            (float(sec.t_bot_mm), float(sec.t_bot_mm + sec.h_web_mm), float(sec.t_web_mm)),
            (float(sec.t_bot_mm + sec.h_web_mm), h_total_mm, float(sec.b_f_mm)),
        ]

        q_mm3 = 0.0
        for y0, y1, width_mm in layers:
            ya = max(ybar_mm, y0)
            yb = min(h_total_mm, y1)
            if yb <= ya:
                continue
            area_mm2 = width_mm * (yb - ya)
            y_centroid_mm = 0.5 * (ya + yb)
            q_mm3 += area_mm2 * (y_centroid_mm - ybar_mm)

        ix_total_cm4 = (float(props["Ix_mm4"]) * float(self.n_beams)) / (10.0 ** 4)
        q_cm3 = q_mm3 / (10.0 ** 3)
        if ybar_mm <= float(sec.t_bot_mm) + 1e-9:
            t_na_mm = float(sec.b_f_mm)
            zone = "ala inferior"
        elif ybar_mm >= float(sec.t_bot_mm + sec.h_web_mm) - 1e-9:
            t_na_mm = float(sec.b_f_mm)
            zone = "ala superior"
        else:
            t_na_mm = float(sec.t_web_mm)
            zone = "alma"
        t_na_cm = t_na_mm / 10.0

        tau_max = None
        if v_kg is not None:
            tau_max = abs(float(v_kg)) * q_cm3 / max(ix_total_cm4 * t_na_cm, 1e-12)

        tau_adm = None if sigma_web_adm_kgcm2 is None else float(sigma_web_adm_kgcm2) / math.sqrt(3.0)
        fs_shear = None
        if tau_max is not None and tau_adm is not None:
            fs_shear = tau_adm / max(tau_max, 1e-12)

        return {
            "V_kg": v_kg,
            "Q_cm3": q_cm3,
            "t_na_cm": t_na_cm,
            "zone": zone,
            "tau_max_kgcm2": tau_max,
            "tau_adm_kgcm2": tau_adm,
            "fs_shear": fs_shear,
        }

    def _compute_composite_shear_check_data(
        self,
        sec: CompositeSection,
        v_kg: Optional[float],
        sigma_web_adm_kgcm2: Optional[float],
    ) -> Dict[str, Any]:
        props = sec.props_mm()
        ybar_global_mm = float(props.get("ybar_global_mm", props["ybar_mm"]))
        y_max_mm = float(props.get("y_max_mm", props["H_mm"]))
        q_mm3 = 0.0
        t_na_mm = 0.0
        for rect in sec.rects:
            y0 = float(rect.y0_mm)
            y1 = float(rect.y1_mm)
            if y1 > ybar_global_mm:
                ya = max(ybar_global_mm, y0)
                yb = min(y_max_mm, y1)
                if yb > ya:
                    area_mm2 = float(rect.b_mm) * (yb - ya)
                    y_centroid_mm = 0.5 * (ya + yb)
                    q_mm3 += area_mm2 * (y_centroid_mm - ybar_global_mm)
            if y0 <= ybar_global_mm <= y1:
                t_na_mm += float(rect.b_mm)

        ix_total_cm4 = float(props["Ix_mm4"]) / (10.0 ** 4)
        q_cm3 = q_mm3 / (10.0 ** 3)
        t_na_cm = t_na_mm / 10.0
        tau_max = None
        if v_kg is not None:
            tau_max = abs(float(v_kg)) * q_cm3 / max(ix_total_cm4 * t_na_cm, 1e-12)

        tau_adm = None if sigma_web_adm_kgcm2 is None else float(sigma_web_adm_kgcm2) / math.sqrt(3.0)
        fs_shear = None
        if tau_max is not None and tau_adm is not None:
            fs_shear = tau_adm / max(tau_max, 1e-12)

        return {
            "V_kg": v_kg,
            "Q_cm3": q_cm3,
            "t_na_cm": t_na_cm,
            "zone": "sección compuesta",
            "tau_max_kgcm2": tau_max,
            "tau_adm_kgcm2": tau_adm,
            "fs_shear": fs_shear,
        }

    def build_verification_export_payload(self, tmpdir: str, *, dpi: int = 300) -> Dict[str, Any]:
        """Arma datos del verificador para el pipeline único de memoria_calculo_docx."""
        self.tbl.clearFocus()
        self._recompute_all()

        headers, data, row_ok = self._collect_table_export_data()
        cards = self._collect_section_export_cards()
        serial_cards: list[dict[str, Any]] = []
        for idx, card in enumerate(cards):
            item = dict(card)
            section = item.pop("section", None)
            if section is not None:
                image_path = os.path.join(tmpdir, f"section_{idx + 1}.png")
                self._render_section_report_image(section, image_path, dpi=dpi)
                item["image_path"] = image_path
            item["title"] = self._build_section_report_title(card)
            serial_cards.append(item)

        return {
            "headers": headers,
            "data": data,
            "row_ok": row_ok,
            "cards": serial_cards,
            "fs_required": float(self.n_min.value()),
            "n_beams": int(self.n_beams),
            "include_bastidor_lateral": bool(self.chk_bastidor_lateral.isChecked()),
            "bastidor_lateral_structural": True,
            "bastidor_lateral_included": bool(self._include_bastidor_lateral_in_geometry()),
            "bastidor_lateral_height_mm": float(self.n_bastidor_lateral_altura.value()),
            "include_piso": bool(self.chk_piso.isChecked()),
            "piso_included": bool(self._include_piso_in_geometry()),
            "material_piso": self._current_material_id(self.cmb_mat_piso),
            "espesor_piso": self._current_piso_thickness_mm(),
            "piso_width_mm": self._current_piso_width_mm(),
            "include_chapon": bool(self.chk_chapon.isChecked()),
            "chapon_included": bool(self.chk_chapon.isChecked() and self._chapon_end_mm() is not None),
            "chapon_context_error": bool(self._chapon_context_missing()),
            "chapon_context_missing_fields": self._chapon_context_missing_fields() if self._chapon_context_missing() else [],
            "material_chapon": CHAPON_MATERIAL_ID,
            "espesor_chapon": self._current_chapon_thickness_mm(),
            "ancho_chapon": CHAPON_ANCHO_MM,
            "chapon_length_mm": self._chapon_length_mm(),
            "chapon_x_start_mm": 0.0,
            "chapon_x_end_mm": self._chapon_end_mm(),
            "deflection_context": dict(self._deflection_context) if self._deflection_context else None,
        }

    def export_verification_report(self, path: str, *, dpi: int = 300) -> None:
        """Compatibilidad: exporta usando el pipeline único de memoria de cálculo."""
        try:
            with tempfile.TemporaryDirectory(prefix="semi_beam_verify_") as td:
                verification = self.build_verification_export_payload(td, dpi=dpi)
                export_memoria_docx(
                    path,
                    header=MemoriaHeader(titulo="Memoria de Cálculo - Verificación de viga", fecha=None),
                    seccion=MemoriaSeccion(fs_min=float(self.n_min.value()), n_vigas=int(self.n_beams)),
                    verification=verification,
                )
        except Exception as e:
            raise RuntimeError(f"No se pudo exportar la memoria de cálculo: {e}")

    def _build_docx_verification_report(
        self,
        path: str,
        *,
        cards: list[dict[str, Any]],
        headers: list[str],
        data: list[list[str]],
        row_ok: list[bool],
        dpi: int,
    ) -> None:
        """Compatibilidad interna: delega al pipeline único de memoria_calculo_docx."""
        try:
            with tempfile.TemporaryDirectory(prefix="semi_beam_verify_") as td:
                serial_cards: list[dict[str, Any]] = []
                for idx, card in enumerate(cards):
                    item = dict(card)
                    section = item.pop("section", None)
                    if section is not None:
                        image_path = os.path.join(td, f"section_{idx + 1}.png")
                        self._render_section_report_image(section, image_path, dpi=dpi)
                        item["image_path"] = image_path
                    item["title"] = self._build_section_report_title(card)
                    serial_cards.append(item)
                export_memoria_docx(
                    path,
                    header=MemoriaHeader(titulo="Memoria de Cálculo - Verificación de viga", fecha=None),
                    seccion=MemoriaSeccion(fs_min=float(self.n_min.value()), n_vigas=int(self.n_beams)),
                    verification={
                        "headers": headers,
                        "data": data,
                        "row_ok": row_ok,
                        "cards": serial_cards,
                        "fs_required": float(self.n_min.value()),
                        "n_beams": int(self.n_beams),
                        "deflection_context": dict(self._deflection_context) if self._deflection_context else None,
                    },
                )
        except Exception as e:
            raise RuntimeError(f"No se pudo exportar la memoria de cálculo: {e}")

    def extract_memoria_data(self) -> dict:
        """Extrae datos necesarios para exportar Memoria de Cálculo (sin dependencias del motor)."""
        def _sigma(label: str) -> str:
            v = self._mat_sigma(label)
            return "-" if v is None else f"{float(v):g}"

        out = {
            "material_top": self._current_material_id(self.cmb_mat_top),
            "material_bot": self._current_material_id(self.cmb_mat_bot),
            "material_web": self._current_material_id(self.cmb_mat_web),
            "material_piso": self._current_material_id(self.cmb_mat_piso),
            "sigma_top_kgcm2": _sigma(self._current_material_id(self.cmb_mat_top)),
            "sigma_bot_kgcm2": _sigma(self._current_material_id(self.cmb_mat_bot)),
            "sigma_web_kgcm2": _sigma(self._current_material_id(self.cmb_mat_web)),
            "sigma_piso_kgcm2": _sigma(self._current_material_id(self.cmb_mat_piso)),
            "t_top_in": self._current_thickness_in(self.cmb_t_top),
            "t_bot_in": self._current_thickness_in(self.cmb_t_bot),
            "bf_text": self.lbl_bf.text(),
            "fs_min": float(self.n_min.value()),
            "n_beams": int(self.n_beams),
            "include_bastidor_lateral": bool(self.chk_bastidor_lateral.isChecked()),
            "bastidor_lateral_structural": True,
            "bastidor_lateral_height_mm": float(self.n_bastidor_lateral_altura.value()),
            "include_piso": bool(self.chk_piso.isChecked()),
            "espesor_piso": self._current_piso_thickness_mm(),
            "piso_width_mm": self._current_piso_width_mm(),
            "include_chapon": bool(self.chk_chapon.isChecked()),
            "chapon_context_error": bool(self._chapon_context_missing()),
            "chapon_context_missing_fields": self._chapon_context_missing_fields() if self._chapon_context_missing() else [],
            "material_chapon": CHAPON_MATERIAL_ID,
            "espesor_chapon": self._current_chapon_thickness_mm(),
            "ancho_chapon": CHAPON_ANCHO_MM,
            "chapon_length_mm": self._chapon_length_mm(),
            "chapon_x_start_mm": 0.0,
            "chapon_x_end_mm": self._chapon_end_mm(),
            "deflection": dict(self._deflection_context) if self._deflection_context else None,
            "rows": [],
        }

        for r in range(self.tbl.rowCount()):
            x_value = _try_float(_get_text(self.tbl, r, self.COL_X))
            v_value = None
            if self._shear_provider is not None and x_value is not None:
                try:
                    v_value = float(self._shear_provider(float(x_value)))
                except Exception:
                    v_value = None
            row = {
                "sec": _get_text(self.tbl, r, self.COL_SEC),
                "x_mm": _get_text(self.tbl, r, self.COL_X),
                "h_web_mm": _get_text(self.tbl, r, self.COL_HWEB),
                "t_web_in": self._current_tweb_in(r),
                "M_kgcm": _get_text(self.tbl, r, self.COL_M),
                "V_kg": "" if v_value is None else _fmt2(v_value),
                "FS": _get_text(self.tbl, r, self.COL_FS),
                "Jx_cm4": _get_text(self.tbl, r, self.COL_JX),
                "ybar_cm": _get_text(self.tbl, r, self.COL_YBAR),
                "cmax_cm": _get_text(self.tbl, r, self.COL_CMAX),
                "Wcrit_cm3": _get_text(self.tbl, r, self.COL_WCRIT),
                "Wreq_cm3": _get_text(self.tbl, r, self.COL_WREQ),
                "sigma_max": _get_text(self.tbl, r, self.COL_SIGMAX),
            }
            out["rows"].append(row)

        return out

    def build_inertia_profile_mm4(self, x_mm) -> Optional["np.ndarray"]:
        import numpy as np

        xq = np.asarray(x_mm, dtype=float).reshape(-1)
        if xq.size == 0:
            return None

        samples = {}
        for r in range(self.tbl.rowCount()):
            x_row = _try_float(_get_text(self.tbl, r, self.COL_X))
            hweb = _try_float(_get_text(self.tbl, r, self.COL_HWEB))
            if x_row is None or hweb is None or hweb <= 0.0:
                continue
            try:
                tweb_in = _parse_frac_in(self._current_tweb_in(r))
                sec = self._make_section(
                    hweb,
                    tweb_in,
                    station_mm=x_row,
                    double_web_enabled=self._row_double_web_enabled(r),
                    double_web_inner_face_offset_mm=self._row_double_web_offset_mm(r),
                )
                ix_mm4 = float(sec.props_mm()["Ix_mm4"]) * float(self._calculation_n_beams(sec))
            except Exception:
                continue
            samples[float(x_row)] = ix_mm4

        if not samples:
            return None

        xs = np.asarray(sorted(samples.keys()), dtype=float)
        ix = np.asarray([samples[xi] for xi in xs], dtype=float)
        if xs.size == 1:
            return np.full_like(xq, float(ix[0]), dtype=float)
        return np.interp(xq, xs, ix, left=float(ix[0]), right=float(ix[-1]))

    def export_state(self) -> Dict[str, Any]:
        rows: List[Dict[str, str]] = []
        for r in range(self.tbl.rowCount()):
            rows.append(
                {
                    "x_mm": _get_text(self.tbl, r, self.COL_X),
                    "h_web_mm": _get_text(self.tbl, r, self.COL_HWEB),
                    "t_web_in": self._current_tweb_in(r),
                    "double_web_enabled": self._row_double_web_enabled(r),
                    "double_web_inner_face_offset_mm": self._row_double_web_offset_mm(r),
                }
            )
        return {
            "material_top": self._current_material_id(self.cmb_mat_top),
            "material_bot": self._current_material_id(self.cmb_mat_bot),
            "material_web": self._current_material_id(self.cmb_mat_web),
            "material_piso": self._current_material_id(self.cmb_mat_piso),
            "t_top_in": self._current_thickness_in(self.cmb_t_top),
            "t_bot_in": self._current_thickness_in(self.cmb_t_bot),
            "fs_min": float(self.n_min.value()),
            "include_bastidor_lateral": bool(self.chk_bastidor_lateral.isChecked()),
            "bastidor_lateral_structural": True,
            "bastidor_lateral_height_mm": float(self.n_bastidor_lateral_altura.value()),
            "include_piso": bool(self.chk_piso.isChecked()),
            "espesor_piso": self._current_piso_thickness_mm(),
            "piso_width_mm": self._current_piso_width_mm(),
            "include_chapon": bool(self.chk_chapon.isChecked()),
            "espesor_chapon": self._current_chapon_thickness_mm(),
            "ancho_chapon": CHAPON_ANCHO_MM,
            "chapon_length_mm": self._chapon_length_mm() or CHAPON_LENGTH_DEFAULT_MM,
            "chapon_x_start_mm": 0.0,
            "chapon_x_end_mm": self._chapon_end_mm(),
            "rows": rows,
        }

    def import_state(self, state: Optional[Dict[str, Any]]) -> None:
        if not isinstance(state, dict):
            return

        def _set_combo_data(cmb: QComboBox, value: Any) -> None:
            idx = cmb.findData(value)
            if idx >= 0:
                cmb.setCurrentIndex(idx)

        self.blockSignals(True)
        self.tbl.blockSignals(True)
        try:
            _set_combo_data(self.cmb_mat_top, state.get("material_top"))
            _set_combo_data(self.cmb_mat_bot, state.get("material_bot"))
            _set_combo_data(self.cmb_mat_web, state.get("material_web"))
            _set_combo_data(self.cmb_mat_piso, state.get("material_piso"))
            _set_combo_data(self.cmb_t_top, state.get("t_top_in"))
            _set_combo_data(self.cmb_t_bot, state.get("t_bot_in"))

            fs_min = state.get("fs_min")
            if fs_min is not None:
                try:
                    self.n_min.setValue(float(fs_min))
                except Exception:
                    pass

            self.chk_bastidor_lateral.setChecked(bool(state.get("include_bastidor_lateral", False)))
            self.chk_bastidor_lateral_structural.setChecked(True)
            bastidor_height = state.get("bastidor_lateral_height_mm")
            if bastidor_height is not None:
                try:
                    self.n_bastidor_lateral_altura.setValue(float(bastidor_height))
                except Exception:
                    pass

            self.chk_piso.setChecked(bool(state.get("include_piso", False)))
            piso_width = state.get("piso_width_mm", state.get("ancho_piso", PISO_ANCHO_MM))
            try:
                self.n_piso_width.setValue(float(piso_width))
            except Exception:
                self.n_piso_width.setValue(PISO_ANCHO_MM)
            espesor_piso = state.get("espesor_piso")
            if espesor_piso is not None:
                try:
                    _set_combo_data(self.cmb_espesor_piso, float(espesor_piso))
                except Exception:
                    pass

            self.chk_chapon.setChecked(bool(state.get("include_chapon", False)))
            chapon_length = state.get("chapon_length_mm", CHAPON_LENGTH_DEFAULT_MM)
            try:
                self.n_chapon_length.setValue(float(chapon_length))
            except Exception:
                self.n_chapon_length.setValue(CHAPON_LENGTH_DEFAULT_MM)
            espesor_chapon = state.get("espesor_chapon")
            if espesor_chapon is not None:
                try:
                    _set_combo_data(self.cmb_espesor_chapon, float(espesor_chapon))
                except Exception:
                    pass
            rows = state.get("rows")
            if isinstance(rows, list):
                for r in range(self.tbl.rowCount()):
                    row = rows[r] if r < len(rows) and isinstance(rows[r], dict) else {}
                    _set_item(self.tbl, r, self.COL_X, str(row.get("x_mm", "")))
                    _set_item(self.tbl, r, self.COL_HWEB, str(row.get("h_web_mm", "")))
                    if r < len(self._tweb_widgets):
                        _set_combo_data(self._tweb_widgets[r], row.get("t_web_in"))
                    if r < len(self._double_web_widgets):
                        chk = self._double_web_widgets[r]
                        chk.blockSignals(True)
                        try:
                            chk.setChecked(bool(row.get("double_web_enabled", False)))
                        finally:
                            chk.blockSignals(False)
                    offset = row.get("double_web_inner_face_offset_mm")
                    _set_item(
                        self.tbl,
                        r,
                        self.COL_DOUBLE_WEB_OFFSET,
                        "" if offset is None else str(offset),
                    )
                    self._sync_double_web_row_controls(r)
        finally:
            self.tbl.blockSignals(False)
            self.blockSignals(False)

        self._update_sigma_labels()
        self._update_bastidor_lateral_controls()
        self._update_piso_controls()
        self._update_chapon_controls()
        self._repaint_preview_from_selection()
        self._recompute_all()
