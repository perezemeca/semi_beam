from __future__ import annotations

from typing import Callable, Optional, List

from PySide6.QtCore import Qt, QTimer, Signal
from PySide6.QtGui import QColor, QBrush
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QFormLayout, QLabel,
    QDoubleSpinBox, QComboBox, QPushButton, QTableWidget, QTableWidgetItem, QSizePolicy,
    QFileDialog, QMessageBox, QStyle
)

import matplotlib
matplotlib.use("QtAgg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.patches import FancyArrowPatch

from semi_beam.sections.i_section import ISection, IN_TO_MM
from semi_beam.sections.flex_check import compute_flex_row
from semi_beam.ui.numeric_delegate import (
    NullableFloatDelegate,
    SpinBoxDelegate,
    apply_table_readability_style,
    combo_cell_style,
    TABLE_ERROR_BG,
    TABLE_INPUT_BG,
    TABLE_OK_BG,
    TABLE_READONLY_BG,
    TABLE_TEXT_COLOR,
)

from semi_beam.materials.material_db import MaterialDB, default_materials_path


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
    t = (s or "").strip().replace(",", ".")
    if t == "":
        return None
    try:
        return float(t)
    except Exception:
        return None


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


class SectionCheckPanel(QWidget):
    inertia_inputs_changed = Signal()

    COL_SEC = 0
    COL_X = 1
    COL_HWEB = 2
    COL_TWEB = 3
    COL_FS = 4
    COL_M = 5

    COL_JX = 6
    COL_YBAR = 7
    COL_CMAX = 8
    COL_WCRIT = 9
    COL_WREQ = 10
    COL_SIGMAX = 11

    TWEB_OPTIONS = ["3/16", "1/4", "5/16", "3/8", "1/2", "5/8"]

    def __init__(self, parent=None):
        super().__init__(parent)

        self._moment_provider: Optional[Callable[[float], float]] = None
        self.n_beams = 2

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
        self._populate_thickness_combo(self.cmb_t_top, ["1/2", "5/8", "3/4"], default="1/2")
        self._populate_thickness_combo(self.cmb_t_bot, ["1/2", "5/8", "3/4"], default="1/2")
        _configure_combo_for_contents(self.cmb_t_top)
        _configure_combo_for_contents(self.cmb_t_bot)

        self.lbl_bf = QLabel("5.0 in - 127 mm")
        self.lbl_info = QLabel("Cálculo: 2 vigas idénticas (I y Wcrit se duplican).")
        self.lbl_info.setWordWrap(True)

        self.cmb_mat_top = QComboBox()
        self.cmb_mat_bot = QComboBox()
        self.cmb_mat_web = QComboBox()
        _configure_combo_for_contents(self.cmb_mat_top)
        _configure_combo_for_contents(self.cmb_mat_bot)
        _configure_combo_for_contents(self.cmb_mat_web)

        self.n_min = QDoubleSpinBox()
        self.n_min.setRange(1.0, 100.0)
        self.n_min.setDecimals(2)
        self.n_min.setValue(2.9)
        self.n_min.setSingleStep(0.1)

        self._populate_material_combos()

        form.addRow("Material planchuela superior:", self.cmb_mat_top)
        form.addRow("Material planchuela inferior:", self.cmb_mat_bot)
        form.addRow("Material alma:", self.cmb_mat_web)
        form.addRow("Espesor planchuela superior - 5.0 in - 127 mm:", self.cmb_t_top)
        form.addRow("Espesor planchuela inferior - 5.0 in - 127 mm:", self.cmb_t_bot)
        form.addRow("FS mínimo:", self.n_min)

        lay.addLayout(form)
        lay.addWidget(self.lbl_info)

        # Preview
        self.fig = plt.Figure(figsize=(3.8, 2.5))
        self.ax = self.fig.add_subplot(111)
        self.canvas = FigureCanvas(self.fig)
        lay.addWidget(self.canvas)

        lay.addWidget(QLabel("Secciones: cargar x, h_viga y Espesor. M se autocompleta desde M(x)."))

        # Tabla
        self.tbl = QTableWidget(8, 12)
        self.tbl.setHorizontalHeaderLabels([
            "Sección", "x [mm]", "h_viga [mm]", "Espesor", "FS", "M [kg·cm]",
            "Jx [cm^4]", "ȳ [cm]", "c_max [cm]", "Wcrit [cm^3]", "Wreq [cm^3]", "σmax [kg/cm²]"
        ])
        self.tbl.horizontalHeader().setStretchLastSection(True)
        apply_table_readability_style(self.tbl)

        self._tweb_widgets: List[QComboBox] = []
        for r in range(self.tbl.rowCount()):
            _set_item(self.tbl, r, self.COL_SEC, str(r + 1))
            _set_item(self.tbl, r, self.COL_X, "")
            _set_item(self.tbl, r, self.COL_HWEB, "")
            _set_item(self.tbl, r, self.COL_FS, "")
            _set_item(self.tbl, r, self.COL_M, "")
            for c in [self.COL_JX, self.COL_YBAR, self.COL_CMAX, self.COL_WCRIT, self.COL_WREQ, self.COL_SIGMAX]:
                _set_item(self.tbl, r, c, "")

            _set_item_editable(self.tbl, r, self.COL_SEC, False)
            _set_item_editable(self.tbl, r, self.COL_X, True)
            _set_item_editable(self.tbl, r, self.COL_HWEB, True)
            _set_item_editable(self.tbl, r, self.COL_FS, False)
            _set_item_editable(self.tbl, r, self.COL_M, False)
            for c in [self.COL_JX, self.COL_YBAR, self.COL_CMAX, self.COL_WCRIT, self.COL_WREQ, self.COL_SIGMAX]:
                _set_item_editable(self.tbl, r, c, False)

            cmb = QComboBox()
            self._populate_thickness_combo(cmb, self.TWEB_OPTIONS, default="1/4")
            _configure_combo_for_contents(cmb)
            cmb.setStyleSheet(combo_cell_style(TABLE_INPUT_BG))
            cmb.currentTextChanged.connect(lambda _t, rr=r: self._schedule_recompute())
            cmb.currentTextChanged.connect(lambda _t: self._emit_inertia_inputs_changed())
            self.tbl.setCellWidget(r, self.COL_TWEB, cmb)
            self._tweb_widgets.append(cmb)
            # cuando cambia t_web (combo), ya tenías schedule; sumá repaint si es la fila actual
            for r, cmb in enumerate(self._tweb_widgets):
                cmb.currentTextChanged.connect(lambda _t, rr=r: self._repaint_preview_from_selection() if rr == self.tbl.currentRow() else None)

        # Delegates numéricos
        self.tbl.setItemDelegateForColumn(self.COL_X, NullableFloatDelegate(self, decimals=2, minv=-1e12, maxv=1e12))
        self.tbl.setItemDelegateForColumn(self.COL_HWEB, SpinBoxDelegate(self, minv=0.0, maxv=5000.0, decimals=1, step=10.0))

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
        self.n_min.valueChanged.connect(self._schedule_recompute)

        self.tbl.itemSelectionChanged.connect(self._repaint_preview_from_selection)
        self.tbl.cellChanged.connect(lambda *_: self._schedule_recompute())
        self.tbl.itemChanged.connect(self._on_table_item_changed)

        self.btn_export.clicked.connect(self._export_table_jpg)

        self._update_sigma_labels()
        self._repaint_preview(h_web_override_mm=200.0, t_web_in=0.25)
        self._recompute_all()

    def _emit_inertia_inputs_changed(self):
        self.inertia_inputs_changed.emit()

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

    def _current_material_id(self, cmb: QComboBox) -> str:
        data = cmb.currentData()
        return "" if data is None else str(data)

    def _current_thickness_in(self, cmb: QComboBox) -> str:
        data = cmb.currentData()
        if data is None:
            return (cmb.currentText() or "").strip()
        return str(data)

    def _current_tweb_in(self, row: int) -> str:
        if 0 <= row < len(self._tweb_widgets):
            return self._current_thickness_in(self._tweb_widgets[row])
        return _get_text(self.tbl, row, self.COL_TWEB)

    def _populate_material_combos(self):
        for cmb in (self.cmb_mat_top, self.cmb_mat_bot, self.cmb_mat_web):
            cmb.blockSignals(True)
            cmb.clear()
            cmb.blockSignals(False)

        if self.mat_db is None:
            for cmb in (self.cmb_mat_top, self.cmb_mat_bot, self.cmb_mat_web):
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

        if self.mat_db.get("F36") is not None:
            idx_top = self.cmb_mat_top.findData("F36")
            idx_bot = self.cmb_mat_bot.findData("F36")
            if idx_top >= 0:
                self.cmb_mat_top.setCurrentIndex(idx_top)
            if idx_bot >= 0:
                self.cmb_mat_bot.setCurrentIndex(idx_bot)
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
        self.cmb_mat_top.setToolTip("-" if sigma_top is None else f"σ_adm = {_fmt2(sigma_top)} kg/cm²")
        self.cmb_mat_bot.setToolTip("-" if sigma_bot is None else f"σ_adm = {_fmt2(sigma_bot)} kg/cm²")
        self.cmb_mat_web.setToolTip("-" if sigma_web is None else f"σ_adm = {_fmt2(sigma_web)} kg/cm²")

    # -------- API ----------
    def set_moment_provider(self, fn: Optional[Callable[[float], float]]):
        self._moment_provider = fn
        for r in range(self.tbl.rowCount()):
            self._auto_fill_M_for_row(r)
        self._schedule_recompute()

    def clear_results_only(self, *, clear_moments_if_no_provider: bool = False):
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

    # -------- Preview ----------
    def _draw_dim_v(self, y1: float, y2: float, x_dim: float, x_obj: float, text: str, *, color="blue"):
        self.ax.plot([x_obj, x_dim], [y1, y1], color=color, linewidth=1.0)
        self.ax.plot([x_obj, x_dim], [y2, y2], color=color, linewidth=1.0)

        arr = FancyArrowPatch((x_dim, y1), (x_dim, y2),
                              arrowstyle="<->", mutation_scale=12,
                              linewidth=1.0, color=color)
        arr.set_fill(False)
        self.ax.add_patch(arr)

        ym = 0.5 * (y1 + y2)
        self.ax.text(x_dim, ym, text,
                     ha="center", va="center", fontsize=10, color=color,
                     bbox=dict(facecolor="white", edgecolor="none", pad=1.2))

    def _make_section(self, h_web_mm: float, t_web_in: float) -> ISection:
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

    def _repaint_preview(self, *, h_web_override_mm: float, t_web_in: float):
        sec = self._make_section(h_web_override_mm, t_web_in)
        p = sec.props_mm()

        H = float(p.get("H_mm", 0.0))
        b = float(sec.b_f_mm)
        t_top = float(sec.t_top_mm)
        t_bot = float(sec.t_bot_mm)
        h = float(sec.h_web_mm)
        tw = float(sec.t_web_mm)

        self.ax.clear()
        self.ax.add_patch(plt.Rectangle((0.0, 0.0), b, t_bot, fill=False, edgecolor="blue", linewidth=1.2))
        self.ax.add_patch(plt.Rectangle((b/2 - tw/2, t_bot), tw, h, fill=False, edgecolor="blue", linewidth=1.2))
        self.ax.add_patch(plt.Rectangle((0.0, t_bot + h), b, t_top, fill=False, edgecolor="blue", linewidth=1.2))

        x_obj_L = 0.0
        x_dim_H = -0.35 * b
        self._draw_dim_v(0.0, H, x_dim_H, x_obj_L, _fmt_int(H), color="blue")

        x_obj_R = b
        x_dim_h = b + 0.22 * b
        self._draw_dim_v(t_bot, t_bot + h, x_dim_h, x_obj_R, _fmt_int(h), color="blue")

        self.ax.set_aspect("equal", adjustable="box")
        self.ax.axis("off")

        xpad = 0.95 * b
        ypad_top = 0.35 * max(H, 1.0)
        ypad_bot = 0.25 * max(H, 1.0)
        self.ax.set_xlim(-xpad, b + xpad)
        self.ax.set_ylim(-ypad_bot, max(H, 1.0) + ypad_top)
        self.canvas.draw()

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
        if c == self.COL_HWEB and r == self.tbl.currentRow():
            self._repaint_preview_from_selection()
        if c in (self.COL_X, self.COL_HWEB):
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
        # ✅ NO borrar foco (esto era lo que te impedía editar bien)
        self._timer.start(120)

    def _set_row_color(self, r: int, ok: Optional[bool], *, paint_widgets: bool = True):
        col_input = QBrush(QColor(TABLE_INPUT_BG))
        col_readonly = QBrush(QColor(TABLE_READONLY_BG))
        col_ok = QBrush(QColor(TABLE_OK_BG))
        col_bad = QBrush(QColor(TABLE_ERROR_BG))
        fg = QBrush(QColor(TABLE_TEXT_COLOR))
        result_brush = col_readonly if ok is None else (col_ok if ok else col_bad)
        editable_cols = {self.COL_X, self.COL_HWEB}
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
        nmin = float(self.n_min.value())
        s_top = self._mat_sigma(self._current_material_id(self.cmb_mat_top))
        s_bot = self._mat_sigma(self._current_material_id(self.cmb_mat_bot))

        if s_top is None or s_bot is None:
            for r in range(self.tbl.rowCount()):
                self._clear_out_cells(r)
                self._set_row_color(r, ok=None)
            return

        for r in range(self.tbl.rowCount()):
            try:
                hweb = _try_float(_get_text(self.tbl, r, self.COL_HWEB))
                M = _try_float(_get_text(self.tbl, r, self.COL_M))
                tweb_in = _parse_frac_in(self._current_tweb_in(r))

                if hweb is None or hweb <= 0 or M is None:
                    self._clear_out_cells(r)
                    self._set_row_color(r, ok=None)
                    continue

                sec = self._make_section(hweb, tweb_in)

                res = compute_flex_row(
                    section=sec,
                    M_kgcm=M,
                    sigma_adm_kgcm2=float(min(s_top, s_bot)),
                    sigma_adm_top_kgcm2=float(s_top),
                    sigma_adm_bot_kgcm2=float(s_bot),
                    n_beams=self.n_beams,
                    round_up_decimals=2,
                )

                self._set_out_cell(r, self.COL_JX, _fmt2(res.Jx_cm4))
                self._set_out_cell(r, self.COL_YBAR, _fmt2(res.ybar_cm))
                self._set_out_cell(r, self.COL_CMAX, _fmt2(res.cmax_cm))
                self._set_out_cell(r, self.COL_WCRIT, _fmt2(res.Wcrit_cm3))
                self._set_out_cell(r, self.COL_WREQ, _fmt2(res.Wreq_cm3))
                self._set_out_cell(r, self.COL_SIGMAX, _fmt2(res.sigma_max_kgcm2))
                self._set_out_cell(r, self.COL_FS, _fmt2(res.FS))

                self._set_row_color(r, ok=(res.FS >= nmin))

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

    # ============================================================
    # API pública (para Memoria de Cálculo)
    # ============================================================
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
                    else:
                        row.append(_get_text(self.tbl, r, c))
                data.append(row)

                fs = _try_float(_get_text(self.tbl, r, self.COL_FS))
                row_ok.append(bool(fs is not None and fs >= nmin))

            import matplotlib.pyplot as plt

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

    def extract_memoria_data(self) -> dict:
        """Extrae datos necesarios para exportar Memoria de Cálculo (sin dependencias del motor)."""
        def _sigma(label: str) -> str:
            v = self._mat_sigma(label)
            return "-" if v is None else f"{float(v):g}"

        out = {
            "material_top": self._current_material_id(self.cmb_mat_top),
            "material_bot": self._current_material_id(self.cmb_mat_bot),
            "material_web": self._current_material_id(self.cmb_mat_web),
            "sigma_top_kgcm2": _sigma(self._current_material_id(self.cmb_mat_top)),
            "sigma_bot_kgcm2": _sigma(self._current_material_id(self.cmb_mat_bot)),
            "sigma_web_kgcm2": _sigma(self._current_material_id(self.cmb_mat_web)),
            "t_top_in": self._current_thickness_in(self.cmb_t_top),
            "t_bot_in": self._current_thickness_in(self.cmb_t_bot),
            "bf_text": self.lbl_bf.text(),
            "fs_min": float(self.n_min.value()),
            "n_beams": int(self.n_beams),
            "rows": [],
        }

        for r in range(self.tbl.rowCount()):
            row = {
                "sec": _get_text(self.tbl, r, self.COL_SEC),
                "x_mm": _get_text(self.tbl, r, self.COL_X),
                "h_web_mm": _get_text(self.tbl, r, self.COL_HWEB),
                "t_web_in": self._current_tweb_in(r),
                "M_kgcm": _get_text(self.tbl, r, self.COL_M),
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
                sec = self._make_section(hweb, tweb_in)
                ix_mm4 = float(sec.props_mm()["Ix_mm4"]) * float(self.n_beams)
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
