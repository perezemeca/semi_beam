from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

from PySide6.QtCore import Qt, QThread, QTimer, Signal
from PySide6.QtGui import QColor, QBrush
from PySide6.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QProgressBar,
    QSlider,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from semi_beam.domain.beam import Beam
from semi_beam.domain.loads import DistUniform, PointForce, PointMoment
from semi_beam.engine.constraints import check_no_overlap, dist_interval
from semi_beam.engine.optimizer_loads import OptimizerConfig, OptimizerSolution, search_configuration
from semi_beam.engine.reactions import ReactionLoad, ReactionsResult, solve_reactions_2support, solve_reactions_3support
from semi_beam.ui.numeric_delegate import NullableFloatDelegate, FlexibleDoubleSpinBox


def _set_item(tbl: QTableWidget, r: int, c: int, text: str):
    it = QTableWidgetItem(str(text))
    it.setTextAlignment(Qt.AlignCenter)
    tbl.setItem(r, c, it)


def _get_text(tbl: QTableWidget, r: int, c: int) -> str:
    it = tbl.item(r, c)
    return "" if it is None else (it.text() or "").strip()


def _try_float(text: str) -> Optional[float]:
    t = (text or "").strip().replace(",", ".")
    if t == "":
        return None
    try:
        return float(t)
    except Exception:
        return None


def _fmt_plain(v: Optional[float], decimals: int = 2) -> str:
    if v is None:
        return "-"
    s = f"{float(v):.{decimals}f}"
    if "." in s:
        s = s.rstrip("0").rstrip(".")
    return s


@dataclass(frozen=True)
class ReactionsPlotState:
    beam: Beam
    point_forces: List[PointForce]
    dist_loads: List[DistUniform]
    moments: List[PointMoment]
    note_text: str
    show_vm: bool


class _SearchWorker(QThread):
    progress = Signal(int, str)
    finished_ok = Signal(object)
    failed = Signal(str)

    def __init__(self, config: OptimizerConfig, *, maximize_margin: bool):
        super().__init__()
        self._config = config
        self._maximize_margin = bool(maximize_margin)
        self._cancel_requested = False

    def cancel(self):
        self._cancel_requested = True

    def run(self):
        try:
            solution = search_configuration(
                self._config,
                maximize_margin=self._maximize_margin,
                progress_callback=lambda pct, msg: self.progress.emit(int(pct), str(msg)),
                is_cancelled=lambda: bool(self._cancel_requested),
            )
        except Exception as exc:
            self.failed.emit(str(exc))
            return
        self.finished_ok.emit(solution)


class SemiTrailerReactionsTab(QWidget):
    plot_data_changed = Signal()

    COL_TYPE = 0
    COL_MAG = 1
    COL_POS = 2
    COL_LEN = 3

    LOAD_TYPES = ["Puntual", "Distribuida", "Momento"]

    def __init__(self, parent=None):
        super().__init__(parent)
        self._plot_state: Optional[ReactionsPlotState] = None
        self._last_result: Optional[ReactionsResult] = None
        self._search_worker: Optional[_SearchWorker] = None

        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(10)

        geo_box = QGroupBox("Geometría / apoyos")
        geo_form = QGridLayout(geo_box)
        geo_form.setColumnStretch(1, 1)
        geo_form.setColumnStretch(3, 1)

        self.mode = QComboBox()
        self.mode.addItems(["2 apoyos", "3 apoyos"])
        self.L = self._make_spin(minv=1.0, maxv=1e7, decimals=1, step=100.0, value=13600.0)
        self.x_a = self._make_spin(minv=0.0, maxv=1e7, decimals=1, step=50.0, value=1500.0)
        self.x_b = self._make_spin(minv=0.0, maxv=1e7, decimals=1, step=50.0, value=10500.0)
        self.x_k = self._make_spin(minv=0.0, maxv=1e7, decimals=1, step=50.0, value=1500.0)
        self.x_t = self._make_spin(minv=0.0, maxv=1e7, decimals=1, step=50.0, value=10500.0)
        self.x_t_min = self._make_spin(minv=0.0, maxv=1e7, decimals=1, step=50.0, value=5200.0)
        self.x_t_max = self._make_spin(minv=0.0, maxv=1e7, decimals=1, step=50.0, value=13100.0)
        self.offset = self._make_spin(minv=3075.0, maxv=4000.0, decimals=0, step=25.0, value=3075.0)
        self.offset_slider = QSlider(Qt.Horizontal)
        self.offset_slider.setRange(3075, 4000)
        self.offset_slider.setSingleStep(25)
        self.offset_slider.setPageStep(25)
        self.offset_slider.setValue(3075)
        self.lbl_x_d = QLabel("-")
        self.limit_ra = self._make_spin(minv=0.0, maxv=1e7, decimals=1, step=100.0, value=9000.0)
        self.limit_rb = self._make_spin(minv=0.0, maxv=1e7, decimals=1, step=100.0, value=15800.0)
        self.limit_rk = self._make_spin(minv=0.0, maxv=1e7, decimals=1, step=100.0, value=15000.0)
        self.limit_rd = self._make_spin(minv=0.0, maxv=1e7, decimals=1, step=100.0, value=9200.0)
        self.limit_rt = self._make_spin(minv=0.0, maxv=1e7, decimals=1, step=100.0, value=22200.0)

        self._lbl_x_a = QLabel("x_A [mm]:")
        self._lbl_x_b = QLabel("x_B [mm]:")
        self._lbl_x_k = QLabel("x_k [mm]:")
        self._lbl_x_t = QLabel("x_t [mm]:")
        self._lbl_offset = QLabel("offset [mm]:")
        self._lbl_x_d = QLabel("x_d = x_t - offset [mm]:")
        self._lbl_xt_min = QLabel("x_t_min [mm]:")
        self._lbl_xt_max = QLabel("x_t_max [mm]:")
        self._lbl_limit_ra = QLabel("R_A_max [kg]:")
        self._lbl_limit_rb = QLabel("R_B_max [kg]:")
        self._lbl_limit_rk = QLabel("R_k_max [kg]:")
        self._lbl_limit_rd = QLabel("R_d_max [kg]:")
        self._lbl_limit_rt = QLabel("R_t_max [kg]:")

        offset_host = QWidget()
        offset_lay = QVBoxLayout(offset_host)
        offset_lay.setContentsMargins(0, 0, 0, 0)
        offset_lay.setSpacing(4)
        offset_lay.addWidget(self.offset)
        offset_lay.addWidget(self.offset_slider)

        geo_form.addWidget(QLabel("Largo L [mm]:"), 0, 0)
        geo_form.addWidget(self.L, 0, 1)
        geo_form.addWidget(QLabel("Configuración:"), 0, 2)
        geo_form.addWidget(self.mode, 0, 3)
        geo_form.addWidget(self._lbl_x_a, 1, 0)
        geo_form.addWidget(self.x_a, 1, 1)
        geo_form.addWidget(self._lbl_x_b, 1, 2)
        geo_form.addWidget(self.x_b, 1, 3)
        geo_form.addWidget(self._lbl_x_k, 2, 0)
        geo_form.addWidget(self.x_k, 2, 1)
        geo_form.addWidget(self._lbl_x_t, 2, 2)
        geo_form.addWidget(self.x_t, 2, 3)
        geo_form.addWidget(self._lbl_offset, 3, 0)
        geo_form.addWidget(offset_host, 3, 1)
        geo_form.addWidget(self._lbl_x_d, 3, 2)
        geo_form.addWidget(self.lbl_x_d, 3, 3)
        geo_form.addWidget(self._lbl_xt_min, 4, 0)
        geo_form.addWidget(self.x_t_min, 4, 1)
        geo_form.addWidget(self._lbl_xt_max, 4, 2)
        geo_form.addWidget(self.x_t_max, 4, 3)
        geo_form.addWidget(self._lbl_limit_ra, 5, 0)
        geo_form.addWidget(self.limit_ra, 5, 1)
        geo_form.addWidget(self._lbl_limit_rb, 5, 2)
        geo_form.addWidget(self.limit_rb, 5, 3)
        geo_form.addWidget(self._lbl_limit_rk, 6, 0)
        geo_form.addWidget(self.limit_rk, 6, 1)
        geo_form.addWidget(self._lbl_limit_rd, 6, 2)
        geo_form.addWidget(self.limit_rd, 6, 3)
        geo_form.addWidget(self._lbl_limit_rt, 7, 0)
        geo_form.addWidget(self.limit_rt, 7, 1)

        root.addWidget(geo_box)

        load_box = QGroupBox("Cargas")
        load_lay = QVBoxLayout(load_box)
        load_lay.addWidget(QLabel("Magnitud: kg para puntuales y distribuidas (P total), kg·mm para momentos."))
        self.tbl = QTableWidget(0, 4)
        self.tbl.setHorizontalHeaderLabels(["Tipo", "Magnitud", "Posición / centro [mm]", "Longitud [mm]"])
        self.tbl.horizontalHeader().setStretchLastSection(True)
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
        self.btn_add = QPushButton("Agregar carga")
        self.btn_del = QPushButton("Eliminar seleccionadas")
        load_btns.addWidget(self.btn_add)
        load_btns.addWidget(self.btn_del)
        load_btns.addStretch(1)
        load_lay.addLayout(load_btns)
        root.addWidget(load_box)

        result_box = QGroupBox("Resultados")
        result_lay = QVBoxLayout(result_box)
        self.lbl_r1 = QLabel("R_A: -")
        self.lbl_r2 = QLabel("R_B: -")
        self.lbl_r3 = QLabel("R_t: -")
        for lbl in (self.lbl_r1, self.lbl_r2, self.lbl_r3):
            lbl.setStyleSheet("font-size: 16px; font-weight: 600;")
            result_lay.addWidget(lbl)
        self.lbl_residuals = QLabel("Residuales: -")
        self.lbl_geometry = QLabel("Geometría: -")
        self.chk_show_vm = QCheckBox("Mostrar V(x) y M(x) además del FBD")
        self.chk_show_vm.setChecked(True)
        result_lay.addWidget(self.lbl_residuals)
        result_lay.addWidget(self.lbl_geometry)
        result_lay.addWidget(self.chk_show_vm)
        root.addWidget(result_box)

        search_box = QGroupBox("Búsqueda")
        search_lay = QVBoxLayout(search_box)
        search_row = QHBoxLayout()
        self.btn_search = QPushButton("Buscar (cumplir límites)")
        self.btn_search_best = QPushButton("Buscar mejor margen")
        self.btn_cancel_search = QPushButton("Cancelar búsqueda")
        self.btn_cancel_search.setEnabled(False)
        search_row.addWidget(self.btn_search)
        search_row.addWidget(self.btn_search_best)
        search_row.addWidget(self.btn_cancel_search)
        search_lay.addLayout(search_row)
        self.progress = QProgressBar()
        self.progress.setRange(0, 100)
        self.progress.setValue(0)
        self.lbl_progress = QLabel("Búsqueda inactiva.")
        search_lay.addWidget(self.progress)
        search_lay.addWidget(self.lbl_progress)
        root.addWidget(search_box)

        notes_box = QGroupBox("Notas")
        notes_lay = QVBoxLayout(notes_box)
        self.note_label = QLabel("(sin notas)")
        self.note_label.setWordWrap(True)
        self.note_label.setTextInteractionFlags(Qt.TextSelectableByMouse | Qt.TextSelectableByKeyboard)
        notes_lay.addWidget(self.note_label)
        root.addWidget(notes_box)
        root.addStretch(1)

        self._timer = QTimer(self)
        self._timer.setSingleShot(True)
        self._timer.timeout.connect(self.recompute_now)

        self.mode.currentIndexChanged.connect(self._on_mode_changed)
        self.offset.valueChanged.connect(self._sync_offset_slider)
        self.offset_slider.valueChanged.connect(self._sync_offset_spin)
        self.chk_show_vm.toggled.connect(lambda *_: self.plot_data_changed.emit())
        for sp in (
            self.L, self.x_a, self.x_b, self.x_k, self.x_t, self.x_t_min, self.x_t_max,
            self.limit_ra, self.limit_rb, self.limit_rk, self.limit_rd, self.limit_rt,
        ):
            sp.editingFinished.connect(self._schedule_recompute)
            sp.valueChanged.connect(lambda *_: self._update_geometry_labels())

        self.btn_add.clicked.connect(self._add_load_row)
        self.btn_del.clicked.connect(self._remove_selected_rows)
        self.tbl.cellChanged.connect(lambda *_: self._schedule_recompute())
        self.btn_search.clicked.connect(lambda: self._start_search(maximize_margin=False))
        self.btn_search_best.clicked.connect(lambda: self._start_search(maximize_margin=True))
        self.btn_cancel_search.clicked.connect(self._cancel_search)

        self._add_load_row(load_type="Puntual")
        self._add_load_row(load_type="Distribuida")
        self._on_mode_changed()
        self._schedule_recompute()

    def _make_spin(self, *, minv: float, maxv: float, decimals: int, step: float, value: float) -> FlexibleDoubleSpinBox:
        sp = FlexibleDoubleSpinBox()
        sp.setRange(float(minv), float(maxv))
        sp.setDecimals(int(decimals))
        sp.setSingleStep(float(step))
        sp.setKeyboardTracking(False)
        sp.setAlignment(Qt.AlignCenter)
        sp.setSpecialValueText("")
        sp.setValue(float(value))
        return sp

    def current_plot_state(self) -> Optional[ReactionsPlotState]:
        return self._plot_state

    def _sync_offset_slider(self, value: float):
        iv = int(round(float(value)))
        if self.offset_slider.value() != iv:
            self.offset_slider.blockSignals(True)
            self.offset_slider.setValue(iv)
            self.offset_slider.blockSignals(False)
        self._update_geometry_labels()
        self._schedule_recompute()

    def _sync_offset_spin(self, value: int):
        fv = float(value)
        if abs(self.offset.value() - fv) > 1e-9:
            self.offset.blockSignals(True)
            self.offset.setValue(fv)
            self.offset.blockSignals(False)
        self._update_geometry_labels()
        self._schedule_recompute()

    def _set_mode_widgets_visible(self, widgets: Sequence[QWidget], visible: bool):
        for w in widgets:
            w.setVisible(bool(visible))

    def _on_mode_changed(self):
        is_three = self.mode.currentIndex() == 1
        self._set_mode_widgets_visible(
            [self._lbl_x_a, self.x_a, self._lbl_x_b, self.x_b, self._lbl_limit_ra, self.limit_ra, self._lbl_limit_rb, self.limit_rb],
            not is_three,
        )
        self._set_mode_widgets_visible(
            [
                self._lbl_x_k, self.x_k, self._lbl_x_t, self.x_t, self._lbl_offset,
                self.offset, self.offset_slider, self._lbl_x_d, self.lbl_x_d,
                self._lbl_xt_min, self.x_t_min, self._lbl_xt_max, self.x_t_max,
                self._lbl_limit_rk, self.limit_rk, self._lbl_limit_rd, self.limit_rd,
                self._lbl_limit_rt, self.limit_rt,
            ],
            is_three,
        )
        if is_three:
            self._apply_default_xt_range()
        self._update_geometry_labels()
        self._schedule_recompute()

    def _apply_default_xt_range(self):
        L = float(self.L.value())
        xk = float(self.x_k.value())
        off = float(self.offset.value())
        xt_min = min(max(xk + off + 500.0, 0.0), max(0.0, L - 500.0))
        xt_max = max(xt_min, L - 500.0)
        self.x_t_min.blockSignals(True)
        self.x_t_max.blockSignals(True)
        self.x_t_min.setValue(xt_min)
        self.x_t_max.setValue(xt_max)
        self.x_t_min.blockSignals(False)
        self.x_t_max.blockSignals(False)

    def _update_geometry_labels(self):
        if self.mode.currentIndex() == 1:
            xd = float(self.x_t.value()) - float(self.offset.value())
            self.lbl_x_d.setText(_fmt_plain(xd, 1))
            self.lbl_geometry.setText(
                f"Geometría: x_k={_fmt_plain(self.x_k.value(), 1)} mm, "
                f"x_d={_fmt_plain(xd, 1)} mm, x_t={_fmt_plain(self.x_t.value(), 1)} mm"
            )
        else:
            self.lbl_x_d.setText("-")
            self.lbl_geometry.setText(
                f"Geometría: x_A={_fmt_plain(self.x_a.value(), 1)} mm, x_B={_fmt_plain(self.x_b.value(), 1)} mm"
            )

    def _schedule_recompute(self):
        self._timer.start(180)

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

    def _set_item_editable(self, row: int, col: int, editable: bool):
        it = self.tbl.item(row, col)
        if it is None:
            _set_item(self.tbl, row, col, "")
            it = self.tbl.item(row, col)
        if it is None:
            return
        flags = it.flags()
        if editable:
            it.setFlags(flags | Qt.ItemIsEditable)
        else:
            it.setFlags(flags & ~Qt.ItemIsEditable)

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
        self._schedule_recompute()

    def _add_load_row(self, *, load_type: str = "Puntual"):
        row = self.tbl.rowCount()
        self.tbl.insertRow(row)
        cmb = QComboBox()
        cmb.addItems(self.LOAD_TYPES)
        cmb.setCurrentText(load_type if load_type in self.LOAD_TYPES else "Puntual")
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
        self._schedule_recompute()

    def _row_error(self, row: int, has_error: bool):
        brush = QBrush(QColor(255, 210, 210) if has_error else QColor(255, 255, 255))
        for col in range(1, self.tbl.columnCount()):
            it = self.tbl.item(row, col)
            if it is None:
                _set_item(self.tbl, row, col, "")
                it = self.tbl.item(row, col)
            if it is not None:
                it.setBackground(brush)
        combo = self.tbl.cellWidget(row, self.COL_TYPE)
        if combo is not None:
            combo.setStyleSheet("QComboBox { background-color: %s; }" % ("#FFD2D2" if has_error else "white"))

    def _build_loads(self) -> Tuple[List[ReactionLoad], List[str], List[int]]:
        loads: List[ReactionLoad] = []
        errors: List[str] = []
        dist_rows: List[int] = []
        dist_intervals: List[Tuple[float, float]] = []
        L = float(self.L.value())
        p_count = 0
        d_count = 0
        m_count = 0

        for row in range(self.tbl.rowCount()):
            self._row_error(row, False)
            load_type = self._type_combo(row).currentText()
            mag = _try_float(_get_text(self.tbl, row, self.COL_MAG))
            pos = _try_float(_get_text(self.tbl, row, self.COL_POS))
            length = _try_float(_get_text(self.tbl, row, self.COL_LEN))
            row_has_error = False

            if mag is None or pos is None:
                errors.append(f"Fila {row + 1}: complete magnitud y posición.")
                row_has_error = True
            elif load_type == "Puntual":
                if not (0.0 <= pos <= L):
                    errors.append(f"Fila {row + 1}: la carga puntual debe estar dentro de [0, L].")
                    row_has_error = True
                else:
                    p_count += 1
                    loads.append(PointForce(label=f"P{p_count}", x_mm=float(pos), value_user=float(mag)))
            elif load_type == "Momento":
                if not (0.0 <= pos <= L):
                    errors.append(f"Fila {row + 1}: el momento puntual debe estar dentro de [0, L].")
                    row_has_error = True
                else:
                    m_count += 1
                    loads.append(PointMoment(label=f"M{m_count}", x_mm=float(pos), M_user_kgmm=float(mag)))
            else:
                if length is None or length <= 0.0:
                    errors.append(f"Fila {row + 1}: la distribuida requiere longitud > 0.")
                    row_has_error = True
                else:
                    try:
                        x1, x2 = dist_interval(float(pos), float(length))
                    except Exception as exc:
                        errors.append(f"Fila {row + 1}: {exc}")
                        row_has_error = True
                    else:
                        if x1 < 0.0 or x2 > L:
                            errors.append(f"Fila {row + 1}: el tramo distribuido debe quedar dentro de [0, L].")
                            row_has_error = True
                        else:
                            d_count += 1
                            dist_rows.append(row)
                            dist_intervals.append((x1, x2))
                            loads.append(
                                DistUniform(
                                    label=f"q{d_count}",
                                    x0_mm=x1,
                                    Lq_mm=float(length),
                                    q_user=float(mag) / float(length),
                                )
                            )

            if row_has_error:
                self._row_error(row, True)

        ok, pairs = check_no_overlap(dist_intervals)
        if not ok:
            for idx_i, idx_j in pairs:
                if idx_i < len(dist_rows):
                    self._row_error(dist_rows[idx_i], True)
                if idx_j < len(dist_rows):
                    self._row_error(dist_rows[idx_j], True)
            errors.append("Las cargas distribuidas no pueden solaparse entre sí.")

        return loads, errors, dist_rows

    def _current_limits(self) -> Dict[str, float]:
        if self.mode.currentIndex() == 0:
            return {
                "R_A": float(self.limit_ra.value()),
                "R_B": float(self.limit_rb.value()),
            }
        return {
            "R_k": float(self.limit_rk.value()),
            "R_d": float(self.limit_rd.value()),
            "R_t": float(self.limit_rt.value()),
        }

    def _validate_geometry(self) -> List[str]:
        errors: List[str] = []
        L = float(self.L.value())
        if L <= 0.0:
            errors.append("L debe ser mayor a 0.")
        if self.mode.currentIndex() == 0:
            xa = float(self.x_a.value())
            xb = float(self.x_b.value())
            if not (0.0 <= xa < xb <= L):
                errors.append("En 2 apoyos se requiere 0 <= x_A < x_B <= L.")
        else:
            xk = float(self.x_k.value())
            xt = float(self.x_t.value())
            off = float(self.offset.value())
            xd = xt - off
            xt_min = float(self.x_t_min.value())
            xt_max = float(self.x_t_max.value())
            if not (0.0 <= xk < xd < xt <= L):
                errors.append("En 3 apoyos se requiere 0 <= x_k < x_d < x_t <= L.")
            if xt_min > xt_max:
                errors.append("Debe cumplirse x_t_min <= x_t_max.")
            if not (xt_min <= xt <= xt_max):
                errors.append("x_t actual debe quedar dentro de [x_t_min, x_t_max].")
        return errors

    def recompute_now(self):
        if self.tbl.state() == QAbstractItemView.State.EditingState:
            self._timer.start(180)
            return
        geometry_errors = self._validate_geometry()
        loads, load_errors, _ = self._build_loads()
        errors = geometry_errors + load_errors
        if errors:
            self._plot_state = None
            self._last_result = None
            self.lbl_r1.setText("R_A: -")
            self.lbl_r2.setText("R_B: -")
            self.lbl_r3.setText("R_t: -")
            self.lbl_residuals.setText("Residuales: -")
            self.note_label.setText("\n".join(f"- {msg}" for msg in errors))
            self.plot_data_changed.emit()
            return

        try:
            if self.mode.currentIndex() == 0:
                result = solve_reactions_2support(
                    float(self.L.value()),
                    (float(self.x_a.value()), float(self.x_b.value())),
                    loads,
                )
                state = self._build_plot_state_2support(loads, result)
            else:
                result = solve_reactions_3support(
                    float(self.L.value()),
                    float(self.x_k.value()),
                    float(self.x_t.value()) - float(self.offset.value()),
                    float(self.x_t.value()),
                    loads,
                )
                state = self._build_plot_state_3support(loads, result)
        except Exception as exc:
            self._plot_state = None
            self._last_result = None
            self.note_label.setText(f"Error: {exc}")
            self.plot_data_changed.emit()
            return

        self._plot_state = state
        self._last_result = result
        self._update_result_labels(result)
        note_lines = [
            f"Residual ΣFy = {_fmt_plain(result.Fy_total_residual, 6)}",
            f"Residual ΣM0 = {_fmt_plain(result.M0_residual, 6)}",
        ]
        note_lines.extend(result.notes)
        self.note_label.setText("\n".join(note_lines))
        self.lbl_residuals.setText(
            f"Residuales: ΣFy={_fmt_plain(result.Fy_total_residual, 6)} | ΣM0={_fmt_plain(result.M0_residual, 6)}"
        )
        self.plot_data_changed.emit()

    def _build_plot_state_2support(self, loads: Sequence[ReactionLoad], result: ReactionsResult) -> ReactionsPlotState:
        L = float(self.L.value())
        points = [load for load in loads if isinstance(load, PointForce)]
        dists = [load for load in loads if isinstance(load, DistUniform)]
        moms = [load for load in loads if isinstance(load, PointMoment)]
        points = list(points) + [
            PointForce(label="RA", x_mm=float(self.x_a.value()), value_user=float(result.reacciones["R_A"])),
            PointForce(label="RB", x_mm=float(self.x_b.value()), value_user=float(result.reacciones["R_B"])),
        ]
        return ReactionsPlotState(
            beam=Beam(L_mm=L),
            point_forces=points,
            dist_loads=list(dists),
            moments=list(moms),
            note_text=self.note_label.text(),
            show_vm=bool(self.chk_show_vm.isChecked()),
        )

    def _build_plot_state_3support(self, loads: Sequence[ReactionLoad], result: ReactionsResult) -> ReactionsPlotState:
        L = float(self.L.value())
        xk = float(self.x_k.value())
        xt = float(self.x_t.value())
        xd = xt - float(self.offset.value())
        points = [load for load in loads if isinstance(load, PointForce)]
        dists = [load for load in loads if isinstance(load, DistUniform)]
        moms = [load for load in loads if isinstance(load, PointMoment)]
        points = list(points) + [
            PointForce(label="Rk", x_mm=xk, value_user=float(result.reacciones["R_k"])),
            PointForce(label="Rd", x_mm=xd, value_user=float(result.reacciones["R_d"])),
            PointForce(label="Rt", x_mm=xt, value_user=float(result.reacciones["R_t"])),
        ]
        return ReactionsPlotState(
            beam=Beam(L_mm=L),
            point_forces=points,
            dist_loads=list(dists),
            moments=list(moms),
            note_text=self.note_label.text(),
            show_vm=bool(self.chk_show_vm.isChecked()),
        )

    def _set_result_style(self, lbl: QLabel, name: str, value: float, limit: float):
        pct = (float(value) / float(limit) * 100.0) if limit > 0.0 else 0.0
        exceeded = limit > 0.0 and value > limit + 1e-9
        lbl.setText(f"{name}: {_fmt_plain(value, 2)} kg | {_fmt_plain(pct, 1)}% del límite")
        lbl.setStyleSheet("font-size: 16px; font-weight: 600; color: %s;" % ("#B00020" if exceeded else "#1F1F1F"))

    def _update_result_labels(self, result: ReactionsResult):
        limits = self._current_limits()
        if self.mode.currentIndex() == 0:
            self._set_result_style(self.lbl_r1, "R_A", float(result.reacciones["R_A"]), float(limits["R_A"]))
            self._set_result_style(self.lbl_r2, "R_B", float(result.reacciones["R_B"]), float(limits["R_B"]))
            self.lbl_r3.setVisible(False)
        else:
            self._set_result_style(self.lbl_r1, "R_k", float(result.reacciones["R_k"]), float(limits["R_k"]))
            self._set_result_style(self.lbl_r2, "R_d", float(result.reacciones["R_d"]), float(limits["R_d"]))
            self._set_result_style(self.lbl_r3, "R_t", float(result.reacciones["R_t"]), float(limits["R_t"]))
            self.lbl_r3.setVisible(True)

    def _build_optimizer_config(self) -> OptimizerConfig:
        loads, errors, _ = self._build_loads()
        geometry_errors = self._validate_geometry()
        if errors or geometry_errors:
            raise ValueError("\n".join(geometry_errors + errors))
        limits = self._current_limits()
        return OptimizerConfig(
            L_mm=float(self.L.value()),
            support_mode="2" if self.mode.currentIndex() == 0 else "3",
            x_a_mm=float(self.x_a.value()) if self.mode.currentIndex() == 0 else None,
            x_b_mm=float(self.x_b.value()) if self.mode.currentIndex() == 0 else None,
            x_k_mm=float(self.x_k.value()) if self.mode.currentIndex() == 1 else None,
            x_t_mm=float(self.x_t.value()) if self.mode.currentIndex() == 1 else None,
            offset_mm=float(self.offset.value()) if self.mode.currentIndex() == 1 else None,
            x_t_min_mm=float(self.x_t_min.value()) if self.mode.currentIndex() == 1 else None,
            x_t_max_mm=float(self.x_t_max.value()) if self.mode.currentIndex() == 1 else None,
            support_limits=limits,
            loads=loads,
        )

    def _start_search(self, *, maximize_margin: bool):
        if self._search_worker is not None and self._search_worker.isRunning():
            QMessageBox.warning(self, "Búsqueda", "Ya hay una búsqueda en curso.")
            return
        try:
            config = self._build_optimizer_config()
        except Exception as exc:
            QMessageBox.warning(self, "Búsqueda", str(exc))
            return

        self.progress.setValue(0)
        self.lbl_progress.setText("Iniciando búsqueda...")
        self.btn_cancel_search.setEnabled(True)
        self.btn_search.setEnabled(False)
        self.btn_search_best.setEnabled(False)

        worker = _SearchWorker(config, maximize_margin=maximize_margin)
        worker.progress.connect(self._on_search_progress)
        worker.finished_ok.connect(self._on_search_finished)
        worker.failed.connect(self._on_search_failed)
        worker.finished.connect(self._on_search_cleanup)
        self._search_worker = worker
        worker.start()

    def _cancel_search(self):
        if self._search_worker is not None:
            self._search_worker.cancel()

    def _on_search_progress(self, pct: int, text: str):
        self.progress.setValue(int(pct))
        self.lbl_progress.setText(text)

    def _on_search_finished(self, solution_obj):
        if not isinstance(solution_obj, OptimizerSolution):
            self._on_search_failed("La búsqueda devolvió un resultado inválido.")
            return
        solution = solution_obj
        self._apply_optimizer_solution(solution)
        self.progress.setValue(100)
        self.lbl_progress.setText("Búsqueda finalizada.")

    def _on_search_failed(self, message: str):
        self.lbl_progress.setText(f"Búsqueda detenida: {message}")

    def _on_search_cleanup(self):
        self.btn_cancel_search.setEnabled(False)
        self.btn_search.setEnabled(True)
        self.btn_search_best.setEnabled(True)
        self._search_worker = None

    def _apply_optimizer_solution(self, solution: OptimizerSolution):
        self.tbl.blockSignals(True)
        point_loads = [ld for ld in solution.loads if isinstance(ld, PointForce)]
        dist_loads = [ld for ld in solution.loads if isinstance(ld, DistUniform)]
        moment_loads = [ld for ld in solution.loads if isinstance(ld, PointMoment)]
        point_idx = 0
        dist_idx = 0
        moment_idx = 0
        for row in range(self.tbl.rowCount()):
            load_type = self._type_combo(row).currentText()
            if load_type == "Puntual":
                if point_idx >= len(point_loads):
                    continue
                load = point_loads[point_idx]
                point_idx += 1
                _set_item(self.tbl, row, self.COL_MAG, _fmt_plain(load.value_user, 2))
                _set_item(self.tbl, row, self.COL_POS, _fmt_plain(load.x_mm, 2))
            elif load_type == "Distribuida":
                if dist_idx >= len(dist_loads):
                    continue
                load = dist_loads[dist_idx]
                dist_idx += 1
                center = float(load.x0_mm) + 0.5 * float(load.Lq_mm)
                total_p = float(load.q_user) * float(load.Lq_mm)
                _set_item(self.tbl, row, self.COL_MAG, _fmt_plain(total_p, 2))
                _set_item(self.tbl, row, self.COL_POS, _fmt_plain(center, 2))
                _set_item(self.tbl, row, self.COL_LEN, _fmt_plain(load.Lq_mm, 2))
            else:
                if moment_idx >= len(moment_loads):
                    continue
                load = moment_loads[moment_idx]
                moment_idx += 1
                _set_item(self.tbl, row, self.COL_MAG, _fmt_plain(load.M_user_kgmm, 2))
                _set_item(self.tbl, row, self.COL_POS, _fmt_plain(load.x_mm, 2))
        self.tbl.blockSignals(False)

        if self.mode.currentIndex() == 1 and solution.x_t_mm is not None and solution.offset_mm is not None:
            self.x_t.setValue(float(solution.x_t_mm))
            self.offset.setValue(float(solution.offset_mm))

        self.recompute_now()
        if solution.notes:
            self.note_label.setText("\n".join(solution.notes))
