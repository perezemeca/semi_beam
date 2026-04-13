from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional, List, Tuple, Dict, Any

import numpy as np

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QFormLayout, QLabel, QComboBox,
    QTableWidget, QTableWidgetItem, QHeaderView, QCheckBox,
    QDoubleSpinBox, QSizePolicy
)
from PySide6.QtWidgets import QStyledItemDelegate

import matplotlib
matplotlib.use("QtAgg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.patches import Rectangle


INCH_TO_MM = 25.4

# Opciones fijas (UI)
WEB_T_OPTIONS_IN = ["3/16", "1/4", "5/16", "3/8"]
FLANGE_T_OPTIONS_IN = ["1/2", "5/8", "3/4"]

# Geometría fija de la viga doble T (planchuela 5")
BF_MM = 127.0  # ancho planchuela (mm)


def frac_in_to_mm(s: str) -> float:
    s = (s or "").strip()
    if s == "":
        return 0.0
    if "/" in s:
        num, den = s.split("/")
        return (float(num) / float(den)) * INCH_TO_MM
    return float(s) * INCH_TO_MM


def fmt_plain(v: float, decimals: int = 2) -> str:
    s = f"{v:.{decimals}f}"
    if "." in s:
        s = s.rstrip("0").rstrip(".")
    return s


# ----------------------------
# Delegados de edición (solo numérico / combo)
# ----------------------------
class FloatSpinDelegate(QStyledItemDelegate):
    """
    Convierte la celda en QDoubleSpinBox (solo números).
    """
    def __init__(self, parent=None, *, minv=-1e12, maxv=1e12, decimals=2, step=50.0):
        super().__init__(parent)
        self.minv = float(minv)
        self.maxv = float(maxv)
        self.decimals = int(decimals)
        self.step = float(step)

    def createEditor(self, parent, option, index):
        sp = QDoubleSpinBox(parent)
        sp.setRange(self.minv, self.maxv)
        sp.setDecimals(self.decimals)
        sp.setSingleStep(self.step)
        sp.setKeyboardTracking(False)
        return sp

    def setEditorData(self, editor, index):
        try:
            val = float(str(index.data()).replace(",", "."))
        except Exception:
            val = 0.0
        editor.setValue(val)

    def setModelData(self, editor, model, index):
        model.setData(index, fmt_plain(editor.value(), self.decimals))


class ComboDelegate(QStyledItemDelegate):
    def __init__(self, parent=None, options: List[str] = None):
        super().__init__(parent)
        self.options = options or []

    def createEditor(self, parent, option, index):
        cb = QComboBox(parent)
        cb.addItems(self.options)
        return cb

    def setEditorData(self, editor, index):
        txt = str(index.data() or "")
        k = self.options.index(txt) if txt in self.options else 0
        editor.setCurrentIndex(k)

    def setModelData(self, editor, model, index):
        model.setData(index, editor.currentText())


# ----------------------------
# Cálculo de propiedades (sección compuesta por rectángulos)
# ----------------------------
@dataclass
class Rect:
    b: float      # mm
    h: float      # mm
    y0: float     # mm (cota inferior desde referencia)
    count: int = 1


def composite_props(rects: List[Rect]) -> Tuple[float, float, float, float, float]:
    """
    Devuelve:
      A_total [mm2]
      ybar [mm]
      Ix_c [mm4]  respecto eje por ybar
      y_min [mm], y_max [mm] (extremos geométricos)
    """
    if not rects:
        return 0.0, 0.0, 0.0, 0.0, 0.0

    As = []
    yAs = []
    Ix0s = []
    ymins = []
    ymaxs = []

    for r in rects:
        A1 = r.b * r.h
        yc = r.y0 + r.h / 2.0
        Ix_local = (r.b * r.h**3) / 12.0  # en su centro
        Ix_ref = Ix_local + A1 * yc**2    # teorema de ejes paralelos hacia y=0
        As.append(A1 * r.count)
        yAs.append((A1 * r.count) * yc)
        Ix0s.append(Ix_ref * r.count)
        ymins.append(r.y0)
        ymaxs.append(r.y0 + r.h)

    A_tot = float(np.sum(As))
    ybar = float(np.sum(yAs) / A_tot) if A_tot > 0 else 0.0
    Ix0 = float(np.sum(Ix0s))
    Ix_c = Ix0 - A_tot * ybar**2

    return A_tot, ybar, Ix_c, float(np.min(ymins)), float(np.max(ymaxs))


def build_I_section_rects(
    h_web_mm: float,
    t_web_mm: float,
    t_top_mm: float,
    t_bot_mm: float,
    n_beams: int = 2,
) -> Tuple[List[Rect], float]:
    """
    Sección doble T idealizada como 3 rectángulos (2 vigas idénticas):
      - planchuela inferior: b=127, h=t_bot, y0=0
      - alma: b=t_web, h=h_web, y0=t_bot
      - planchuela superior: b=127, h=t_top, y0=t_bot+h_web

    Referencia y=0: cara inferior de planchuela inferior.
    Retorna rects y H_total.
    """
    H = t_bot_mm + h_web_mm + t_top_mm
    rects = [
        Rect(b=BF_MM, h=t_bot_mm, y0=0.0, count=n_beams),
        Rect(b=t_web_mm, h=h_web_mm, y0=t_bot_mm, count=n_beams),
        Rect(b=BF_MM, h=t_top_mm, y0=t_bot_mm + h_web_mm, count=n_beams),
    ]
    return rects, H


# ----------------------------
# Panel UI
# ----------------------------
class SectionCheckPanel(QWidget):
    """
    Verificador a flexión pura.
    - Preview: 1 sola viga con cotas H y h_web.
    - Cálculo: 2 vigas (n_beams=2).
    - Tabla: 8 secciones.
    """
    def __init__(self):
        super().__init__()
        self._M_provider: Optional[Callable[[float], float]] = None  # M en kg·cm
        self._updating = False

        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)

        # ---- Config global (igual “estilo anterior”: simple) ----
        cfg = QWidget()
        form = QFormLayout(cfg)
        form.setRowWrapPolicy(QFormLayout.WrapAllRows)
        form.setFieldGrowthPolicy(QFormLayout.ExpandingFieldsGrow)

        self.cb_Fy = QComboBox()
        self.cb_Fy.addItems(["F36 (Fy=3600 kg/cm²)", "F24 (Fy=2400 kg/cm²)"])
        self.cb_Fy.setCurrentIndex(0)

        self.cb_t_top = QComboBox()
        self.cb_t_top.addItems(FLANGE_T_OPTIONS_IN)
        self.cb_t_top.setCurrentText("5/8")

        self.cb_t_bot = QComboBox()
        self.cb_t_bot.addItems(FLANGE_T_OPTIONS_IN)
        self.cb_t_bot.setCurrentText("3/4")

        self.chk_autoM = QCheckBox("Auto M(x) desde diagrama")
        self.chk_autoM.setChecked(True)

        form.addRow("Material (Fy):", self.cb_Fy)
        form.addRow("t planchuela superior [in]:", self.cb_t_top)
        form.addRow("t planchuela inferior [in]:", self.cb_t_bot)
        form.addRow(self.chk_autoM)

        root.addWidget(cfg)

        # ---- Preview (1 sección) ----
        self.fig = plt.Figure(figsize=(3.4, 2.2))
        self.ax = self.fig.add_subplot(1, 1, 1)
        self.canvas = FigureCanvas(self.fig)
        self.canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.canvas.setMinimumHeight(230)
        root.addWidget(self.canvas)

        # ---- Tabla (8 secciones) ----
        self.tbl = QTableWidget(8, 11)
        self.tbl.setHorizontalHeaderLabels([
            "Sección",
            "x [mm]",
            "M [kg·cm]",
            "h_web [mm]",
            "t_web [in]",
            "FS (=Wd/Wn)",
            "Jx [cm4]",
            "Yc [cm]",
            "Wd [cm3]",
            "Wn [cm3]",
            "OK"
        ])
        self.tbl.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.tbl.verticalHeader().setVisible(False)
        root.addWidget(self.tbl)

        # Delegados para “solo números / combos”
        self.tbl.setItemDelegateForColumn(1, FloatSpinDelegate(self, minv=-1e12, maxv=1e12, decimals=2, step=50.0))   # x
        self.tbl.setItemDelegateForColumn(2, FloatSpinDelegate(self, minv=-1e18, maxv=1e18, decimals=2, step=1000.0))  # M
        self.tbl.setItemDelegateForColumn(3, FloatSpinDelegate(self, minv=1.0, maxv=5000.0, decimals=1, step=10.0))    # h_web
        self.tbl.setItemDelegateForColumn(4, ComboDelegate(self, WEB_T_OPTIONS_IN))                                     # t_web

        # Inicializar filas
        self._init_rows()

        # Señales (autorefresco)
        self.cb_Fy.currentIndexChanged.connect(self.refresh_all)
        self.cb_t_top.currentIndexChanged.connect(self.refresh_all)
        self.cb_t_bot.currentIndexChanged.connect(self.refresh_all)
        self.chk_autoM.toggled.connect(self.refresh_all)

        self.tbl.itemChanged.connect(self._on_table_changed)
        self.tbl.itemSelectionChanged.connect(self._update_preview_from_selection)

        # Primer refresco
        self.refresh_all()

    # -------- API para main_window --------
    def set_moment_provider(self, func: Callable[[float], float]):
        """func(x_mm) -> M [kg·cm]"""
        self._M_provider = func
        self.refresh_all()

    def on_equilibrium_changed(self):
        """Cuando cambian diagramas/entradas, recalcular todo."""
        self.refresh_all()

    def serialize_state(self) -> Dict[str, Any]:
        rows = []
        for r in range(self.tbl.rowCount()):
            rows.append([self._cell_text(r, c) for c in range(self.tbl.columnCount())])
        return {
            "Fy_idx": self.cb_Fy.currentIndex(),
            "t_top": self.cb_t_top.currentText(),
            "t_bot": self.cb_t_bot.currentText(),
            "autoM": self.chk_autoM.isChecked(),
            "rows": rows,
        }

    def restore_state(self, st: Dict[str, Any]):
        if not st:
            return
        self._updating = True
        try:
            self.cb_Fy.setCurrentIndex(int(st.get("Fy_idx", 0)))
            self.cb_t_top.setCurrentText(str(st.get("t_top", "5/8")))
            self.cb_t_bot.setCurrentText(str(st.get("t_bot", "3/4")))
            self.chk_autoM.setChecked(bool(st.get("autoM", True)))

            rows = st.get("rows", [])
            if len(rows) == 8:
                for r in range(8):
                    for c in range(min(len(rows[r]), self.tbl.columnCount())):
                        self._set_cell(r, c, rows[r][c], editable=(c in {1, 2, 3, 4}))
        finally:
            self._updating = False
        self.refresh_all()

    # -------- Internos --------
    def _Fy_value(self) -> float:
        return 3600.0 if self.cb_Fy.currentIndex() == 0 else 2400.0  # kg/cm2

    def _cell_text(self, r: int, c: int) -> str:
        it = self.tbl.item(r, c)
        return it.text() if it else ""

    def _set_cell(self, r: int, c: int, txt: str, editable: bool = True):
        it = self.tbl.item(r, c)
        if it is None:
            it = QTableWidgetItem("")
            it.setTextAlignment(Qt.AlignCenter)
            self.tbl.setItem(r, c, it)

        it.setText(str(txt))
        flags = it.flags()
        if editable:
            it.setFlags(flags | Qt.ItemIsEditable)
        else:
            it.setFlags(flags & ~Qt.ItemIsEditable)

    def _init_rows(self):
        # Columnas editables: x(1), M(2), h_web(3), t_web(4)
        for r in range(8):
            self._set_cell(r, 0, str(r + 1), editable=False)
            self._set_cell(r, 1, "0", editable=True)        # x
            self._set_cell(r, 2, "0", editable=True)        # M (editable, pero si autoM ON, se sobreescribe)
            self._set_cell(r, 3, "130", editable=True)      # h_web
            self._set_cell(r, 4, "5/16", editable=True)     # t_web (combo)
            # calculadas
            for c in range(5, 11):
                self._set_cell(r, c, "", editable=False)

        # Selección inicial
        self.tbl.setCurrentCell(0, 1)

    def _on_table_changed(self, it: QTableWidgetItem):
        if self._updating:
            return
        self.refresh_all()

    def refresh_all(self):
        if self._updating:
            return
        self._updating = True
        try:
            # Auto M(x) por fila si hay provider
            if self._M_provider is not None and self.chk_autoM.isChecked():
                for r in range(8):
                    try:
                        x = float(self._cell_text(r, 1).replace(",", "."))
                    except Exception:
                        continue
                    M = float(self._M_provider(x))  # kg·cm
                    self._set_cell(r, 2, fmt_plain(M, 2), editable=True)

            # Recalcular cada fila
            for r in range(8):
                self._recalc_row(r)

            # Preview de la selección
            self._update_preview_from_selection()

        finally:
            self._updating = False

    def _recalc_row(self, r: int):
        # Leer inputs
        try:
            M_kgcm = float(self._cell_text(r, 2).replace(",", "."))
        except Exception:
            M_kgcm = 0.0

        try:
            h_web = float(self._cell_text(r, 3).replace(",", "."))
        except Exception:
            h_web = 130.0

        t_web_in = (self._cell_text(r, 4) or "5/16").strip()
        t_web = frac_in_to_mm(t_web_in)

        t_top = frac_in_to_mm(self.cb_t_top.currentText())
        t_bot = frac_in_to_mm(self.cb_t_bot.currentText())

        rects, H = build_I_section_rects(
            h_web_mm=h_web,
            t_web_mm=t_web,
            t_top_mm=t_top,
            t_bot_mm=t_bot,
            n_beams=2,  # cálculo para 2 vigas
        )
        A, ybar, Ix_c, y_min, y_max = composite_props(rects)

        # fibras extremas
        y_top = (y_max - ybar)
        y_bot = (ybar - y_min)
        y_ext = max(y_top, y_bot)
        if y_ext <= 1e-9:
            y_ext = 1.0

        Wd_mm3 = Ix_c / y_ext

        # Unidades a cm
        Jx_cm4 = Ix_c / 1e4
        Yc_cm = ybar / 10.0
        Wd_cm3 = Wd_mm3 / 1e3

        Fy = self._Fy_value()  # kg/cm2
        Wn_cm3 = (M_kgcm / Fy) if abs(Fy) > 1e-12 else 0.0

        FS = (Wd_cm3 / Wn_cm3) if abs(Wn_cm3) > 1e-12 else 0.0

        # Guardar calculadas
        self._set_cell(r, 5, fmt_plain(FS, 2), editable=False)
        self._set_cell(r, 6, fmt_plain(Jx_cm4, 2), editable=False)
        self._set_cell(r, 7, fmt_plain(Yc_cm, 2), editable=False)
        self._set_cell(r, 8, fmt_plain(Wd_cm3, 2), editable=False)
        self._set_cell(r, 9, fmt_plain(Wn_cm3, 2), editable=False)

        ok = (FS >= 2.9)
        self._set_cell(r, 10, "OK" if ok else "NO", editable=False)

        # Colorear FS + OK
        fs_item = self.tbl.item(r, 5)
        ok_item = self.tbl.item(r, 10)
        if fs_item:
            fs_item.setBackground(Qt.green if ok else Qt.red)
        if ok_item:
            ok_item.setBackground(Qt.green if ok else Qt.red)

    def _update_preview_from_selection(self):
        r = self.tbl.currentRow()
        if r < 0:
            r = 0

        try:
            h_web = float(self._cell_text(r, 3).replace(",", "."))
        except Exception:
            h_web = 130.0

        t_web_in = (self._cell_text(r, 4) or "5/16").strip()
        t_web = frac_in_to_mm(t_web_in)

        t_top = frac_in_to_mm(self.cb_t_top.currentText())
        t_bot = frac_in_to_mm(self.cb_t_bot.currentText())

        rects_1, H = build_I_section_rects(
            h_web_mm=h_web,
            t_web_mm=t_web,
            t_top_mm=t_top,
            t_bot_mm=t_bot,
            n_beams=1,  # preview 1 viga
        )

        from matplotlib.patches import FancyArrowPatch

        self.ax.clear()

        # Dibujar la sección (1 viga)
        def w_vis(b_mm: float) -> float:
            return max(30.0, min(220.0, b_mm / 1.6))  # un poco más “ancha” para leer mejor

        # rectángulos de la sección (preview 1 viga)
        rects_1, H = build_I_section_rects(
            h_web_mm=h_web,
            t_web_mm=t_web,
            t_top_mm=t_top,
            t_bot_mm=t_bot,
            n_beams=1,
        )

        x_section_half = 0.0
        max_w = 0.0
        for rr in rects_1:
            w = w_vis(rr.b)
            max_w = max(max_w, w)
            x_left = -w / 2.0
            self.ax.add_patch(Rectangle((x_left, rr.y0), w, rr.h, fill=False, linewidth=2.0, edgecolor="black"))

        # --- Helpers de acotación ---
        def draw_dim_vertical(x_dim: float, y1: float, y2: float, text: str, side: str):
            """
            Cota vertical tipo dibujo técnico:
            - línea de cota con flechas abiertas
            - texto centrado
            - líneas auxiliares desde la pieza hacia la cota
            """
            # flechas abiertas y línea
            arr = FancyArrowPatch(
                (x_dim, y1), (x_dim, y2),
                arrowstyle="<->", mutation_scale=12,
                lw=1.6, color="blue"
            )
            self.ax.add_patch(arr)

            # texto centrado (interrumpe visualmente la línea porque se superpone)
            self.ax.text(x_dim, (y1 + y2) / 2.0, text, ha="center", va="center",
                        fontsize=12, color="blue",
                        bbox=dict(facecolor="white", edgecolor="none", pad=0.6))

            # líneas auxiliares (extension lines)
            # salen desde el borde del perfil hacia la línea de cota
            x_edge = (max_w / 2.0) if side == "right" else (-max_w / 2.0)
            gap = 8.0  # separación chica para que no toque el perfil
            x0 = x_edge + gap if side == "right" else x_edge - gap

            self.ax.plot([x0, x_dim], [y1, y1], lw=1.2, color="blue")
            self.ax.plot([x0, x_dim], [y2, y2], lw=1.2, color="blue")


        # Vista y límites
        y_min = -max(25.0, 0.25 * H)
        y_max = H + max(45.0, 0.35 * H)

        self.ax.set_xlim(-max_w - 120, max_w + 120)
        self.ax.set_ylim(y_min, y_max)
        self.ax.set_aspect("auto")
        self.ax.axis("off")

        # ---- COTAS ----
        # H total a la izquierda
        x_dim_L = -max_w/2.0 - 70.0
        draw_dim_vertical(
            x_dim=x_dim_L,
            y1=0.0,
            y2=H,
            text=f"H={fmt_plain(H,0)} mm",
            side="left"
        )

        # h_web a la derecha (entre caras internas de planchuelas)
        x_dim_R = +max_w/2.0 + 70.0
        y1 = t_bot
        y2 = t_bot + h_web
        draw_dim_vertical(
            x_dim=x_dim_R,
            y1=y1,
            y2=y2,
            text=f"h_web={fmt_plain(h_web,0)} mm",
            side="right"
        )

        self.canvas.draw()

