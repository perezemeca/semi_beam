"""
FBD Viga Isostática (Desktop Windows) — PySide6 + Matplotlib
Unidades: kg (kgf), cm, kg·cm

Requisitos:
  pip install pyside6 matplotlib numpy

Ejecutar:
  python fbd_desktop.py

Empaquetar a .exe (Windows):
  pip install pyinstaller
  pyinstaller --onefile --noconsole --name FBD_Viga fbd_desktop.py
  (si hiciera falta) pyinstaller --onefile --noconsole --name FBD_Viga --collect-all matplotlib fbd_desktop.py

Convenciones (internas):
  - Fuerza interna Fy > 0 => hacia arriba
  - Distribuida interna w_up > 0 => hacia arriba
  - Momento interno M > 0 => CCW (sagging +)

Convenciones (usuario) por etiqueta (labels):
  Rp1 > 0 (up)    -> se dibuja abajo con flecha hacia arriba
  Rp2 > 0 (down)  -> se dibuja arriba con flecha hacia abajo
  Rt  > 0 (up)
  Rd  > 0 (up)
  q   > 0 (down)  -> arriba con flechas hacia abajo
  Pn  > 0 (down)  -> arriba con flechas hacia abajo (P1, P2, P3, ...)

Regla especial:
  - Para fuerzas puntuales tipo Pn con x<0 o x>L:
    se interpreta como un MOMENTO en el borde (x=0 o x=L), usando el equivalente:
      M = (x - x_edge) * Fy_internal
    (y NO se agrega la fuerza dentro de la viga).
"""

from __future__ import annotations

import sys
import re
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict

import numpy as np

from PySide6.QtCore import Qt, QTimer
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QGroupBox,
    QLabel, QDoubleSpinBox, QPushButton, QTableWidget, QTableWidgetItem,
    QSizePolicy, QFormLayout, QSplitter
)

import matplotlib
matplotlib.use("QtAgg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.patches import Rectangle, Arc, FancyArrowPatch


# -----------------------------
# Configuración de convenciones
# -----------------------------
LABEL_SIGN_CONVENTION: Dict[str, str] = {
    "Rp1": "up_positive",
    "Rp2": "down_positive",
    "Rt":  "up_positive",
    "Rd":  "up_positive",
    "P":   "down_positive",
    "q":   "down_positive",
}

# Para "P" (incluye P1, P2, ...) fuera de [0,L] -> momento en borde
OUTSIDE_AS_EDGE_MOMENT_LABELS = {"P"}

P_INDEX_RE = re.compile(r"^P(\d+)$", re.IGNORECASE)


def is_indexed_P(label: str) -> bool:
    return bool(P_INDEX_RE.match(label.strip()))


def p_index(label: str) -> Optional[int]:
    m = P_INDEX_RE.match(label.strip())
    if not m:
        return None
    return int(m.group(1))


def label_kind(label: str) -> str:
    """
    Normaliza el label para aplicar convenciones:
      - P1, P2, ... => 'P'
    """
    l = (label or "").strip()
    if l.upper() == "P" or is_indexed_P(l):
        return "P"
    return l


def next_free_p_index(used: set[int]) -> int:
    k = 1
    while k in used:
        k += 1
    return k


def to_internal_Fy(label: str, value_user: float) -> float:
    """
    Convierte el valor ingresado por el usuario a fuerza interna Fy (+up).
    """
    kind = label_kind(label)
    conv = LABEL_SIGN_CONVENTION.get(kind, "down_positive")
    if conv == "up_positive":
        return float(value_user)           # + => up
    return -float(value_user)              # + => down => Fy negativo


def to_internal_w_up(label: str, q_user: float) -> float:
    """
    Distribuida uniforme ingresada por usuario. Para q: positivo hacia abajo.
    Interno w_up = -q_user (si convención es down_positive).
    """
    kind = label_kind(label)
    conv = LABEL_SIGN_CONVENTION.get(kind, "down_positive")
    if conv == "up_positive":
        return float(q_user)
    return -float(q_user)


# -----------------------------
# Modelo de datos
# -----------------------------
@dataclass(frozen=True)
class PointForce:
    label: str
    x_cm: float
    value_user: float


@dataclass(frozen=True)
class DistUniform:
    label: str
    x0_cm: float
    Lq_cm: float
    q_user: float


@dataclass(frozen=True)
class PointMoment:
    label: str
    x_cm: float
    M_user_kgcm: float  # + CCW (sagging+)


@dataclass(frozen=True)
class Beam:
    L_cm: float


@dataclass(frozen=True)
class NormalizedPointForce:
    label: str
    x_cm: float
    Fy_internal: float
    value_user: float


@dataclass(frozen=True)
class NormalizedDistUniform:
    label: str
    x1_cm: float
    x2_cm: float
    w_up_internal: float
    q_user: float


@dataclass(frozen=True)
class NormalizedPointMoment:
    label: str
    x_cm: float
    M_internal: float
    M_user_kgcm: float


@dataclass
class FBDData:
    beam: Beam
    point_forces: List[NormalizedPointForce]
    dist_loads: List[NormalizedDistUniform]
    moments: List[NormalizedPointMoment]
    notes: List[str]


# -----------------------------
# Normalización / Validación
# -----------------------------
def _clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def normalize_inputs(
    beam: Beam,
    point_forces: List[PointForce],
    dist_loads: List[DistUniform],
    moments: List[PointMoment],
) -> FBDData:
    L = float(beam.L_cm)
    notes: List[str] = []

    n_points: List[NormalizedPointForce] = []
    n_dists: List[NormalizedDistUniform] = []
    n_moms: List[NormalizedPointMoment] = []

    # 1) Fuerzas puntuales
    for pf in point_forces:
        label = (pf.label or "P").strip()
        x = float(pf.x_cm)
        val = float(pf.value_user)
        Fy = to_internal_Fy(label, val)

        if x < 0.0 or x > L:
            if label_kind(label) in OUTSIDE_AS_EDGE_MOMENT_LABELS:
                x_edge = 0.0 if x < 0.0 else L
                M = (x - x_edge) * Fy  # CCW+ interno
                n_moms.append(NormalizedPointMoment(
                    label=f"{label}_equiv",
                    x_cm=x_edge,
                    M_internal=M,
                    M_user_kgcm=M,
                ))
                notes.append(f'P fuera de rango (label="{label}", x={x:g}): -> momento en x={x_edge:g}, M={M:g} kg·cm.')
            else:
                x_clamped = 0.0 if x < 0.0 else L
                n_points.append(NormalizedPointForce(label=label, x_cm=x_clamped, Fy_internal=Fy, value_user=val))
                notes.append(f'Fuerza fuera de rango (label="{label}", x={x:g}): -> x={x_clamped:g} manteniendo fuerza.')
        else:
            n_points.append(NormalizedPointForce(label=label, x_cm=x, Fy_internal=Fy, value_user=val))

    # 2) Distribuidas uniformes: recorte a [0, L]
    for dl in dist_loads:
        label = (dl.label or "q").strip()
        x0 = float(dl.x0_cm)
        Lq = float(dl.Lq_cm)
        q_user = float(dl.q_user)

        if Lq <= 0:
            notes.append(f'Distribuida inválida (label="{label}"): Lq<=0 (Lq={Lq:g}). Se ignoró.')
            continue

        x1 = x0
        x2 = x0 + Lq

        x1c = _clamp(x1, 0.0, L)
        x2c = _clamp(x2, 0.0, L)
        if x2c <= x1c + 1e-9:
            notes.append(f'Distribuida fuera de la viga (label="{label}", [{x1:g},{x2:g}]): se ignoró.')
            continue

        w_up = to_internal_w_up(label, q_user)
        n_dists.append(NormalizedDistUniform(
            label=label,
            x1_cm=x1c,
            x2_cm=x2c,
            w_up_internal=w_up,
            q_user=q_user
        ))

        if abs(x1c - x1) > 1e-9 or abs(x2c - x2) > 1e-9:
            notes.append(f'Distribuida recortada (label="{label}") de [{x1:g},{x2:g}] a [{x1c:g},{x2c:g}].')

    # 3) Momentos puntuales: clamped a [0, L] para dibujo
    for pm in moments:
        label = (pm.label or "M").strip()
        x = float(pm.x_cm)
        M_user = float(pm.M_user_kgcm)
        M_internal = M_user

        if x < 0.0 or x > L:
            x_clamped = 0.0 if x < 0.0 else L
            notes.append(f'Momento fuera de rango (label="{label}", x={x:g}): se movió a x={x_clamped:g}.')
            x = x_clamped

        n_moms.append(NormalizedPointMoment(
            label=label,
            x_cm=x,
            M_internal=M_internal,
            M_user_kgcm=M_user
        ))

    return FBDData(
        beam=beam,
        point_forces=n_points,
        dist_loads=n_dists,
        moments=n_moms,
        notes=notes
    )


# -----------------------------
# Render (solo dibujo)
# -----------------------------
@dataclass(frozen=True)
class RenderStyle:
    beam_lw: float = 3.5

    arrow_lw: float = 1.0
    arrow_scale: float = 11.0

    dist_rect_lw: float = 0.9
    dist_rect_alpha: float = 0.12

    moment_arc_lw: float = 1.0
    moment_arrow_lw: float = 1.0
    moment_arrow_scale: float = 11.0

    # Alturas (fijas, ajustables por UI)
    arrow_height_pctL: float = 12.0
    dist_height_pctL: float = 12.0
    moment_radius_pctL: float = 8.0

    moment_delta_deg: float = 18.0
    font_size: int = 10


def _draw_arrow(ax, x: float, y0: float, y1: float, style: RenderStyle):
    ax.annotate(
        "",
        xy=(x, y1),
        xytext=(x, y0),
        arrowprops=dict(
            arrowstyle="-|>",
            lw=style.arrow_lw,
            mutation_scale=style.arrow_scale,
            color="red",
            facecolor="red",
            shrinkA=0,
            shrinkB=0,
        ),
    )


def _draw_moment(ax, x: float, M_internal: float, r: float, style: RenderStyle):
    """
    Arco largo fijo + flecha sobre el arco (sentido según signo de M).
    Evita bugs de Arc cuando theta2 < theta1.
    """
    theta1, theta2 = 30.0, 330.0  # arco estable (largo)

    arc = Arc(
        (x, 0.0),
        width=2.0 * r,
        height=2.0 * r,
        angle=0,
        theta1=theta1,
        theta2=theta2,
        lw=style.moment_arc_lw,
        color="red",
    )
    ax.add_patch(arc)

    delta = float(style.moment_delta_deg)

    if M_internal >= 0:
        a_start = np.deg2rad(theta2 - delta)
        a_end = np.deg2rad(theta2)
    else:
        a_start = np.deg2rad(theta1 + delta)
        a_end = np.deg2rad(theta1)

    start = (x + r * np.cos(a_start), 0.0 + r * np.sin(a_start))
    end = (x + r * np.cos(a_end), 0.0 + r * np.sin(a_end))

    arrow = FancyArrowPatch(
        posA=start,
        posB=end,
        arrowstyle="-|>",
        mutation_scale=style.moment_arrow_scale,
        lw=style.moment_arrow_lw,
        color="red",
        shrinkA=0,
        shrinkB=0,
    )
    ax.add_patch(arrow)


def render_fbd(ax, data: FBDData, style: RenderStyle):
    L = float(data.beam.L_cm)
    ax.clear()

    # Alturas fijas en función de L
    arrow_h = (style.arrow_height_pctL / 100.0) * L
    dist_h = (style.dist_height_pctL / 100.0) * L
    moment_r = (style.moment_radius_pctL / 100.0) * L
    moment_r = max(moment_r, 0.04 * L)

    # Viga (línea azul)
    ax.plot([0, L], [0, 0], linewidth=style.beam_lw, color="blue")

    # Distribuidas uniformes
    for dl in data.dist_loads:
        x1, x2 = dl.x1_cm, dl.x2_cm
        w_up = dl.w_up_internal
        q_user = dl.q_user
        label = dl.label

        if w_up < 0:
            # carga hacia abajo => arriba con flechas hacia abajo
            rect = Rectangle(
                (x1, 0.0),
                x2 - x1,
                dist_h,
                facecolor="red",
                alpha=style.dist_rect_alpha,
                edgecolor="red",
                linewidth=style.dist_rect_lw,
            )
            ax.add_patch(rect)

            n_lines = max(3, int((x2 - x1) / (L / 30.0)))
            xs = np.linspace(x1, x2, n_lines)
            for xi in xs:
                _draw_arrow(ax, xi, dist_h, 0.0, style)

            ax.text((x1 + x2) / 2.0, dist_h + 0.04 * L, f"{label}={q_user:g} kg/cm",
                    ha="center", va="bottom", fontsize=style.font_size, color="red")
        else:
            # carga hacia arriba => abajo con flechas hacia arriba
            rect = Rectangle(
                (x1, -dist_h),
                x2 - x1,
                dist_h,
                facecolor="red",
                alpha=style.dist_rect_alpha,
                edgecolor="red",
                linewidth=style.dist_rect_lw,
            )
            ax.add_patch(rect)

            n_lines = max(3, int((x2 - x1) / (L / 30.0)))
            xs = np.linspace(x1, x2, n_lines)
            for xi in xs:
                _draw_arrow(ax, xi, -dist_h, 0.0, style)

            ax.text((x1 + x2) / 2.0, -dist_h - 0.04 * L, f"{label}={q_user:g} kg/cm",
                    ha="center", va="top", fontsize=style.font_size, color="red")

    # Fuerzas puntuales (incluye P1,P2,... y reacciones)
    for pf in data.point_forces:
        x = pf.x_cm
        Fy = pf.Fy_internal
        label = pf.label
        val_user = pf.value_user

        if Fy < 0:
            _draw_arrow(ax, x, arrow_h, 0.0, style)
            ax.text(x, arrow_h + 0.02 * L, f"{label}={val_user:g} kg",
                    ha="center", va="bottom", fontsize=style.font_size, color="red")
        else:
            _draw_arrow(ax, x, -arrow_h, 0.0, style)
            ax.text(x, -arrow_h - 0.02 * L, f"{label}={val_user:g} kg",
                    ha="center", va="top", fontsize=style.font_size, color="red")

    # Momentos puntuales
    for pm in data.moments:
        x = pm.x_cm
        M = pm.M_internal
        label = pm.label
        M_user = pm.M_user_kgcm

        _draw_moment(ax, x, M, moment_r, style)

        ax.text(x + 1.1 * moment_r, 1.1 * moment_r, f"{label}={M_user:g} kg·cm",
                ha="left", va="bottom", fontsize=style.font_size, color="red")

    # Límites y proporción
    max_y = max(arrow_h, dist_h, moment_r) * 1.9
    max_y = max(max_y, 0.35 * L)

    ax.set_xlim(-0.05 * L, 1.05 * L)
    ax.set_ylim(-max_y, max_y)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x [cm]")
    ax.set_yticks([])
    ax.set_title("Diagrama de Cuerpo Libre (FBD)")
    ax.grid(True, axis="x", alpha=0.25)


# -----------------------------
# UI (PySide6)
# -----------------------------
class FBDApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("FBD Viga Isostática (kg, cm)")
        self.resize(1300, 720)

        root = QWidget()
        self.setCentralWidget(root)
        main = QHBoxLayout(root)
        main.setContentsMargins(8, 8, 8, 8)

        splitter = QSplitter(Qt.Horizontal)
        splitter.setChildrenCollapsible(False)  # evita que se “colapse” un panel
        splitter.setHandleWidth(8)              # grosor del divisor (más fácil de agarrar)
        main.addWidget(splitter)

        # Panel izquierdo: Datos
        left_widget = QWidget()
        left_widget.setMinimumWidth(320)        # ajustá según tu gusto / tamaño de texto
        left = QVBoxLayout(left_widget)
        left.setContentsMargins(0, 0, 0, 0)

        # Panel derecho: Gráfico
        right_widget = QWidget()
        right_widget.setMinimumWidth(520)       # opcional, para que no quede ridículamente chico
        right = QVBoxLayout(right_widget)
        right.setContentsMargins(0, 0, 0, 0)

        splitter.addWidget(left_widget)
        splitter.addWidget(right_widget)

        # Reparto inicial de tamaños (izq, der)
        splitter.setSizes([460, 840])

        # Que el gráfico “gane” espacio al agrandar la ventana
        splitter.setStretchFactor(0, 0)  # datos
        splitter.setStretchFactor(1, 1)  # gráfico

        # Viga
        gb_beam = QGroupBox("Viga")
        beam_layout = QHBoxLayout(gb_beam)
        beam_layout.addWidget(QLabel("L [cm]:"))
        self.L_spin = QDoubleSpinBox()
        self.L_spin.setRange(1.0, 1e9)
        self.L_spin.setValue(1200.0)
        self.L_spin.setDecimals(2)
        self.L_spin.setSingleStep(10.0)
        beam_layout.addWidget(self.L_spin)
        beam_layout.addStretch(1)
        left.addWidget(gb_beam)

        # Escala visual (compacta)
        gb_view = QGroupBox("Escala visual (solo gráfico)")
        form = QFormLayout(gb_view)
        form.setLabelAlignment(Qt.AlignLeft)
        form.setFormAlignment(Qt.AlignTop)
        form.setHorizontalSpacing(10)
        form.setVerticalSpacing(6)

        self.arrow_pct = QDoubleSpinBox()
        self.arrow_pct.setRange(2.0, 30.0)
        self.arrow_pct.setValue(12.0)
        self.arrow_pct.setDecimals(1)
        self.arrow_pct.setSingleStep(0.5)

        self.dist_pct = QDoubleSpinBox()
        self.dist_pct.setRange(2.0, 30.0)
        self.dist_pct.setValue(12.0)
        self.dist_pct.setDecimals(1)
        self.dist_pct.setSingleStep(0.5)

        self.moment_pct = QDoubleSpinBox()
        self.moment_pct.setRange(2.0, 20.0)
        self.moment_pct.setValue(8.0)
        self.moment_pct.setDecimals(1)
        self.moment_pct.setSingleStep(0.5)

        form.addRow("Alt. puntuales [%L]:", self.arrow_pct)
        form.addRow("Alt. distribuida [%L]:", self.dist_pct)
        form.addRow("Radio momento [%L]:", self.moment_pct)

        left.addWidget(gb_view)

        # Tabla: fuerzas puntuales (autolabel P1,P2,...)
        self._build_points_table(left)

        # Tabla: distribuidas
        self.tbl_dists = self._make_table(
            title="Cargas distribuidas uniformes (label típico: q)",
            columns=["label", "x0_cm", "Lq_cm", "q"],
            add_label="Agregar distribuida",
            remove_label="Eliminar seleccionadas",
            parent_layout=left,
            default_row=("q", "", "", ""),
        )

        # Tabla: momentos
        self.tbl_moments = self._make_table(
            title="Momentos puntuales (M en kg·cm, CCW+)",
            columns=["label", "x_cm", "M_kgcm"],
            add_label="Agregar momento",
            remove_label="Eliminar seleccionadas",
            parent_layout=left,
            default_row=("M", "", ""),
        )

        # Notas
        self.note_label = QLabel(
            'Convenciones por label:\n'
            '  Rp1,Rt,Rd: positivo = UP (se dibuja abajo ↑)\n'
            '  Rp2,q,Pn:  positivo = DOWN (se dibuja arriba ↓)\n'
            'Regla: Pn fuera de [0,L] -> se convierte a momento en borde.'
        )
        self.note_label.setWordWrap(True)
        left.addWidget(self.note_label)

        left.addStretch(1)

        # Plot
        self.fig = plt.Figure()
        self.canvas = FigureCanvas(self.fig)
        self.canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.ax = self.fig.add_subplot(111)
        right.addWidget(self.canvas)

        # Debounce redraw
        self._redraw_timer = QTimer(self)
        self._redraw_timer.setSingleShot(True)
        self._redraw_timer.timeout.connect(self._redraw_now)

        # Signals
        self.L_spin.valueChanged.connect(self.schedule_redraw)
        self.arrow_pct.valueChanged.connect(self.schedule_redraw)
        self.dist_pct.valueChanged.connect(self.schedule_redraw)
        self.moment_pct.valueChanged.connect(self.schedule_redraw)

        self.tbl_points.itemChanged.connect(self.schedule_redraw)
        self.tbl_dists.itemChanged.connect(self.schedule_redraw)
        self.tbl_moments.itemChanged.connect(self.schedule_redraw)

        self._redraw_now()

    def schedule_redraw(self, *args):
        self._redraw_timer.start(80)

    # -----------------------------
    # Tabla de puntuales con P1,P2,P3...
    # -----------------------------
    def _build_points_table(self, parent_layout: QVBoxLayout):
        gb = QGroupBox('Fuerzas puntuales (P1,P2,..., Rp1, Rp2, Rt, Rd)')
        v = QVBoxLayout(gb)

        self.tbl_points = QTableWidget(0, 3)
        self.tbl_points.setHorizontalHeaderLabels(["label", "x_cm", "valor"])
        self.tbl_points.horizontalHeader().setStretchLastSection(True)
        v.addWidget(self.tbl_points)

        btns = QHBoxLayout()
        btn_add = QPushButton("Agregar fuerza")
        btn_rem = QPushButton("Eliminar seleccionadas")
        btns.addWidget(btn_add)
        btns.addWidget(btn_rem)
        btns.addStretch(1)
        v.addLayout(btns)

        btn_add.clicked.connect(self._add_point_row)
        btn_rem.clicked.connect(self._remove_selected_point_rows)

        parent_layout.addWidget(gb)

    def _next_p_label_for_table(self) -> str:
        used: set[int] = set()
        for r in range(self.tbl_points.rowCount()):
            it = self.tbl_points.item(r, 0)
            if not it:
                continue
            idx = p_index(it.text().strip().upper())
            if idx is not None:
                used.add(idx)
        k = next_free_p_index(used)
        return f"P{k}"

    def _add_point_row(self):
        r = self.tbl_points.rowCount()
        self.tbl_points.insertRow(r)

        label = self._next_p_label_for_table()  # P1, P2, ...
        it0 = QTableWidgetItem(label)
        it0.setTextAlignment(Qt.AlignCenter)
        self.tbl_points.setItem(r, 0, it0)

        it1 = QTableWidgetItem("")
        it1.setTextAlignment(Qt.AlignCenter)
        self.tbl_points.setItem(r, 1, it1)

        it2 = QTableWidgetItem("")
        it2.setTextAlignment(Qt.AlignCenter)
        self.tbl_points.setItem(r, 2, it2)

    def _remove_selected_point_rows(self):
        rows = sorted(set([i.row() for i in self.tbl_points.selectedItems()]), reverse=True)
        for rr in rows:
            self.tbl_points.removeRow(rr)

    # -----------------------------
    # Tablas genéricas
    # -----------------------------
    def _make_table(self, title, columns, add_label, remove_label, parent_layout, default_row):
        gb = QGroupBox(title)
        v = QVBoxLayout(gb)

        tbl = QTableWidget(0, len(columns))
        tbl.setHorizontalHeaderLabels(columns)
        tbl.horizontalHeader().setStretchLastSection(True)
        v.addWidget(tbl)

        btns = QHBoxLayout()
        btn_add = QPushButton(add_label)
        btn_rem = QPushButton(remove_label)
        btns.addWidget(btn_add)
        btns.addWidget(btn_rem)
        btns.addStretch(1)
        v.addLayout(btns)

        def add_row():
            r = tbl.rowCount()
            tbl.insertRow(r)
            for c, val in enumerate(default_row):
                it = QTableWidgetItem(str(val))
                it.setTextAlignment(Qt.AlignCenter)
                tbl.setItem(r, c, it)

        def remove_rows():
            rows = sorted(set([i.row() for i in tbl.selectedItems()]), reverse=True)
            for r in rows:
                tbl.removeRow(r)

        btn_add.clicked.connect(add_row)
        btn_rem.clicked.connect(remove_rows)

        parent_layout.addWidget(gb)
        return tbl

    # -----------------------------
    # Parsing tablas
    # -----------------------------
    def _read_row(self, tbl: QTableWidget, r: int) -> List[str]:
        vals: List[str] = []
        for c in range(tbl.columnCount()):
            it = tbl.item(r, c)
            vals.append(it.text().strip() if it else "")
        return vals

    def _try_float(self, s: str) -> Optional[float]:
        if s is None:
            return None
        t = s.strip()
        if t == "":
            return None
        try:
            return float(t.replace(",", "."))
        except ValueError:
            return None

    def _parse_point_forces(self) -> List[PointForce]:
        """
        Regla:
          - Las filas con label vacío se ignoran (porque el botón ya crea Pn).
          - Si alguien cambia a 'P' sin número, se renumera a un índice libre (P#).
          - Si alguien escribe P2 manualmente, se respeta y se evita duplicar.
        """
        raw: List[Tuple[str, float, float]] = []
        used_idx: set[int] = set()

        for r in range(self.tbl_points.rowCount()):
            label_s, x_s, v_s = self._read_row(self.tbl_points, r)
            label = (label_s or "").strip()
            x = self._try_float(x_s)
            v = self._try_float(v_s)
            if x is None or v is None:
                continue

            if label == "":
                label = "P"  # por si el usuario lo borró

            idx = p_index(label.upper())
            if idx is not None:
                used_idx.add(idx)

            raw.append((label, x, v))

        out: List[PointForce] = []
        for label, x, v in raw:
            if label.strip().upper() == "P":
                k = next_free_p_index(used_idx)
                used_idx.add(k)
                label = f"P{k}"
            out.append(PointForce(label=label, x_cm=x, value_user=v))

        return out

    def _parse_dist_loads(self) -> List[DistUniform]:
        out: List[DistUniform] = []
        for r in range(self.tbl_dists.rowCount()):
            label_s, x0_s, Lq_s, q_s = self._read_row(self.tbl_dists, r)
            label = (label_s or "q").strip()
            x0 = self._try_float(x0_s)
            Lq = self._try_float(Lq_s)
            q = self._try_float(q_s)
            if x0 is None or Lq is None or q is None:
                continue
            out.append(DistUniform(label=label, x0_cm=x0, Lq_cm=Lq, q_user=q))
        return out

    def _parse_moments(self) -> List[PointMoment]:
        out: List[PointMoment] = []
        for r in range(self.tbl_moments.rowCount()):
            label_s, x_s, m_s = self._read_row(self.tbl_moments, r)
            label = (label_s or "M").strip()
            x = self._try_float(x_s)
            m = self._try_float(m_s)
            if x is None or m is None:
                continue
            out.append(PointMoment(label=label, x_cm=x, M_user_kgcm=m))
        return out

    # -----------------------------
    # Redraw
    # -----------------------------
    def _redraw_now(self):
        try:
            L = float(self.L_spin.value())
            beam = Beam(L_cm=L)

            pforces = self._parse_point_forces()
            dloads = self._parse_dist_loads()
            moms = self._parse_moments()

            data = normalize_inputs(beam, pforces, dloads, moms)

            style = RenderStyle(
                arrow_height_pctL=float(self.arrow_pct.value()),
                dist_height_pctL=float(self.dist_pct.value()),
                moment_radius_pctL=float(self.moment_pct.value()),
                beam_lw=3.5,
                arrow_lw=1.0,
                arrow_scale=11.0,
                moment_arc_lw=1.0,
                moment_arrow_lw=1.0,
                moment_arrow_scale=11.0,
                dist_rect_lw=0.9,
                dist_rect_alpha=0.12,
                moment_delta_deg=18.0,
                font_size=10
            )

            render_fbd(self.ax, data, style)
            self.fig.tight_layout()
            self.canvas.draw()

            if data.notes:
                self.note_label.setText(
                    "Notas:\n- " + "\n- ".join(data.notes) +
                    "\n\nConvenciones por label:\n"
                    "  Rp1,Rt,Rd: positivo = UP (abajo ↑)\n"
                    "  Rp2,q,Pn:  positivo = DOWN (arriba ↓)\n"
                    "Regla: Pn fuera de [0,L] -> momento en borde."
                )
            else:
                self.note_label.setText(
                    "Convenciones por label:\n"
                    "  Rp1,Rt,Rd: positivo = UP (abajo ↑)\n"
                    "  Rp2,q,Pn:  positivo = DOWN (arriba ↓)\n"
                    "Regla: Pn fuera de [0,L] -> momento en borde."
                )

        except Exception as e:
            self.note_label.setText(f"Error al graficar: {e}")


def main():
    app = QApplication(sys.argv)
    w = FBDApp()
    w.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
