from __future__ import annotations

import math
from typing import Optional, Tuple, List, Dict

import numpy as np
from matplotlib.patches import Rectangle, FancyArrowPatch, PathPatch
from matplotlib.path import Path
from matplotlib.transforms import IdentityTransform

from semi_beam.domain.cases import FBDData
from semi_beam.view.style import RenderStyle


# -------------------------
# Helpers generales
# -------------------------
def _ceil_mm(x: float) -> int:
    return int(math.ceil(float(x)))


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


# -------------------------
# Momento circular (en pixeles)
# -------------------------
def _draw_moment_px(ax, x: float, M_internal: float, r_px: float, style: RenderStyle):
    """
    Arco de momento SIEMPRE circular (en pixeles), radio fijo r_px.
    IMPORTANTE: llamar SOLO después de set_xlim/set_ylim/set_aspect.
    """
    cx, cy = ax.transData.transform((x, 0.0))

    theta1, theta2 = 30.0, 330.0
    n = 120
    ang = np.deg2rad(np.linspace(theta1, theta2, n))
    xs = cx + r_px * np.cos(ang)
    ys = cy + r_px * np.sin(ang)

    verts = np.column_stack([xs, ys])
    codes = np.full(n, Path.LINETO, dtype=int)
    codes[0] = Path.MOVETO
    path = Path(verts, codes)

    arc_patch = PathPatch(
        path,
        fill=False,
        lw=style.moment_arc_lw,
        edgecolor="red",
        transform=IdentityTransform(),
        zorder=10,
        clip_on=True,
    )
    arc_patch.set_clip_path(ax.patch)
    ax.add_patch(arc_patch)

    delta = float(style.moment_delta_deg)
    if M_internal >= 0:
        a0 = np.deg2rad(theta2 - delta)
        a1 = np.deg2rad(theta2)
    else:
        a0 = np.deg2rad(theta1 + delta)
        a1 = np.deg2rad(theta1)

    start = (cx + r_px * np.cos(a0), cy + r_px * np.sin(a0))
    end = (cx + r_px * np.cos(a1), cy + r_px * np.sin(a1))

    arrow = FancyArrowPatch(
        posA=start,
        posB=end,
        arrowstyle="-|>",
        mutation_scale=style.moment_arrow_scale,
        lw=style.moment_arrow_lw,
        color="red",
        shrinkA=0,
        shrinkB=0,
        transform=IdentityTransform(),
        zorder=11,
        clip_on=True,
    )
    arrow.set_clip_path(ax.patch)
    ax.add_patch(arrow)

    # label sugerido (pix->data)
    lx_pix = cx - 0.9 * r_px
    ly_pix = cy + 1.1 * r_px
    lx, ly = ax.transData.inverted().transform((lx_pix, ly_pix))
    return float(lx), float(ly)


# -------------------------
# Acotación (CAD-like)
# -------------------------
def _find_carrozable_end_mm(data: FBDData) -> float:
    """
    Inferir L carrozable:
    - si existe una distribuida label 'q' que arranca en 0, usa su x2.
    - si no, usa el máximo x2 de distribuidas que arrancan en 0
    - si no, fallback: L total de viga
    """
    cand: List[float] = []
    for dl in data.dist_loads:
        x1 = float(dl.x1_mm)
        x2 = float(dl.x2_mm)
        if abs(x1 - 0.0) < 1e-9:
            if (dl.label or "").strip().lower() == "q":
                return x2
            cand.append(x2)
    if cand:
        return max(cand)
    return float(data.beam.L_mm)


def _collect_dimension_targets(data: FBDData) -> Dict[str, List[float]]:
    """
    Posiciones a acotar por lado (arriba/abajo) según dónde se dibuja la acción:
    - Puntuales: Fy_internal < 0 => arriba ; Fy_internal > 0 => abajo
    - Distribuidas: w_up_internal < 0 => arriba ; else => abajo
    """
    top: List[float] = []
    bottom: List[float] = []

    for pf in data.point_forces:
        x = float(pf.x_mm)
        if pf.Fy_internal < 0:
            top.append(x)
        else:
            bottom.append(x)

    for dl in data.dist_loads:
        x1 = float(dl.x1_mm)
        x2 = float(dl.x2_mm)
        if dl.w_up_internal < 0:
            top.extend([x1, x2])
        else:
            bottom.extend([x1, x2])

    def _clean(xs: List[float]) -> List[float]:
        return sorted(set([x for x in xs if abs(x) > 1e-9]))

    return {"top": _clean(top), "bottom": _clean(bottom)}


def _draw_dimension_from_ref(
    ax,
    *,
    x_ref_real: float,
    x_real: float,
    y_dim: float,
    text: str,
    gap_obj_mm: float,
    color: str = "blue",
):
    lw = 0.9
    x0 = float(x_ref_real)
    x1 = float(x_real)
    y = float(y_dim)

    sgny = 1.0 if y >= 0 else -1.0

    # Auxiliares: gap con viga/flechas rojas, pero llegan hasta y (línea de cota)
    y_start = 0.0 + sgny * float(gap_obj_mm)
    y_end = y

    ax.plot([x0, x0], [y_start, y_end], lw=lw, color=color)
    ax.plot([x1, x1], [y_start, y_end], lw=lw, color=color)

    # Centro y span
    xm = 0.5 * (x0 + x1)
    span = abs(x1 - x0)
    if span < 1e-6:
        return

    # ---------- hueco centrado basado en ancho real del texto ----------
    # Aseguramos renderer disponible
    fig = ax.figure
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()

    # Creamos el texto "dummy" para medir (NO visible)
    t = ax.text(
        xm, y, text,
        ha="center", va="center",
        fontsize=9,
        color=color,
        alpha=0.0,   # invisible
        zorder=0
    )
    bb = t.get_window_extent(renderer=renderer)
    t.remove()

    text_w_px = bb.width

    # Convertimos px -> unidades de datos en X
    p0 = ax.transData.inverted().transform((0.0, 0.0))
    p1 = ax.transData.inverted().transform((text_w_px, 0.0))
    text_w_data = abs(p1[0] - p0[0])

    # Margen alrededor del texto (en px) y convertido a datos
    margin_px = 14.0  # ajustable (más chico = hueco más chico)
    m0 = ax.transData.inverted().transform((0.0, 0.0))
    m1 = ax.transData.inverted().transform((margin_px, 0.0))
    margin_data = abs(m1[0] - m0[0])

    gap_mid = text_w_data + 2.0 * margin_data

    # límites de seguridad: no más de 35% del span, no menos de 2% del span
    gap_mid = min(gap_mid, 0.35 * span)
    gap_mid = max(gap_mid, 0.02 * span)

    xL = xm - 0.5 * gap_mid
    xR = xm + 0.5 * gap_mid
    # ---------------------------------------------------------------

    # segmentos de línea de cota
    ax.plot([x0, xL], [y, y], lw=lw, color=color)
    ax.plot([xR, x1], [y, y], lw=lw, color=color)

    # Flechas huecas
    arrL = FancyArrowPatch(
        posA=(xL, y), posB=(x0, y),
        arrowstyle="->",
        mutation_scale=12.0,
        lw=lw,
        edgecolor=color,
        facecolor="none",
        shrinkA=0,
        shrinkB=0,
        zorder=15,
    )
    ax.add_patch(arrL)

    arrR = FancyArrowPatch(
        posA=(xR, y), posB=(x1, y),
        arrowstyle="->",
        mutation_scale=12.0,
        lw=lw,
        edgecolor=color,
        facecolor="none",
        shrinkA=0,
        shrinkB=0,
        zorder=15,
    )
    ax.add_patch(arrR)

    # Texto centrado exacto sobre la línea
    ax.text(
        xm,
        y,
        text,
        ha="center",
        va="center",
        fontsize=9,
        color=color,
        bbox=dict(facecolor="white", edgecolor="none", pad=0.9),
        zorder=20,
    )


# -------------------------
# Render principal
# -------------------------
def render_fbd(
    ax,
    data: FBDData,
    style: RenderStyle,
    y_zoom: float = 1.0,
    xlim: Optional[Tuple[float, float]] = None,
):
    L_total = float(data.beam.L_mm)
    L_car = float(_find_carrozable_end_mm(data))

    ax.clear()

    arrow_h = (style.arrow_height_pctL / 100.0) * L_total
    dist_h = (style.dist_height_pctL / 100.0) * L_total

    # Viga
    ax.plot([0, L_total], [0, 0], linewidth=style.beam_lw, color="blue")

    # Distribuidas
    for dl in data.dist_loads:
        x1, x2 = float(dl.x1_mm), float(dl.x2_mm)
        w_up = dl.w_up_internal
        q_user = dl.q_user
        label = dl.label

        if w_up < 0:  # down => arriba
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

            n_lines = max(3, int((x2 - x1) / (max(L_total, 1.0) / 30.0)))
            xs = np.linspace(x1, x2, n_lines)
            for xi in xs:
                _draw_arrow(ax, float(xi), dist_h, 0.0, style)

            ax.text(
                (x1 + x2) / 2.0,
                dist_h + 0.04 * L_total,
                f"{label}={q_user:g} kg/mm",
                ha="center",
                va="bottom",
                fontsize=style.font_size,
                color="red",
            )
        else:  # up => abajo
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

            n_lines = max(3, int((x2 - x1) / (max(L_total, 1.0) / 30.0)))
            xs = np.linspace(x1, x2, n_lines)
            for xi in xs:
                _draw_arrow(ax, float(xi), -dist_h, 0.0, style)

            ax.text(
                (x1 + x2) / 2.0,
                -dist_h - 0.04 * L_total,
                f"{label}={q_user:g} kg/mm",
                ha="center",
                va="top",
                fontsize=style.font_size,
                color="red",
            )

    # Puntuales
    for pf in data.point_forces:
        x = float(pf.x_mm)
        Fy = pf.Fy_internal
        label = pf.label
        val_user = pf.value_user

        if Fy < 0:  # down => arriba
            _draw_arrow(ax, x, arrow_h, 0.0, style)
            ax.text(
                x,
                arrow_h + 0.02 * L_total,
                f"{label}={val_user:g} kg",
                ha="center",
                va="bottom",
                fontsize=style.font_size,
                color="red",
            )
        else:  # up => abajo
            _draw_arrow(ax, x, -arrow_h, 0.0, style)
            ax.text(
                x,
                -arrow_h - 0.02 * L_total,
                f"{label}={val_user:g} kg",
                ha="center",
                va="top",
                fontsize=style.font_size,
                color="red",
            )

    # Momentos (se dibujan luego en px)
    moments_to_draw = list(data.moments)

    # Altura base de cargas
    max_y = max(arrow_h, dist_h) * 1.9
    max_y = max(max_y, 0.35 * L_total)
    max_y *= float(y_zoom)

    # xlim extendido
    if xlim is None:
        xs = [0.0, L_total]
        xs += [float(pf.x_mm) for pf in data.point_forces]
        xs += [float(pm.x_mm) for pm in data.moments]
        for dl in data.dist_loads:
            xs += [float(dl.x1_mm), float(dl.x2_mm)]
        x_min = float(min(xs))
        x_max = float(max(xs))
        span = max(1.0, x_max - x_min)
        margin = 0.05 * span
        ax.set_xlim(x_min - margin, x_max + margin)
    else:
        ax.set_xlim(float(xlim[0]), float(xlim[1]))

    # -------------------------
    # COTAS (arriba/abajo por lado) + Lc + Ltotal en ARRIBA
    # -------------------------
    targets = _collect_dimension_targets(data)

    def _add_unique(lst: List[float], x: float):
        if abs(x) < 1e-9:
            return
        if all(abs(x - v) > 1e-9 for v in lst):
            lst.append(x)

    _add_unique(targets["top"], L_car)
    _add_unique(targets["top"], L_total)

    # ORDEN mirando desde x=0: más cercano primero
    targets["top"].sort(key=lambda v: abs(v))
    targets["bottom"].sort(key=lambda v: abs(v))

    # GAP inferior: evita tocar viga y flechas rojas (pero SÍ toca línea de cota)
    gap_obj_mm = max(arrow_h, dist_h) + 0.06 * L_total

    # niveles de cotas
    y_top0 = +1.25 * max_y
    y_bot0 = -1.25 * max_y
    dy = 0.22 * max_y

    # Ajustar ylim para que entren cotas
    top_max = y_top0 + dy * max(0, len(targets["top"]) - 1) + 0.35 * max_y
    bot_min = y_bot0 - dy * max(0, len(targets["bottom"]) - 1) - 0.35 * max_y

    ax.set_ylim(min(bot_min, -max_y - 0.15 * L_total), max(top_max, max_y + 0.15 * L_total))
    ax.set_aspect("auto", adjustable="box")

    # REF
    #ax.text(0.0, y_top0 + 0.10 * max_y, "REF", ha="left", va="bottom", fontsize=9, color="red")

    # Cotas arriba
    for i, x in enumerate(targets["top"]):
        y = y_top0 + i * dy
        dist = abs(x - 0.0)
        _draw_dimension_from_ref(
            ax,
            x_ref_real=0.0,
            x_real=float(x),
            y_dim=float(y),
            text=f"{_ceil_mm(dist)}",
            gap_obj_mm=float(gap_obj_mm),
            color="blue",
        )

    # Cotas abajo
    for i, x in enumerate(targets["bottom"]):
        y = y_bot0 - i * dy
        dist = abs(x - 0.0)
        _draw_dimension_from_ref(
            ax,
            x_ref_real=0.0,
            x_real=float(x),
            y_dim=float(y),
            text=f"{_ceil_mm(dist)}",
            gap_obj_mm=float(gap_obj_mm),
            color="blue",
        )

    # -------------------------
    # Momentos (circular en px) - después de límites definitivos
    # -------------------------
    moment_r_px = float(max(14.0, 12.0 * float(style.moment_radius_pctL) / 8.0))

    for pm in moments_to_draw:
        x = float(pm.x_mm)
        M = pm.M_internal
        label = pm.label
        M_user = pm.M_user_kgmm

        lx, ly = _draw_moment_px(ax, x, M, moment_r_px, style)
        ax.text(
            lx,
            ly,
            f"{label}={M_user:g} kg·mm",
            ha="left",
            va="bottom",
            fontsize=style.font_size,
            color="red",
        )

    # Formato
    ax.set_xlabel("x [mm]")
    ax.set_yticks([])
    ax.set_title("Diagrama de Cuerpo Libre (FBD)")
    ax.grid(True, axis="x", alpha=0.25)
