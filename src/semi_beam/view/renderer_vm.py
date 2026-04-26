from __future__ import annotations

from typing import Optional, Tuple, List, Set
import numpy as np

from semi_beam.view.label_placement import annotate_smart


# -------------------------
# Helpers formato
# -------------------------
def _fmt_plain(v: float, decimals: int = 2) -> str:
    """Formato fijo (sin notación científica) y recorte de ceros."""
    s = f"{float(v):.{decimals}f}"
    if "." in s:
        s = s.rstrip("0").rstrip(".")
    return s


def _clamp(v: float, lo: float, hi: float) -> float:
    return float(min(max(v, lo), hi))


# -------------------------
# Extremos locales robustos
# -------------------------
def _find_local_extrema_indices(y: np.ndarray, *, tol_slope: float) -> List[tuple[str, int]]:
    """
    Detecta extremos locales por cambios de signo en dy, IGNORANDO mesetas (dy≈0).
    Devuelve lista de ("max"/"min", idx_en_y).
    """
    n = len(y)
    if n < 5:
        return []

    dy = np.diff(y)
    # signos de la pendiente con tolerancia
    s = np.zeros_like(dy, dtype=int)
    s[dy > +tol_slope] = +1
    s[dy < -tol_slope] = -1

    # Nos quedamos solo con cambios entre pendientes no-nulas
    nz = np.nonzero(s)[0]  # índices en dy donde hay pendiente clara
    if nz.size < 2:
        return []

    s_nz = s[nz]  # signos compactados
    out: List[tuple[str, int]] = []

    # Si pasamos de + a - => máximo. Si de - a + => mínimo.
    # El extremo cae cerca de la "unión": usamos idx = nz[k] + 1 (en y)
    for k in range(1, len(s_nz)):
        if s_nz[k - 1] > 0 and s_nz[k] < 0:
            out.append(("max", int(nz[k - 1] + 1)))
        elif s_nz[k - 1] < 0 and s_nz[k] > 0:
            out.append(("min", int(nz[k - 1] + 1)))

    return out


def _select_extrema_with_spacing(
    x: np.ndarray,
    y: np.ndarray,
    candidates: List[tuple[str, int]],
    *,
    y_abs_min: float,
    min_dx: float,
) -> List[tuple[str, int]]:
    """
    Filtra:
    - extremos con |y| muy chico (cerca de 0)
    - demasiados extremos cercanos en X (aplica separación mínima min_dx)
    Estrategia: prioriza por |y| descendente y se queda con los más relevantes.
    """
    if not candidates:
        return []

    # Filtrar por amplitud (evita etiquetar M=0 en tramos planos)
    cand2 = [(k, i) for (k, i) in candidates if abs(float(y[i])) >= y_abs_min]
    if not cand2:
        return []

    # Prioridad por magnitud
    cand2.sort(key=lambda ki: abs(float(y[ki[1]])), reverse=True)

    picked: List[tuple[str, int]] = []
    picked_x: List[float] = []

    for kind, i in cand2:
        xi = float(x[i])
        if all(abs(xi - xj) >= min_dx for xj in picked_x):
            picked.append((kind, i))
            picked_x.append(xi)

    # Orden final por x (más prolijo)
    picked.sort(key=lambda ki: float(x[ki[1]]))
    return picked


def _prefer_for_point(xi: float, yi: float, x_mid: float) -> tuple[str, ...]:
    if yi >= 0.0:
        return ("NE", "NW", "SE", "SW") if xi <= x_mid else ("NW", "NE", "SW", "SE")
    return ("SE", "SW", "NE", "NW") if xi <= x_mid else ("SW", "SE", "NW", "NE")


def _annotate_curve_extrema(
    ax,
    x: np.ndarray,
    y: np.ndarray,
    *,
    label_scale: float,
    label_suffix: str,
    color: str,
    include_locals: bool,
    avoid_line=None,
):
    """
    Anota extremos relevantes de una curva usando el helper común.
    """
    if len(x) == 0:
        return

    max_abs = float(np.max(np.abs(y))) if len(y) else 1.0
    max_abs = max(max_abs, 1.0)
    x_min, x_max = ax.get_xlim()
    x_span = max(1.0, float(x_max - x_min))
    x_mid = 0.5 * (x_min + x_max)

    extrema: List[tuple[str, int]] = []
    if include_locals:
        tol_slope = 1e-6 * max_abs
        extrema.extend(_find_local_extrema_indices(y, tol_slope=tol_slope))
    extrema.extend([("max", int(np.argmax(y))), ("min", int(np.argmin(y)))])

    seen: Set[int] = set()
    uniq: List[tuple[str, int]] = []
    for kind, idx in extrema:
        if idx in seen:
            continue
        seen.add(idx)
        uniq.append((kind, idx))

    y_abs_min = max(0.01 * max_abs, 1.0)
    min_dx = max(0.03 * x_span, 120.0)
    picked = _select_extrema_with_spacing(x, y, uniq, y_abs_min=y_abs_min, min_dx=min_dx)

    if not picked:
        return

    for kind, i in picked:
        xi = float(x[i])
        yi = float(y[i])
        ax.scatter([xi], [yi], s=18, zorder=6, color=color)
        prefer = _prefer_for_point(xi, yi, x_mid)
        annotate_smart(
            ax,
            (xi, yi),
            f"{_fmt_plain(yi / label_scale, 2)} {label_suffix}",
            color=color,
            prefer=prefer,
            avoid_line=avoid_line,
        )


# -------------------------
# Render
# -------------------------
def render_shear(ax, diag, y_zoom: float = 1.0, xlim: Optional[Tuple[float, float]] = None):
    ax.clear()
    ax._smart_label_bboxes = []
    x, V, _ = diag.sample(n_per_segment=80)

    line = ax.plot(x, V, color="#1C4E80")[0]
    ax.axhline(0.0, linewidth=1.0)

    if xlim is None:
        ax.set_xlim(diag.x_start, diag.x_end)
    else:
        ax.set_xlim(xlim[0], xlim[1])

    vmax = float(np.max(np.abs(V))) if len(V) else 1.0
    vmax = max(vmax, 1.0)
    pad = 1.15
    ax.set_ylim(-vmax * y_zoom * pad, vmax * y_zoom * pad)
    _annotate_curve_extrema(
        ax,
        np.asarray(x, dtype=float),
        np.asarray(V, dtype=float),
        label_scale=1.0,
        label_suffix="kg",
        color="#1C4E80",
        include_locals=False,
        avoid_line=line,
    )

    ax.set_ylabel("V [kg]")
    ax.set_title("Diagrama de Corte V(x)")
    ax.grid(True, alpha=0.25)
    return line


def render_moment(ax, diag, y_zoom: float = 1.0, xlim: Optional[Tuple[float, float]] = None):
    ax.clear()
    ax._smart_label_bboxes = []
    x, _, M = diag.sample(n_per_segment=80)

    line = ax.plot(x, M, color="#2B2D42")[0]
    ax.axhline(0.0, linewidth=1.0)

    if xlim is None:
        ax.set_xlim(diag.x_start, diag.x_end)
    else:
        ax.set_xlim(xlim[0], xlim[1])

    mmax = float(np.max(np.abs(M))) if len(M) else 1.0
    mmax = max(mmax, 1.0)
    pad = 1.15
    ax.set_ylim(-mmax * y_zoom * pad, mmax * y_zoom * pad)

    # ✅ extremos (filtrados y sin spam en M=0)
    _annotate_curve_extrema(
        ax,
        np.asarray(x, dtype=float),
        np.asarray(M, dtype=float),
        label_scale=10.0,
        label_suffix="kg·cm",
        color="#2B2D42",
        include_locals=True,
        avoid_line=line,
    )

    ax.set_ylabel("M [kg·mm]")
    ax.set_xlabel("x [mm]")
    ax.set_title("Diagrama de Momento Flector M(x)")
    ax.grid(True, alpha=0.25)
    return line


def render_deflection(ax, result, y_zoom: float = 1.0, xlim: Optional[Tuple[float, float]] = None):
    ax.clear()
    ax._smart_label_bboxes = []
    x = np.asarray(result.x_mm, dtype=float)
    v_total = np.asarray(result.v_total_mm, dtype=float)
    v_pre = np.asarray(result.v_precamber_mm, dtype=float)

    line = ax.plot(x, v_total, label="v_total(x)", color="#145DA0", linewidth=1.8)[0]
    ax.plot(x, v_pre, label="Convexidad inicial", color="#7AA6D1", linewidth=1.1, linestyle="--")
    ax.axhline(0.0, linewidth=1.0, color="#555555")
    ax.axhline(float(result.limit_y_mm), linewidth=1.1, color="#B00020", linestyle="--", label="Límite")

    ax.scatter([float(result.x_vmin_mm)], [float(result.vmin_mm)], color="#B00020", s=24, zorder=6)
    if xlim is None:
        ax.set_xlim(float(x[0]), float(x[-1]))
    else:
        ax.set_xlim(xlim[0], xlim[1])

    vmax = float(np.max(np.abs(np.concatenate([v_total, v_pre, np.asarray([result.limit_y_mm], dtype=float)])))) if len(x) else 1.0
    vmax = max(vmax, 1.0)
    pad = 1.20
    ax.set_ylim(-vmax * y_zoom * pad, vmax * y_zoom * pad)
    x_min, x_max = ax.get_xlim()
    x_mid = 0.5 * (x_min + x_max)
    prefer = _prefer_for_point(float(result.x_vmin_mm), float(result.vmin_mm), x_mid)
    annotate_smart(
        ax,
        (float(result.x_vmin_mm), float(result.vmin_mm)),
        f"vmin={_fmt_plain(float(result.vmin_mm), 2)} mm @ x={_fmt_plain(float(result.x_vmin_mm), 0)} mm",
        color="#B00020",
        prefer=prefer,
        avoid_line=line,
    )
    ax.set_ylabel("v [mm]")
    ax.set_xlabel("x [mm]")
    ax.set_title("Deformada Total v(x)")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best", fontsize=8, framealpha=0.9)
    return line
