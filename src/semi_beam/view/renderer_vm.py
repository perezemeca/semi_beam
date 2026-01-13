from __future__ import annotations

from typing import Optional, Tuple, List, Set
import numpy as np


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


def _annotate_moment_extrema(ax, x: np.ndarray, M_kgmm: np.ndarray):
    """
    Marca máximos/mínimos locales y anota valor en kg·cm.
    - Máximo: label arriba
    - Mínimo: label abajo
    - Reubica labels dentro del recuadro
    - Evita etiquetar M≈0 en tramos planos
    """
    if len(x) == 0:
        return

    max_abs = float(np.max(np.abs(M_kgmm))) if len(M_kgmm) else 1.0
    max_abs = max(max_abs, 1.0)

    # tolerancia de pendiente (para ignorar "ruido" y mesetas)
    tol_slope = 1e-6 * max_abs  # ajustable: más grande => menos extremos

    # candidatos locales
    extrema = _find_local_extrema_indices(M_kgmm, tol_slope=tol_slope)

    # agregamos extremos globales, pero NO queremos spam en 0:
    i_max = int(np.argmax(M_kgmm))
    i_min = int(np.argmin(M_kgmm))
    extrema.extend([("max", i_max), ("min", i_min)])

    # de-duplicar por índice
    seen: Set[int] = set()
    uniq: List[tuple[str, int]] = []
    for kind, i in extrema:
        if i in seen:
            continue
        seen.add(i)
        uniq.append((kind, i))

    # umbral mínimo para mostrar (evita marcar línea base)
    y_abs_min = max(0.01 * max_abs, 1.0)  # 1% del máximo o 1 kg·mm

    # separación mínima en x entre etiquetas
    x_min, x_max = ax.get_xlim()
    x_span = max(1.0, float(x_max - x_min))
    min_dx = max(0.03 * x_span, 120.0)  # 3% del ancho o 120 mm

    picked = _select_extrema_with_spacing(
        x, M_kgmm, uniq,
        y_abs_min=y_abs_min,
        min_dx=min_dx,
    )

    if not picked:
        return

    # márgenes internos para que el texto no “se salga”
    y_min, y_max = ax.get_ylim()
    y_span = max(1.0, float(y_max - y_min))
    # Margen fijo para que las etiquetas no toquen el borde (X e Y)
    mx = 0.03 * x_span
    my = 0.03 * y_span

    for kind, i in picked:
        xi = float(x[i])
        Mi = float(M_kgmm[i])
        Mi_kgcm = Mi / 10.0

        # marcador pequeño
        ax.scatter([xi], [Mi], s=18, zorder=6)

        # posición preferida del texto
        if kind == "max":
            tx, ty = xi, Mi + my
            va = "bottom"
        else:
            tx, ty = xi, Mi - my
            va = "top"

        # clamp a la caja del gráfico
        tx = _clamp(tx, x_min + mx, x_max - mx)
        ty = _clamp(ty, y_min + my, y_max - my)

        ax.text(
            tx, ty,
            f"{_fmt_plain(Mi_kgcm, 2)} kg·cm",
            ha="center", va=va, fontsize=8, zorder=7
        )


# -------------------------
# Render
# -------------------------
def render_shear(ax, diag, y_zoom: float = 1.0, xlim: Optional[Tuple[float, float]] = None):
    ax.clear()
    x, V, _ = diag.sample(n_per_segment=80)

    ax.plot(x, V)
    ax.axhline(0.0, linewidth=1.0)

    if xlim is None:
        ax.set_xlim(diag.x_start, diag.x_end)
    else:
        ax.set_xlim(xlim[0], xlim[1])

    vmax = float(np.max(np.abs(V))) if len(V) else 1.0
    vmax = max(vmax, 1.0)
    pad = 1.15
    ax.set_ylim(-vmax * y_zoom * pad, vmax * y_zoom * pad)

    ax.set_ylabel("V [kg]")
    ax.set_title("Diagrama de Corte V(x)")
    ax.grid(True, alpha=0.25)


def render_moment(ax, diag, y_zoom: float = 1.0, xlim: Optional[Tuple[float, float]] = None):
    ax.clear()
    x, _, M = diag.sample(n_per_segment=80)

    ax.plot(x, M)
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
    _annotate_moment_extrema(ax, np.asarray(x, dtype=float), np.asarray(M, dtype=float))

    ax.set_ylabel("M [kg·mm]")
    ax.set_xlabel("x [mm]")
    ax.set_title("Diagrama de Momento Flector M(x)")
    ax.grid(True, alpha=0.25)
