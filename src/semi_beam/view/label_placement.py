from __future__ import annotations

from typing import Iterable, Sequence

import numpy as np
from matplotlib.transforms import Bbox


_OFFSET_MAP = {
    "NE": (12.0, 12.0),
    "NW": (-12.0, 12.0),
    "SE": (12.0, -12.0),
    "SW": (-12.0, -12.0),
    "N": (0.0, 14.0),
    "S": (0.0, -14.0),
    "E": (14.0, 0.0),
    "W": (-14.0, 0.0),
}


def _get_renderer(ax):
    canvas = ax.figure.canvas
    try:
        return canvas.get_renderer()
    except Exception:
        canvas.draw()
        return canvas.get_renderer()


def _alignment_from_offset(dx_pts: float, dy_pts: float) -> tuple[str, str]:
    ha = "left" if dx_pts >= 0.0 else "right"
    va = "bottom" if dy_pts >= 0.0 else "top"
    if abs(dx_pts) < 1e-9:
        ha = "center"
    if abs(dy_pts) < 1e-9:
        va = "center"
    return ha, va


def _extract_line_display_points(ax, avoid_line) -> np.ndarray | None:
    if avoid_line is None:
        return None

    x = y = None
    if hasattr(avoid_line, "get_xdata") and hasattr(avoid_line, "get_ydata"):
        try:
            x = np.asarray(avoid_line.get_xdata(orig=False), dtype=float)
            y = np.asarray(avoid_line.get_ydata(orig=False), dtype=float)
        except Exception:
            x = y = None
    elif isinstance(avoid_line, (tuple, list)) and len(avoid_line) == 2:
        try:
            x = np.asarray(avoid_line[0], dtype=float)
            y = np.asarray(avoid_line[1], dtype=float)
        except Exception:
            x = y = None

    if x is None or y is None or x.size == 0 or y.size == 0:
        return None

    mask = np.isfinite(x) & np.isfinite(y)
    if not np.any(mask):
        return None

    pts = np.column_stack([x[mask], y[mask]])
    if pts.size == 0:
        return None
    return np.asarray(ax.transData.transform(pts), dtype=float)


def _candidate_offsets(prefer: Sequence[str]) -> list[tuple[float, float]]:
    out: list[tuple[float, float]] = []
    magnitudes = (12.0, 18.0, 24.0)
    for mag in magnitudes:
        for key in prefer:
            base = _OFFSET_MAP.get(str(key).upper())
            if base is None:
                continue
            sx = 0.0 if abs(base[0]) < 1e-9 else np.sign(base[0])
            sy = 0.0 if abs(base[1]) < 1e-9 else np.sign(base[1])
            dx = float(base[0]) if sx == 0.0 else float(sx * mag)
            dy = float(base[1]) if sy == 0.0 else float(sy * mag)
            cand = (dx, dy)
            if cand not in out:
                out.append(cand)
    if not out:
        out.append((12.0, 12.0))
    return out


def _make_annotation(ax, xy, text: str, dx_pts: float, dy_pts: float, color: str):
    ha, va = _alignment_from_offset(dx_pts, dy_pts)
    ann = ax.annotate(
        text,
        xy=xy,
        xytext=(float(dx_pts), float(dy_pts)),
        textcoords="offset points",
        ha=ha,
        va=va,
        color=color,
        bbox=dict(boxstyle="round,pad=0.25", fc="white", ec=color, alpha=0.85),
        arrowprops=dict(arrowstyle="->", color=color, lw=1.0),
        clip_on=True,
        annotation_clip=True,
        zorder=20,
    )
    return ann


def _annotation_text_bbox(ann, renderer) -> Bbox:
    try:
        ann.update_positions(renderer)
        ann.update_bbox_position_size(renderer)
    except Exception:
        pass
    patch = ann.get_bbox_patch()
    if patch is not None:
        return patch.get_window_extent(renderer=renderer)
    return ann.get_window_extent(renderer=renderer)


def _bbox_inside_axes(bbox: Bbox, axes_bbox: Bbox, pad_px: float) -> bool:
    return (
        bbox.x0 >= axes_bbox.x0 + pad_px
        and bbox.y0 >= axes_bbox.y0 + pad_px
        and bbox.x1 <= axes_bbox.x1 - pad_px
        and bbox.y1 <= axes_bbox.y1 - pad_px
    )


def _bbox_hits_line(bbox: Bbox, line_pts: np.ndarray | None, pad_px: float) -> bool:
    if line_pts is None or line_pts.size == 0:
        return False
    x0 = bbox.x0 - pad_px
    y0 = bbox.y0 - pad_px
    x1 = bbox.x1 + pad_px
    y1 = bbox.y1 + pad_px
    xs = line_pts[:, 0]
    ys = line_pts[:, 1]
    mask = (xs >= x0) & (xs <= x1) & (ys >= y0) & (ys <= y1)
    return bool(np.any(mask))


def _bbox_hits_used(bbox: Bbox, used_bboxes: Iterable[Bbox], pad_px: float) -> bool:
    probe = Bbox.from_extents(bbox.x0 - pad_px, bbox.y0 - pad_px, bbox.x1 + pad_px, bbox.y1 + pad_px)
    for other in used_bboxes:
        expanded = Bbox.from_extents(other.x0 - pad_px, other.y0 - pad_px, other.x1 + pad_px, other.y1 + pad_px)
        if probe.overlaps(expanded):
            return True
    return False


def _clamped_offset(ax, xy, text: str, *, color: str, dx_pts: float, dy_pts: float, pad_px: float, axes_bbox: Bbox, renderer) -> tuple[float, float]:
    ann = _make_annotation(ax, xy, text, dx_pts, dy_pts, color)
    try:
        bbox = _annotation_text_bbox(ann, renderer)
    finally:
        ann.remove()

    px_per_pt = float(ax.figure.dpi) / 72.0
    dx_px = 0.0
    dy_px = 0.0
    if bbox.x0 < axes_bbox.x0 + pad_px:
        dx_px += (axes_bbox.x0 + pad_px) - bbox.x0
    if bbox.x1 > axes_bbox.x1 - pad_px:
        dx_px -= bbox.x1 - (axes_bbox.x1 - pad_px)
    if bbox.y0 < axes_bbox.y0 + pad_px:
        dy_px += (axes_bbox.y0 + pad_px) - bbox.y0
    if bbox.y1 > axes_bbox.y1 - pad_px:
        dy_px -= bbox.y1 - (axes_bbox.y1 - pad_px)
    return float(dx_pts + (dx_px / px_per_pt)), float(dy_pts + (dy_px / px_per_pt))


def annotate_smart(ax, xy, text, *, color="red", prefer=("NE", "NW", "SE", "SW"), avoid_line=None, pad_px=6):
    renderer = _get_renderer(ax)
    axes_bbox = ax.get_window_extent(renderer=renderer)
    line_pts = _extract_line_display_points(ax, avoid_line)

    if not hasattr(ax, "_smart_label_bboxes"):
        ax._smart_label_bboxes = []
    used_bboxes: list[Bbox] = list(ax._smart_label_bboxes)

    for dx_pts, dy_pts in _candidate_offsets(prefer):
        ann = _make_annotation(ax, xy, text, dx_pts, dy_pts, color)
        bbox = _annotation_text_bbox(ann, renderer)
        ok = (
            _bbox_inside_axes(bbox, axes_bbox, float(pad_px))
            and not _bbox_hits_line(bbox, line_pts, float(pad_px))
            and not _bbox_hits_used(bbox, used_bboxes, float(pad_px))
        )
        if ok:
            ax._smart_label_bboxes.append(bbox)
            return ann
        ann.remove()

    dx_pts, dy_pts = _candidate_offsets(prefer)[0]
    cdx, cdy = _clamped_offset(
        ax,
        xy,
        text,
        color=str(color),
        dx_pts=float(dx_pts),
        dy_pts=float(dy_pts),
        pad_px=float(pad_px),
        axes_bbox=axes_bbox,
        renderer=renderer,
    )
    ann = _make_annotation(ax, xy, text, cdx, cdy, str(color))
    bbox = _annotation_text_bbox(ann, renderer)
    ax._smart_label_bboxes.append(bbox)
    return ann
