# path: src/semi_beam/view/diagram_hover.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Optional

import numpy as np


@dataclass
class HoverCurve:
    ax: Any
    line: Any
    label: str
    x_unit: str = "mm"
    y_unit: str = ""
    y_display_scale: float = 1.0
    x_decimals: int = 0
    y_decimals: int = 0


@dataclass
class _NearestPoint:
    curve: HoverCurve
    x: float
    y: float
    distance_px: float


class DiagramHoverInspector:
    def __init__(self, canvas, curves: Iterable[HoverCurve], tolerance_px: float = 14):
        self.canvas = canvas
        self.curves = list(curves)
        self.tolerance_px = float(tolerance_px)
        self._motion_cid: Optional[int] = self.canvas.mpl_connect("motion_notify_event", self._on_motion)
        self._axes_leave_cid: Optional[int] = self.canvas.mpl_connect("axes_leave_event", self._on_axes_leave)
        self._figure_leave_cid: Optional[int] = self.canvas.mpl_connect("figure_leave_event", self._on_figure_leave)
        self._markers: dict[Any, Any] = {}
        self._annotations: dict[Any, Any] = {}
        self._visible_ax: Any = None

    def disconnect(self) -> None:
        for cid in (self._motion_cid, self._axes_leave_cid, self._figure_leave_cid):
            if cid is not None:
                self.canvas.mpl_disconnect(cid)
        self._motion_cid = None
        self._axes_leave_cid = None
        self._figure_leave_cid = None
        self.hide()

    def replace_curve(self, ax, curve: Optional[HoverCurve]) -> None:
        """Replace one subplot curve without discarding hover state on other axes."""
        self.curves = [current for current in self.curves if current.ax is not ax]
        if curve is not None:
            self.curves.append(curve)
        for artists in (self._markers, self._annotations):
            artist = artists.pop(ax, None)
            if artist is not None:
                try:
                    artist.remove()
                except (NotImplementedError, ValueError):
                    pass
        if self._visible_ax is ax:
            self._visible_ax = None

    def hide(self) -> None:
        changed = False
        for artist in list(self._markers.values()) + list(self._annotations.values()):
            if artist.get_visible():
                artist.set_visible(False)
                changed = True
        self._visible_ax = None
        if changed:
            self.canvas.draw_idle()

    def _on_axes_leave(self, event) -> None:
        self.hide()

    def _on_figure_leave(self, event) -> None:
        self.hide()

    def _on_motion(self, event) -> None:
        if event.inaxes is None or event.x is None or event.y is None:
            self.hide()
            return

        nearest = self._nearest_point(event)
        if nearest is None or nearest.distance_px > self.tolerance_px:
            self.hide()
            return

        self._show(nearest)

    def _nearest_point(self, event) -> Optional[_NearestPoint]:
        best: Optional[_NearestPoint] = None

        for curve in self.curves:
            if curve.ax is not event.inaxes:
                continue

            xdata = np.asarray(curve.line.get_xdata(), dtype=float)
            ydata = np.asarray(curve.line.get_ydata(), dtype=float)
            if xdata.size == 0 or ydata.size == 0:
                continue

            n = min(xdata.size, ydata.size)
            x = xdata[:n]
            y = ydata[:n]
            finite = np.isfinite(x) & np.isfinite(y)
            if not np.any(finite):
                continue

            x = x[finite]
            y = y[finite]
            points_px = curve.ax.transData.transform(np.column_stack([x, y]))
            dx = points_px[:, 0] - float(event.x)
            dy = points_px[:, 1] - float(event.y)
            distances = np.hypot(dx, dy)
            idx = int(np.argmin(distances))
            distance = float(distances[idx])

            if best is None or distance < best.distance_px:
                best = _NearestPoint(
                    curve=curve,
                    x=float(x[idx]),
                    y=float(y[idx]),
                    distance_px=distance,
                )

        return best

    def _show(self, nearest: _NearestPoint) -> None:
        curve = nearest.curve
        ax = curve.ax
        marker, annotation = self._ensure_artists(ax, curve)
        changed = False

        for other_ax, other_marker in self._markers.items():
            should_show = other_ax is ax
            if other_marker.get_visible() != should_show:
                other_marker.set_visible(should_show)
                changed = True
        for other_ax, other_annotation in self._annotations.items():
            should_show = other_ax is ax
            if other_annotation.get_visible() != should_show:
                other_annotation.set_visible(should_show)
                changed = True

        old_x, old_y = marker.get_data()
        if len(old_x) != 1 or len(old_y) != 1 or float(old_x[0]) != nearest.x or float(old_y[0]) != nearest.y:
            marker.set_data([nearest.x], [nearest.y])
            changed = True
        if not marker.get_visible():
            marker.set_visible(True)
            changed = True

        label = self._label_for(nearest)
        if annotation.get_text() != label:
            annotation.set_text(label)
            changed = True
        if not annotation.get_visible():
            annotation.set_visible(True)
            changed = True

        self._visible_ax = ax
        if changed:
            self.canvas.draw_idle()

    def _ensure_artists(self, ax, curve: HoverCurve):
        return self._marker_for(ax, curve), self._annotation_for(ax)

    def _marker_for(self, ax, curve: HoverCurve):
        marker = self._markers.get(ax)
        if marker is None:
            color = curve.line.get_color()
            marker = ax.plot(
                [],
                [],
                marker="o",
                markersize=6,
                markerfacecolor="white",
                markeredgecolor=color,
                markeredgewidth=1.5,
                linestyle="None",
                zorder=20,
                visible=False,
            )[0]
            self._markers[ax] = marker
        return marker

    def _annotation_for(self, ax):
        annotation = self._annotations.get(ax)
        if annotation is None:
            annotation = ax.text(
                0.98,
                0.95,
                "",
                transform=ax.transAxes,
                ha="right",
                va="top",
                bbox={"boxstyle": "round,pad=0.25", "fc": "white", "ec": "#555555", "alpha": 0.95},
                fontsize=8,
                zorder=21,
                visible=False,
            )
            self._annotations[ax] = annotation
        return annotation

    def _label_for(self, point: _NearestPoint) -> str:
        curve = point.curve
        x_text = _format_number(point.x, curve.x_decimals)
        y_text = _format_number(point.y * curve.y_display_scale, curve.y_decimals)
        x_unit = f" {curve.x_unit}" if curve.x_unit else ""
        y_unit = f" {curve.y_unit}" if curve.y_unit else ""
        return f"x = {x_text}{x_unit}\n{curve.label} = {y_text}{y_unit}"


def _format_number(value: float, decimals: int) -> str:
    text = f"{float(value):,.{int(decimals)}f}"
    return text.replace(",", "_").replace(".", ",").replace("_", ".")
