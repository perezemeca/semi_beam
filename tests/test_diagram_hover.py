# path: tests/test_diagram_hover.py
from __future__ import annotations

from types import SimpleNamespace

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from semi_beam.view.diagram_hover import DiagramHoverInspector, HoverCurve, _format_number


def _event_at(ax, x: float, y: float):
    px, py = ax.transData.transform((x, y))
    return SimpleNamespace(inaxes=ax, x=float(px), y=float(py))


def test_nearest_point_finds_curve_point_in_screen_coordinates():
    fig, ax = plt.subplots()
    try:
        (line,) = ax.plot([0, 10, 20], [0, 100, 0])
        fig.canvas.draw()

        curve = HoverCurve(ax=ax, line=line, label="V", y_unit="kg")
        inspector = DiagramHoverInspector(fig.canvas, [curve])
        try:
            nearest = inspector._nearest_point(_event_at(ax, 10, 100))
        finally:
            inspector.disconnect()

        assert nearest is not None
        assert nearest.x == 10
        assert nearest.y == 100
        assert nearest.curve.label == "V"
    finally:
        plt.close(fig)


def test_nearest_point_ignores_non_finite_values():
    fig, ax = plt.subplots()
    try:
        (line,) = ax.plot([0, 10, np.nan, 20], [0, 100, 500, np.inf])
        fig.canvas.draw()

        curve = HoverCurve(ax=ax, line=line, label="V", y_unit="kg")
        inspector = DiagramHoverInspector(fig.canvas, [curve])
        try:
            nearest = inspector._nearest_point(_event_at(ax, 10, 100))
        finally:
            inspector.disconnect()

        assert nearest is not None
        assert nearest.x == 10
        assert nearest.y == 100
    finally:
        plt.close(fig)


def test_format_number_uses_spanish_style_separators():
    assert _format_number(7800, 0) == "7.800"
    assert _format_number(-18.4, 1) == "-18,4"
