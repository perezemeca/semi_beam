import matplotlib

matplotlib.use("Agg")

import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from matplotlib.text import Annotation

from semi_beam.view.label_placement import annotate_smart


def test_annotate_smart_keeps_text_bbox_inside_axes():
    fig = Figure(figsize=(5, 3), dpi=100)
    canvas = FigureCanvasAgg(fig)
    ax = fig.add_subplot(111)

    x = np.linspace(0.0, 10.0, 200)
    y = 0.2 * x + 0.1
    line = ax.plot(x, y)[0]
    ax.set_xlim(0.0, 10.0)
    ax.set_ylim(0.0, 2.2)

    canvas.draw()

    ann = annotate_smart(
        ax,
        (10.0, y[-1]),
        "borde derecho",
        color="red",
        prefer=("NE", "SE", "NW", "SW"),
        avoid_line=line,
    )

    canvas.draw()
    renderer = canvas.get_renderer()

    text_bbox = ann.get_bbox_patch().get_window_extent(renderer=renderer)
    axes_bbox = ax.get_window_extent(renderer=renderer)

    assert text_bbox.x0 >= axes_bbox.x0 - 1.0
    assert text_bbox.y0 >= axes_bbox.y0 - 1.0
    assert text_bbox.x1 <= axes_bbox.x1 + 1.0
    assert text_bbox.y1 <= axes_bbox.y1 + 1.0


def test_annotate_smart_does_not_rasterize_discarded_candidates(monkeypatch):
    fig = Figure(figsize=(5, 3), dpi=100)
    canvas = FigureCanvasAgg(fig)
    ax = fig.add_subplot(111)
    line = ax.plot([0.0, 10.0], [0.0, 2.0])[0]
    ax.set_xlim(0.0, 10.0)
    ax.set_ylim(0.0, 2.0)
    canvas.draw()

    draw_calls = 0
    original_draw = Annotation.draw

    def draw(annotation, renderer):
        nonlocal draw_calls
        draw_calls += 1
        return original_draw(annotation, renderer)

    monkeypatch.setattr(Annotation, "draw", draw)
    annotation = annotate_smart(
        ax,
        (10.0, 2.0),
        "borde derecho",
        prefer=("NE", "SE", "NW", "SW"),
        avoid_line=line,
    )

    assert draw_calls == 0
    canvas.draw()
    assert draw_calls == 1
    assert annotation.get_bbox_patch() is not None
