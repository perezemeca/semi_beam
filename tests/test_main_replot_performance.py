import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from matplotlib.text import Annotation
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication

from semi_beam.domain.beam import Beam
from semi_beam.domain.loads import DistUniform, PointForce, PointMoment
from semi_beam.ui.main_window import FBDApp, SessionCache


def _app() -> QApplication:
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


def _cache(*, heavy: bool) -> SessionCache:
    if heavy:
        points = [
            PointForce("Rp1", 1000.0, 9000.0),
            PointForce("P1", 2200.0, 3500.0),
            PointForce("P2", 4300.0, 4200.0),
            PointForce("P3", 7300.0, -1600.0),
            PointForce("P4", 10600.0, 2800.0),
            PointForce("Rt", 11200.0, 12500.0),
        ]
        dists = [
            DistUniform("q1", 5200.0, 1200.0, 3.5),
            DistUniform("q2", 8500.0, 900.0, 4.0),
        ]
        moments = [
            PointMoment("M1", 3000.0, 250000.0),
            PointMoment("M2", 9800.0, -180000.0),
        ]
    else:
        points = [
            PointForce("Rp1", 1000.0, 10000.0),
            PointForce("P1", 4000.0, 15000.0),
            PointForce("Rt", 11000.0, 5000.0),
        ]
        dists = []
        moments = []
    return SessionCache(
        beam_plot=Beam(L_mm=12000.0),
        points=points,
        dists=dists,
        moms=moments,
        note_text="test",
        deflection_supports=(1000.0, 11000.0),
    )


def _prepare_window() -> tuple[FBDApp, object]:
    window = FBDApp()
    tab = window.tab_semi
    window.tabs.setCurrentWidget(tab)
    panel = tab.section_panel
    panel.tbl.blockSignals(True)
    try:
        panel.tbl.item(0, panel.COL_X).setText("0")
        panel.tbl.item(0, panel.COL_HWEB).setText("450")
    finally:
        panel.tbl.blockSignals(False)
    panel._timer.stop()
    window._redraw_timer.stop()
    return window, tab


def _axis_signature(window: FBDApp) -> tuple[tuple[int, int, int, int], ...]:
    return tuple(
        (len(ax.lines), len(ax.patches), len(ax.collections), len(ax.texts))
        for ax in (window.ax_fbd, window.ax_V, window.ax_M, window.ax_defl)
    )


def test_main_replot_preserves_all_axes_across_state_transitions():
    app = _app()
    window, tab = _prepare_window()

    tab.set_cache(_cache(heavy=False))
    window._replot_active_tab()
    app.processEvents()
    simple_signature = _axis_signature(window)

    tab.set_cache(_cache(heavy=True))
    window._replot_active_tab()
    app.processEvents()
    heavy_signature = _axis_signature(window)

    tab.set_cache(_cache(heavy=False))
    window._replot_active_tab()
    app.processEvents()

    assert _axis_signature(window) == simple_signature
    assert heavy_signature != simple_signature
    assert all(len(ax.lines) > 0 for ax in (window.ax_fbd, window.ax_V, window.ax_M, window.ax_defl))
    assert all(
        any(isinstance(text, Annotation) and text.get_text() for text in ax.texts)
        for ax in (window.ax_fbd, window.ax_V, window.ax_M, window.ax_defl)
    )

    QTest.qWait(300)
    assert not window._redraw_timer.isActive()
    assert not window.canvas._resize_draw_timer.isActive()
    assert not tab.section_panel._timer.isActive()
    window.close()


def test_main_replot_axes_remain_exportable(tmp_path):
    app = _app()
    window, tab = _prepare_window()
    tab.set_cache(_cache(heavy=True))
    window._replot_active_tab()
    app.processEvents()

    for name, ax in (
        ("fbd", window.ax_fbd),
        ("v", window.ax_V),
        ("m", window.ax_M),
        ("deflection", window.ax_defl),
    ):
        output = tmp_path / f"{name}.jpg"
        window._save_axis_snapshot(ax, str(output), dpi=120)
        assert output.stat().st_size > 0

    window.close()
