import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtTest import QSignalSpy, QTest
from PySide6.QtWidgets import QApplication

from semi_beam.ui.main_window import FBDApp


def _app() -> QApplication:
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


def _stop_background_timers(window: FBDApp) -> None:
    window._redraw_timer.stop()
    window._resize_timer.stop()
    window.tab_reactions._timer.stop()
    for tab in (window.tab_acoplado, window.tab_semi, window.tab_bitren, window.tab_reactions):
        tab.section_panel._timer.stop()


def test_resize_burst_has_one_final_replot_and_reaches_idle():
    app = _app()
    window = FBDApp()
    window.show()
    QTest.qWait(250)
    _stop_background_timers(window)

    tab = window.tab_acoplado
    tab.Lc.setValue(13600.0)
    tab._add_load_row(load_type="Puntual")
    tab.tbl.item(0, tab.COL_MAG).setText("5000")
    tab.tbl.item(0, tab.COL_POS).setText("2500")
    window._plot_inputs_for_tab(tab)
    QTest.qWait(100)
    _stop_background_timers(window)

    replots = 0
    section_checks = 0
    canvas_resize_events = 0
    active_panel = tab.section_panel
    original_replot = window._replot_active_tab
    original_section_check = active_panel._recompute_all

    def final_replot() -> None:
        nonlocal replots
        replots += 1
        original_replot()

    def section_check() -> None:
        nonlocal section_checks
        section_checks += 1
        original_section_check()

    def canvas_resized(_event) -> None:
        nonlocal canvas_resize_events
        canvas_resize_events += 1

    window._resize_timer.timeout.disconnect()
    window._resize_timer.timeout.connect(final_replot)
    active_panel._timer.timeout.disconnect()
    active_panel._timer.timeout.connect(section_check)
    window.canvas.mpl_connect("resize_event", canvas_resized)

    start_width = window.width()
    final_width = start_width
    resize_snapshot_seen = False
    for step in range(20):
        final_width = start_width + 20 + step * 8
        window.resize(final_width, window.height())
        resize_snapshot_seen = resize_snapshot_seen or window.canvas._resize_snapshot is not None
        app.processEvents()

    assert canvas_resize_events > 0
    assert resize_snapshot_seen
    QTest.qWait(600)

    assert replots == 1
    assert section_checks <= 1
    assert window.width() == final_width
    assert round(window.fig.get_figwidth() * window.fig.dpi) == round(
        window.canvas.width() * window.canvas.device_pixel_ratio
    )
    assert not window.canvas._resize_draw_timer.isActive()
    assert not window.canvas._resize_draw_pending
    assert window.canvas._resize_snapshot is None
    assert not window._resize_timer.isActive()
    assert not active_panel._timer.isActive()
    assert window.ax_fbd.texts or window.ax_fbd.lines or window.ax_fbd.patches
    assert window.ax_V.lines
    assert window.ax_M.lines
    assert window.ax_defl.texts or window.ax_defl.lines

    settled_counts = (replots, section_checks)
    QTest.qWait(200)
    assert (replots, section_checks) == settled_counts
    assert not window.canvas._resize_draw_timer.isActive()
    assert not window._resize_timer.isActive()
    assert not active_panel._timer.isActive()
    window.close()


def test_functional_draw_cancels_pending_resize_flush():
    _app()
    window = FBDApp()
    canvas = window.canvas
    resize_flushes = QSignalSpy(canvas._resize_draw_timer.timeout)

    canvas._handling_resize_event = True
    for _ in range(10):
        canvas.draw_idle()
    canvas._handling_resize_event = False

    assert canvas._resize_draw_timer.isActive()
    canvas.draw_idle()
    assert not canvas._resize_draw_timer.isActive()

    QTest.qWait(canvas.RESIZE_DRAW_INTERVAL_MS + 20)
    assert resize_flushes.count() == 0
    window.close()
