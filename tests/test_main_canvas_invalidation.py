import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtTest import QSignalSpy, QTest
from PySide6.QtWidgets import QApplication

from semi_beam.domain.beam import Beam
from semi_beam.domain.loads import PointForce
from semi_beam.ui.main_window import FBDApp, SessionCache


def _app() -> QApplication:
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


def test_material_change_recomputes_section_without_redrawing_main_canvas():
    app = _app()
    window = FBDApp()
    tab = window.tab_semi
    window.tabs.setCurrentWidget(tab)
    panel = tab.section_panel
    panel.tbl.blockSignals(True)
    try:
        panel.tbl.item(0, panel.COL_X).setText("1000")
        panel.tbl.item(0, panel.COL_HWEB).setText("450")
    finally:
        panel.tbl.blockSignals(False)

    tab.set_cache(
        SessionCache(
            beam_plot=Beam(L_mm=12000.0),
            points=[
                PointForce("Rp1", 1000.0, 10000.0),
                PointForce("P1", 4000.0, 15000.0),
                PointForce("Rt", 11000.0, 5000.0),
            ],
            dists=[],
            moms=[],
            note_text="test",
            deflection_supports=(1000.0, 11000.0),
        )
    )
    window._replot_active_tab()
    app.processEvents()
    panel._timer.stop()
    window._redraw_timer.stop()

    section_recomputes = QSignalSpy(panel._timer.timeout)
    main_draws = 0

    def count_draw(_event) -> None:
        nonlocal main_draws
        main_draws += 1

    connection_id = window.canvas.mpl_connect("draw_event", count_draw)
    combo = panel.cmb_mat_web
    assert combo.count() > 1
    target = (combo.currentIndex() + 1) % combo.count()
    combo.setCurrentIndex(target)

    assert panel._timer.isActive()
    assert not window._redraw_timer.isActive()
    QTest.qWait(180)

    assert combo.currentIndex() == target
    assert section_recomputes.count() == 1
    assert main_draws == 0
    assert not panel._timer.isActive()
    assert not window._redraw_timer.isActive()

    QTest.qWait(180)
    assert section_recomputes.count() == 1
    assert main_draws == 0
    window.canvas.mpl_disconnect(connection_id)
    window.close()
