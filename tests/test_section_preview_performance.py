import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication

from semi_beam.ui.section_check_panel import SectionCheckPanel


def _app() -> QApplication:
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


def _set_row_geometry(panel: SectionCheckPanel, row: int, *, x_mm: float, h_web_mm: float) -> None:
    panel.tbl.blockSignals(True)
    try:
        panel.tbl.item(row, panel.COL_X).setText(str(x_mm))
        panel.tbl.item(row, panel.COL_HWEB).setText(str(h_web_mm))
    finally:
        panel.tbl.blockSignals(False)


def _select_and_render(panel: SectionCheckPanel, row: int, column: int) -> None:
    panel.tbl.setCurrentCell(row, column)
    panel._repaint_preview_from_selection()
    QApplication.processEvents()
    panel._timer.stop()


def _renderer_spy(panel: SectionCheckPanel):
    calls = []
    original = panel._draw_section_preview_on

    def draw(ax, section) -> None:
        calls.append(section)
        original(ax, section)

    panel._draw_section_preview_on = draw
    return calls


def test_same_row_cell_navigation_does_not_redraw_identical_preview():
    _app()
    panel = SectionCheckPanel()
    _set_row_geometry(panel, 0, x_mm=1000, h_web_mm=220)
    _select_and_render(panel, 0, panel.COL_X)
    signature = panel._last_preview_signature
    renders = _renderer_spy(panel)

    for column in (
        panel.COL_HWEB,
        panel.COL_FS,
        panel.COL_M,
        panel.COL_JX,
        panel.COL_YBAR,
        panel.COL_CMAX,
        panel.COL_WCRIT,
        panel.COL_WREQ,
        panel.COL_SIGMAX,
        panel.COL_X,
    ):
        panel.tbl.setCurrentCell(0, column)
        QApplication.processEvents()

    assert panel.tbl.currentRow() == 0
    assert panel.tbl.currentColumn() == panel.COL_X
    assert panel._last_preview_signature == signature
    assert renders == []
    panel.close()


def test_different_row_with_different_geometry_redraws_once():
    _app()
    panel = SectionCheckPanel()
    _set_row_geometry(panel, 0, x_mm=1000, h_web_mm=220)
    _set_row_geometry(panel, 1, x_mm=1000, h_web_mm=360)
    _select_and_render(panel, 0, panel.COL_HWEB)
    first_signature = panel._last_preview_signature
    renders = _renderer_spy(panel)

    panel.tbl.setCurrentCell(1, panel.COL_HWEB)
    QApplication.processEvents()

    assert len(renders) == 1
    assert panel._last_preview_signature != first_signature
    assert renders[0].h_web_mm == 360
    panel.close()


def test_geometric_edit_invalidates_preview_signature():
    _app()
    panel = SectionCheckPanel()
    _set_row_geometry(panel, 0, x_mm=1000, h_web_mm=220)
    _select_and_render(panel, 0, panel.COL_HWEB)
    first_signature = panel._last_preview_signature
    renders = _renderer_spy(panel)

    panel.tbl.item(0, panel.COL_HWEB).setText("350")
    QApplication.processEvents()

    assert len(renders) == 1
    assert panel._last_preview_signature != first_signature
    assert renders[0].h_web_mm == 350
    panel._timer.stop()
    panel.close()


def test_material_edit_does_not_redraw_unchanged_geometry():
    _app()
    panel = SectionCheckPanel()
    _set_row_geometry(panel, 0, x_mm=1000, h_web_mm=220)
    _select_and_render(panel, 0, panel.COL_HWEB)
    signature = panel._last_preview_signature
    renders = _renderer_spy(panel)
    combo = panel.cmb_mat_web

    assert combo.count() > 1
    combo.setCurrentIndex((combo.currentIndex() + 1) % combo.count())
    QApplication.processEvents()

    assert panel._last_preview_signature == signature
    assert renders == []
    panel._timer.stop()
    panel.close()


def test_station_edit_redraws_when_chapon_inclusion_changes():
    _app()
    panel = SectionCheckPanel()
    panel.chk_chapon.setChecked(True)
    _set_row_geometry(panel, 0, x_mm=1000, h_web_mm=220)
    _select_and_render(panel, 0, panel.COL_X)
    first_signature = panel._last_preview_signature
    renders = _renderer_spy(panel)

    panel.tbl.item(0, panel.COL_X).setText("3000")
    QApplication.processEvents()

    assert len(renders) == 1
    assert panel._last_preview_signature != first_signature
    panel._timer.stop()
    panel.close()


def test_fast_navigation_renders_only_distinct_final_geometry_and_reaches_idle():
    _app()
    panel = SectionCheckPanel()
    _set_row_geometry(panel, 0, x_mm=1000, h_web_mm=220)
    _set_row_geometry(panel, 1, x_mm=1000, h_web_mm=220)
    _set_row_geometry(panel, 2, x_mm=1000, h_web_mm=360)
    _select_and_render(panel, 0, panel.COL_X)
    renders = _renderer_spy(panel)

    for step in range(9):
        panel.tbl.setCurrentCell(step % 2, panel.COL_HWEB if step % 2 else panel.COL_FS)
        QApplication.processEvents()
    panel.tbl.setCurrentCell(2, panel.COL_HWEB)
    QApplication.processEvents()

    assert panel.tbl.currentRow() == 2
    assert len(renders) == 1
    assert renders[0].h_web_mm == 360

    QTest.qWait(180)
    assert not panel._timer.isActive()
    settled_renders = len(renders)
    QTest.qWait(180)
    assert len(renders) == settled_renders
    assert not panel._timer.isActive()
    panel.close()
