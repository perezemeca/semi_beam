import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import Qt
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication

from semi_beam.ui.main_window import FBDApp
from semi_beam.ui.numeric_delegate import TABLE_ERROR_BG, TABLE_OK_BG, TABLE_READONLY_BG
from semi_beam.ui.reactions_tab import SemiTrailerReactionsTab
from semi_beam.ui.section_check_panel import SectionCheckPanel


def _app() -> QApplication:
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


def _replace_timer_callback(timer, callback) -> None:
    timer.timeout.disconnect()
    timer.timeout.connect(callback)


def test_section_panel_real_edit_recomputes_once_and_reaches_idle():
    _app()
    panel = SectionCheckPanel()
    panel.set_moment_provider(lambda _x_mm: 125000.0)
    panel.tbl.item(0, panel.COL_HWEB).setText("450")
    panel._timer.stop()

    calls = 0

    def recompute() -> None:
        nonlocal calls
        calls += 1
        panel._recompute_all()

    _replace_timer_callback(panel._timer, recompute)
    panel.tbl.item(0, panel.COL_X).setText("1000")

    assert panel._timer.isActive()
    QTest.qWait(160)
    assert calls == 1
    assert panel.tbl.item(0, panel.COL_FS).text()
    assert panel.tbl.item(0, panel.COL_FS).background().color().name() == TABLE_OK_BG.lower()
    assert not panel._timer.isActive()

    QTest.qWait(180)
    assert calls == 1
    assert not panel._timer.isActive()


def test_section_panel_internal_result_update_does_not_emit_input_changes():
    _app()
    panel = SectionCheckPanel()
    panel.set_moment_provider(lambda _x_mm: 125000.0)
    panel.tbl.item(0, panel.COL_X).setText("1000")
    panel.tbl.item(0, panel.COL_HWEB).setText("450")
    panel._timer.stop()
    cell_changes = 0
    item_changes = 0

    def count_cell_change(*_args) -> None:
        nonlocal cell_changes
        cell_changes += 1

    def count_item_change(*_args) -> None:
        nonlocal item_changes
        item_changes += 1

    panel.tbl.cellChanged.connect(count_cell_change)
    panel.tbl.itemChanged.connect(count_item_change)
    panel._recompute_all()

    assert cell_changes == 0
    assert item_changes == 0
    assert not panel._timer.isActive()


def test_reactions_tab_real_edit_recomputes_once_and_reaches_idle():
    _app()
    tab = SemiTrailerReactionsTab()
    tab.tbl.setRowCount(0)
    tab._add_load_row(load_type="Puntual")
    tab.tbl.item(0, tab.COL_MAG).setText("500")
    tab.tbl.item(0, tab.COL_POS).setText("6000")
    tab._timer.stop()

    calls = 0

    def recompute() -> None:
        nonlocal calls
        calls += 1
        tab.recompute_now()

    _replace_timer_callback(tab._timer, recompute)
    tab.tbl.item(0, tab.COL_MAG).setText("600")

    assert tab._timer.isActive()
    QTest.qWait(220)
    assert calls == 1
    assert tab._last_result is not None
    assert not tab._timer.isActive()

    QTest.qWait(220)
    assert calls == 1
    assert not tab._timer.isActive()


def test_reactions_tab_internal_row_updates_do_not_schedule_recompute():
    _app()
    tab = SemiTrailerReactionsTab()
    tab._timer.stop()
    cell_changes = 0

    def count_cell_change(*_args) -> None:
        nonlocal cell_changes
        cell_changes += 1

    tab.tbl.cellChanged.connect(count_cell_change)
    tab._apply_row_visual_state(0, has_error=True)
    assert tab.tbl.item(0, tab.COL_MAG).background().color().name() == TABLE_ERROR_BG.lower()
    tab._set_item_editable(0, tab.COL_LEN, False)

    assert cell_changes == 0
    assert not (tab.tbl.item(0, tab.COL_LEN).flags() & Qt.ItemIsEditable)
    assert tab.tbl.item(0, tab.COL_LEN).background().color().name() == TABLE_READONLY_BG.lower()
    assert not tab._timer.isActive()


def test_hidden_reactions_tab_does_not_schedule_active_tab_redraw():
    app = _app()
    window = FBDApp()
    window.tabs.setCurrentWidget(window.tab_acoplado)
    app.processEvents()
    window._redraw_timer.stop()

    window.tab_reactions.plot_data_changed.emit()
    app.processEvents()

    assert not window._redraw_timer.isActive()

    window.tabs.blockSignals(True)
    window.tabs.setCurrentWidget(window.tab_reactions)
    window.tabs.blockSignals(False)
    window.tab_reactions.plot_data_changed.emit()

    assert window._redraw_timer.isActive()
    window._redraw_timer.stop()
    window.close()
