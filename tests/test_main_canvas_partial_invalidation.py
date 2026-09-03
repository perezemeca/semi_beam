import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication

from semi_beam.domain.beam import Beam
from semi_beam.domain.loads import DistUniform, PointForce, PointMoment
import semi_beam.ui.main_window as main_window
from semi_beam.ui.main_window import FBDApp, SessionCache


def _app() -> QApplication:
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


def _cache() -> SessionCache:
    return SessionCache(
        beam_plot=Beam(L_mm=12000.0),
        points=[
            PointForce("Rp1", 1000.0, 9000.0),
            PointForce("P1", 2200.0, 3500.0),
            PointForce("P2", 4300.0, 4200.0),
            PointForce("Rt", 11200.0, 12500.0),
        ],
        dists=[DistUniform("q1", 5200.0, 1200.0, 3.5)],
        moms=[PointMoment("M1", 3000.0, 250000.0)],
        note_text="test",
        deflection_supports=(1000.0, 11000.0),
    )


def _prepare() -> tuple[QApplication, FBDApp, object]:
    app = _app()
    window = FBDApp()
    window.show()
    app.processEvents()
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
    tab.set_cache(_cache())
    tab.set_view_mode("solved")
    window._replot_active_tab()
    app.processEvents()
    QTest.qWait(300)
    panel._timer.stop()
    window._redraw_timer.stop()
    window._resize_timer.stop()
    window.canvas._resize_draw_timer.stop()
    return app, window, tab


def _artist_ids(ax) -> tuple[int, ...]:
    return tuple(id(artist) for artist in (*ax.lines, *ax.patches, *ax.collections, *ax.texts))


def _replace_section_timer_with_counter(panel):
    original = panel._recompute_all
    calls = []

    def recompute():
        calls.append(True)
        original()

    panel._timer.timeout.disconnect()
    panel._timer.timeout.connect(recompute)
    return calls


def test_section_geometry_only_rerenders_deflection_and_preserves_other_axes(monkeypatch):
    _app_instance, window, tab = _prepare()
    panel = tab.section_panel
    hover = window._diagram_hover
    v_curve = next(curve for curve in hover.curves if curve.ax is window.ax_V)
    deflection_curve = next(curve for curve in hover.curves if curve.ax is window.ax_defl)
    v_marker, v_annotation = hover._ensure_artists(window.ax_V, v_curve)
    hover._ensure_artists(window.ax_defl, deflection_curve)
    preserved = tuple(_artist_ids(ax) for ax in (window.ax_fbd, window.ax_V, window.ax_M))
    initial_deflection = np.asarray(window.ax_defl.lines[0].get_ydata(), dtype=float).copy()
    calls = {"plot": 0, "fbd": 0, "v": 0, "m": 0, "deflection": 0}

    original_plot = window._plot_triplet

    def plot(*args, **kwargs):
        calls["plot"] += 1
        return original_plot(*args, **kwargs)

    monkeypatch.setattr(window, "_plot_triplet", plot)
    for name, key in (
        ("render_fbd", "fbd"),
        ("render_shear", "v"),
        ("render_moment", "m"),
        ("render_deflection", "deflection"),
    ):
        original = getattr(main_window, name)

        def render(*args, _original=original, _key=key, **kwargs):
            calls[_key] += 1
            return _original(*args, **kwargs)

        monkeypatch.setattr(main_window, name, render)

    section_checks = _replace_section_timer_with_counter(panel)
    draws = []
    connection_id = window.canvas.mpl_connect("draw_event", lambda _event: draws.append(True))

    panel.tbl.item(0, panel.COL_HWEB).setText("520")
    QTest.qWait(220)

    assert calls == {"plot": 0, "fbd": 0, "v": 0, "m": 0, "deflection": 1}
    assert len(section_checks) == 1
    assert len(draws) == 1
    assert tuple(_artist_ids(ax) for ax in (window.ax_fbd, window.ax_V, window.ax_M)) == preserved
    assert not np.allclose(np.asarray(window.ax_defl.lines[0].get_ydata(), dtype=float), initial_deflection)
    assert window._diagram_hover is hover
    assert hover._markers[window.ax_V] is v_marker
    assert hover._annotations[window.ax_V] is v_annotation
    assert window.ax_defl not in hover._markers
    assert window.ax_defl not in hover._annotations
    assert {curve.ax for curve in window._diagram_hover.curves} == {window.ax_V, window.ax_M, window.ax_defl}
    assert not window._redraw_timer.isActive()
    assert not panel._timer.isActive()
    window.canvas.mpl_disconnect(connection_id)
    window.close()


def test_section_geometry_round_trip_restores_deflection_without_residual_artists():
    _app_instance, window, tab = _prepare()
    panel = tab.section_panel
    preserved = tuple(_artist_ids(ax) for ax in (window.ax_fbd, window.ax_V, window.ax_M))
    initial_y = np.asarray(window.ax_defl.lines[0].get_ydata(), dtype=float).copy()
    initial_signature = (
        len(window.ax_defl.lines),
        len(window.ax_defl.patches),
        len(window.ax_defl.collections),
        len(window.ax_defl.texts),
    )

    panel.tbl.item(0, panel.COL_HWEB).setText("520")
    QTest.qWait(220)
    panel.tbl.item(0, panel.COL_HWEB).setText("450")
    QTest.qWait(220)

    assert tuple(_artist_ids(ax) for ax in (window.ax_fbd, window.ax_V, window.ax_M)) == preserved
    assert np.allclose(np.asarray(window.ax_defl.lines[0].get_ydata(), dtype=float), initial_y)
    assert (
        len(window.ax_defl.lines),
        len(window.ax_defl.patches),
        len(window.ax_defl.collections),
        len(window.ax_defl.texts),
    ) == initial_signature
    assert not window._redraw_timer.isActive()
    assert not panel._timer.isActive()
    window.close()


def test_section_geometry_without_valid_inertia_preserves_other_axes():
    _app_instance, window, tab = _prepare()
    panel = tab.section_panel
    preserved = tuple(_artist_ids(ax) for ax in (window.ax_fbd, window.ax_V, window.ax_M))

    panel.tbl.item(0, panel.COL_HWEB).setText("")
    QTest.qWait(220)

    assert tuple(_artist_ids(ax) for ax in (window.ax_fbd, window.ax_V, window.ax_M)) == preserved
    assert not window.ax_defl.lines
    assert any("no disponible" in text.get_text().lower() for text in window.ax_defl.texts)
    assert not window._redraw_timer.isActive()
    assert not panel._timer.isActive()
    window.close()


def test_section_geometry_without_cache_falls_back_to_full_replot(monkeypatch):
    app = _app()
    window = FBDApp()
    tab = window.tab_semi
    window.tabs.setCurrentWidget(tab)
    window._redraw_timer.stop()
    tab.section_panel._timer.stop()
    calls = []
    original = window._replot_active_tab

    def full_replot():
        calls.append(True)
        original()

    monkeypatch.setattr(window, "_replot_active_tab", full_replot)
    tab.section_panel.tbl.item(0, tab.section_panel.COL_HWEB).setText("450")
    QTest.qWait(500)
    app.processEvents()

    assert len(calls) == 1
    assert not window._redraw_timer.isActive()
    assert not tab.section_panel._timer.isActive()
    window.close()


@pytest.mark.parametrize("invalid_state", ("diag", "hover", "v_artist"))
def test_partial_refresh_falls_back_when_reusable_plot_state_is_invalid(monkeypatch, invalid_state):
    _app_instance, window, tab = _prepare()
    if invalid_state == "diag":
        tab.set_diag(None)
    elif invalid_state == "hover":
        window._disconnect_diagram_hover()
    else:
        v_curve = next(curve for curve in window._diagram_hover.curves if curve.ax is window.ax_V)
        v_curve.line.remove()

    full_calls = []
    original_full = window._plot_triplet

    def full(*args, **kwargs):
        full_calls.append(True)
        return original_full(*args, **kwargs)

    monkeypatch.setattr(window, "_plot_triplet", full)
    tab.section_panel.tbl.item(0, tab.section_panel.COL_HWEB).setText("520")
    QTest.qWait(220)

    assert len(full_calls) == 1
    assert tab.get_diag() is not None
    assert window._diagram_hover is not None
    assert not window._redraw_timer.isActive()
    window.close()


def test_load_and_global_geometry_changes_still_request_full_replot(monkeypatch):
    _app_instance, window, tab = _prepare()
    calls = {"fbd": 0, "v": 0, "m": 0}
    for name, key in (("render_fbd", "fbd"), ("render_shear", "v"), ("render_moment", "m")):
        original = getattr(main_window, name)

        def render(*args, _original=original, _key=key, **kwargs):
            calls[_key] += 1
            return _original(*args, **kwargs)

        monkeypatch.setattr(main_window, name, render)

    tab.Lc.setValue(tab.Lc.value() + 100.0)
    QTest.qWait(220)

    assert calls == {"fbd": 1, "v": 1, "m": 1}
    assert not window._redraw_timer.isActive()
    window.close()


def test_all_section_geometry_controls_use_partial_refresh(monkeypatch):
    _app_instance, window, tab = _prepare()
    panel = tab.section_panel
    panel.tbl.blockSignals(True)
    try:
        panel.tbl.item(0, panel.COL_DOUBLE_WEB_OFFSET).setText("20")
    finally:
        panel.tbl.blockSignals(False)
    partial_calls = []
    full_calls = []
    original_partial = window._refresh_deflection_only
    original_full = window._plot_triplet

    def partial(*args, **kwargs):
        partial_calls.append(True)
        return original_partial(*args, **kwargs)

    def full(*args, **kwargs):
        full_calls.append(True)
        return original_full(*args, **kwargs)

    monkeypatch.setattr(window, "_refresh_deflection_only", partial)
    monkeypatch.setattr(window, "_plot_triplet", full)
    actions = (
        lambda: panel.cmb_t_top.setCurrentIndex((panel.cmb_t_top.currentIndex() + 1) % panel.cmb_t_top.count()),
        lambda: panel._tweb_widgets[0].setCurrentIndex(
            (panel._tweb_widgets[0].currentIndex() + 1) % panel._tweb_widgets[0].count()
        ),
        lambda: panel._double_web_widgets[0].setChecked(not panel._double_web_widgets[0].isChecked()),
        lambda: panel.chk_piso.setChecked(not panel.chk_piso.isChecked()),
        lambda: panel.chk_chapon.setChecked(not panel.chk_chapon.isChecked()),
        lambda: panel.chk_frame_reinforcement.setChecked(not panel.chk_frame_reinforcement.isChecked()),
        lambda: panel.chk_bastidor_lateral.setChecked(not panel.chk_bastidor_lateral.isChecked()),
        lambda: panel.tbl.item(0, panel.COL_X).setText("3000"),
    )

    for action in actions:
        partial_calls.clear()
        full_calls.clear()
        action()
        QTest.qWait(220)
        assert len(partial_calls) == 1
        assert full_calls == []
        assert not window._redraw_timer.isActive()
        assert not panel._timer.isActive()

    window.close()


def test_partial_update_remains_exportable_and_full_resize_replot_still_works(tmp_path):
    app, window, tab = _prepare()
    panel = tab.section_panel
    panel.tbl.item(0, panel.COL_HWEB).setText("520")
    QTest.qWait(220)
    updated_deflection = np.asarray(window.ax_defl.lines[0].get_ydata(), dtype=float).copy()

    for name, ax in (
        ("fbd", window.ax_fbd),
        ("v", window.ax_V),
        ("m", window.ax_M),
        ("deflection", window.ax_defl),
    ):
        output = tmp_path / f"{name}.jpg"
        window._save_axis_snapshot(ax, str(output), dpi=120)
        assert output.stat().st_size > 0

    window.tabs.setCurrentWidget(window.tab_acoplado)
    app.processEvents()
    window.tabs.setCurrentWidget(tab)
    app.processEvents()
    assert np.allclose(np.asarray(window.ax_defl.lines[0].get_ydata(), dtype=float), updated_deflection)

    replots = []
    original = window._replot_active_tab
    window._resize_timer.timeout.disconnect()

    def full_replot():
        replots.append(True)
        original()

    window._resize_timer.timeout.connect(full_replot)
    window.resize(window.width() + 80, window.height())
    app.processEvents()
    QTest.qWait(500)

    assert len(replots) == 1
    assert all(ax.lines or ax.patches or ax.texts for ax in (window.ax_fbd, window.ax_V, window.ax_M, window.ax_defl))
    assert not window._resize_timer.isActive()
    assert not window.canvas._resize_draw_timer.isActive()
    window.close()


def test_immediate_full_replot_cancels_pending_deflection_refresh(monkeypatch):
    _app_instance, window, tab = _prepare()
    partial_calls = []
    original_partial = window._refresh_deflection_only

    def partial(*args, **kwargs):
        partial_calls.append(True)
        return original_partial(*args, **kwargs)

    monkeypatch.setattr(window, "_refresh_deflection_only", partial)
    window._schedule_deflection_replot_tab(tab)
    assert window._redraw_timer.isActive()

    window._replot_active_tab()
    QTest.qWait(220)

    assert partial_calls == []
    assert not window._redraw_timer.isActive()
    assert window._scheduled_replot_kind == "full"
    assert window._scheduled_replot_tab is None
    window.close()


def test_pending_resize_full_replot_suppresses_deflection_schedule(monkeypatch):
    _app_instance, window, tab = _prepare()
    partial_calls = []
    full_calls = []
    original_partial = window._refresh_deflection_only
    original_full = window._replot_active_tab

    def partial(*args, **kwargs):
        partial_calls.append(True)
        return original_partial(*args, **kwargs)

    def full():
        full_calls.append(True)
        original_full()

    monkeypatch.setattr(window, "_refresh_deflection_only", partial)
    window._resize_timer.timeout.disconnect()
    window._resize_timer.timeout.connect(full)
    window._resize_timer.start(120)

    window._schedule_deflection_replot_tab(tab)
    QTest.qWait(260)

    assert partial_calls == []
    assert len(full_calls) == 1
    assert not window._redraw_timer.isActive()
    assert not window._resize_timer.isActive()
    window.close()
