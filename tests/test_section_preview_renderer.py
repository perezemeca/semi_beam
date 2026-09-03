import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication
from matplotlib.patches import FancyArrowPatch, Rectangle

from semi_beam.sections.i_section import CompositeSection, ISection
from semi_beam.ui.section_check_panel import SectionCheckPanel


def _app() -> QApplication:
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


def _artist_counts(panel: SectionCheckPanel) -> tuple[int, int, int]:
    return len(panel.ax.patches), len(panel.ax.lines), len(panel.ax.texts)


def _make_composite(panel: SectionCheckPanel) -> CompositeSection:
    panel.chk_bastidor_lateral.setChecked(True)
    panel.chk_piso.setChecked(True)
    panel.chk_chapon.setChecked(True)
    panel.n_chapon_length.setValue(2000.0)
    panel.chk_frame_reinforcement.setChecked(True)
    section = panel._make_section(
        360.0,
        0.25,
        station_mm=1000.0,
        double_web_enabled=True,
        double_web_inner_face_offset_mm=20.0,
    )
    assert isinstance(section, CompositeSection)
    return section


def test_preview_rebuild_removes_artists_across_topology_changes():
    _app()
    panel = SectionCheckPanel()
    simple = panel._make_section(220.0, 0.25)
    assert isinstance(simple, ISection)

    panel._draw_section_preview_on(panel.ax, simple)
    assert _artist_counts(panel) == (5, 4, 2)
    assert sum(isinstance(patch, Rectangle) for patch in panel.ax.patches) == 3
    assert sum(isinstance(patch, FancyArrowPatch) for patch in panel.ax.patches) == 2

    composite = _make_composite(panel)
    panel._draw_section_preview_on(panel.ax, composite)
    assert _artist_counts(panel) == (len(composite.rects) + 1, 2, 1)
    assert sum(isinstance(patch, Rectangle) for patch in panel.ax.patches) == len(composite.rects)
    assert sum(isinstance(patch, FancyArrowPatch) for patch in panel.ax.patches) == 1

    panel.chk_bastidor_lateral.setChecked(False)
    panel.chk_piso.setChecked(False)
    panel.chk_chapon.setChecked(False)
    panel.chk_frame_reinforcement.setChecked(False)
    panel._draw_section_preview_on(panel.ax, simple)
    assert _artist_counts(panel) == (5, 4, 2)
    assert all(text.get_text() for text in panel.ax.texts)
    panel._timer.stop()
    panel.close()


def test_preview_error_state_does_not_leave_or_retain_stale_artists():
    _app()
    panel = SectionCheckPanel()
    panel.tbl.setCurrentCell(0, panel.COL_HWEB)
    checkbox = panel._double_web_widgets[0]
    checkbox.blockSignals(True)
    checkbox.setChecked(True)
    checkbox.blockSignals(False)
    panel.tbl.blockSignals(True)
    panel.tbl.item(0, panel.COL_DOUBLE_WEB_OFFSET).setText("1000")
    panel.tbl.blockSignals(False)
    panel._last_preview_signature = None

    panel._repaint_preview(h_web_override_mm=220.0, t_web_in=0.25)
    assert _artist_counts(panel) == (0, 0, 1)
    assert panel.ax.texts[0].get_text() == "ERR DOBLE ALMA"

    checkbox.blockSignals(True)
    checkbox.setChecked(False)
    checkbox.blockSignals(False)
    panel._repaint_preview(h_web_override_mm=220.0, t_web_in=0.25)
    assert _artist_counts(panel) == (5, 4, 2)
    assert all(text.get_text() != "ERR DOBLE ALMA" for text in panel.ax.texts)
    panel._timer.stop()
    panel.close()
