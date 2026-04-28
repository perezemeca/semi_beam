import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication

from semi_beam.ui.section_check_panel import FLEX_NO_DEMAND_TEXT, SectionCheckPanel, build_web_rects


def _app():
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


def _cell(panel: SectionCheckPanel, row: int, col: int) -> str:
    item = panel.tbl.item(row, col)
    return "" if item is None else item.text()


def _load_valid_row(panel: SectionCheckPanel, *, row: int = 0, x_mm: str = "1000", h_web_mm: str = "450", moment: str = "125000") -> None:
    panel.tbl.item(row, panel.COL_X).setText(x_mm)
    panel.tbl.item(row, panel.COL_HWEB).setText(h_web_mm)
    panel.tbl.item(row, panel.COL_M).setText(moment)


def test_section_panel_zero_moment_shows_no_flex_demand_without_chapon_error():
    _app()
    panel = SectionCheckPanel()
    panel.chk_chapon.setChecked(True)
    _load_valid_row(panel, moment="0")

    panel._recompute_all()

    fs_text = _cell(panel, 0, panel.COL_FS)
    assert fs_text == FLEX_NO_DEMAND_TEXT
    assert _cell(panel, 0, panel.COL_WREQ) == "0"
    assert _cell(panel, 0, panel.COL_SIGMAX) == "0"
    assert "ERR" not in fs_text
    assert "inf" not in fs_text.lower()
    assert "nan" not in fs_text.lower()
    assert not fs_text.replace(".", "", 1).isdigit()


def test_section_panel_composite_payload_includes_table_values():
    _app()
    panel = SectionCheckPanel()
    panel.set_beam_context(largo_viga_mm=5000.0, posicion_perno_mm=1000.0)
    panel.chk_bastidor_lateral.setChecked(True)
    panel.chk_piso.setChecked(True)
    panel.chk_chapon.setChecked(True)
    _load_valid_row(panel, x_mm="1500")

    panel._recompute_all()
    cards = panel._collect_section_export_cards()

    assert cards
    table_values = cards[0]["table_values"]
    assert table_values == {
        "FS": _cell(panel, 0, panel.COL_FS),
        "Jx_cm4": _cell(panel, 0, panel.COL_JX),
        "ybar_cm": _cell(panel, 0, panel.COL_YBAR),
        "cmax_cm": _cell(panel, 0, panel.COL_CMAX),
        "Wcrit_cm3": _cell(panel, 0, panel.COL_WCRIT),
        "Wreq_cm3": _cell(panel, 0, panel.COL_WREQ),
        "sigma_max_kgcm2": _cell(panel, 0, panel.COL_SIGMAX),
    }


def test_section_panel_err_chapon_when_context_missing():
    _app()
    panel = SectionCheckPanel()
    panel.chk_chapon.setChecked(True)
    _load_valid_row(panel)

    panel._recompute_all()
    cards = panel._collect_section_export_cards()

    assert _cell(panel, 0, panel.COL_FS) == "ERR CHAPÓN"
    assert cards
    card = cards[0]
    assert card["chapon_context_error"] is True
    assert {
        "largo_viga_mm",
        "posicion_perno_mm",
    }.intersection(card["chapon_context_missing_fields"])
    assert card["ok"] is False
    assert card["fs_text"] == "ERR CHAPÓN"


def test_section_panel_err_mat_when_component_material_missing(monkeypatch):
    _app()
    panel = SectionCheckPanel()
    original_material_sigma_by_id = panel._material_sigma_by_id

    def fake_material_sigma_by_id(mat_id: str):
        if str(mat_id or "").strip() == "SAE1010":
            return None
        return original_material_sigma_by_id(mat_id)

    monkeypatch.setattr(panel, "_material_sigma_by_id", fake_material_sigma_by_id)
    panel.chk_bastidor_lateral.setChecked(True)
    _load_valid_row(panel)

    panel._recompute_all()
    cards = panel._collect_section_export_cards()

    assert _cell(panel, 0, panel.COL_FS) == "ERR MAT"
    assert cards
    card = cards[0]
    assert card["material_error"] is True
    assert card["missing_material_components"]
    assert card["ok"] is False
    assert card["fs_text"] == "ERR MAT"


def test_double_web_helper_builds_symmetric_web_rects():
    rects = build_web_rects(
        b_f_mm=127.0,
        t_web_mm=6.35,
        h_web_mm=450.0,
        y0_mm=12.7,
        double_web_enabled=True,
        double_web_inner_face_offset_mm=20.0,
    )

    assert len(rects) == 2
    assert rects[0].x0_mm == -26.35
    assert rects[0].x0_mm + rects[0].b_mm == -20.0
    assert rects[1].x0_mm == 20.0
    assert rects[1].x0_mm + rects[1].b_mm == 26.35


def test_section_panel_imports_old_rows_without_double_web_fields():
    _app()
    panel = SectionCheckPanel()

    panel.import_state({"rows": [{"x_mm": "1000", "h_web_mm": "450", "t_web_in": "1/4"}]})

    assert panel._row_double_web_enabled(0) is False
    assert panel._row_double_web_offset_mm(0) is None


def test_section_panel_export_import_preserves_double_web_row_state():
    _app()
    panel = SectionCheckPanel()
    panel._double_web_widgets[0].setChecked(True)
    panel.tbl.item(0, panel.COL_DOUBLE_WEB_OFFSET).setText("20")

    state = panel.export_state()
    restored = SectionCheckPanel()
    restored.import_state(state)

    assert state["rows"][0]["double_web_enabled"] is True
    assert state["rows"][0]["double_web_inner_face_offset_mm"] == 20.0
    assert restored._row_double_web_enabled(0) is True
    assert restored._row_double_web_offset_mm(0) == 20.0


def test_section_panel_invalid_double_web_returns_error_state():
    _app()
    panel = SectionCheckPanel()
    panel.set_moment_provider(lambda _x_mm: 125000.0)
    panel._double_web_widgets[0].setChecked(True)
    panel.tbl.item(0, panel.COL_DOUBLE_WEB_OFFSET).setText("60")
    panel.tbl.item(0, panel.COL_X).setText("1000")
    panel.tbl.item(0, panel.COL_HWEB).setText("450")

    panel._recompute_all()
    cards = panel._collect_section_export_cards()

    assert _cell(panel, 0, panel.COL_FS) == "ERR DOBLE ALMA"
    assert cards[0]["fs_text"] == "ERR DOBLE ALMA"
    assert cards[0]["ok"] is False
    assert cards[0]["double_web_error"]


def test_section_panel_double_web_disabled_preserves_existing_results():
    _app()
    base = SectionCheckPanel()
    base.set_moment_provider(lambda _x_mm: 125000.0)
    _load_valid_row(base)
    base._recompute_all()

    disabled = SectionCheckPanel()
    disabled.set_moment_provider(lambda _x_mm: 125000.0)
    _load_valid_row(disabled)
    disabled.tbl.item(0, disabled.COL_DOUBLE_WEB_OFFSET).setText("60")
    disabled._recompute_all()

    assert _cell(disabled, 0, disabled.COL_FS) == _cell(base, 0, base.COL_FS)
    assert _cell(disabled, 0, disabled.COL_JX) == _cell(base, 0, base.COL_JX)
    assert _cell(disabled, 0, disabled.COL_WCRIT) == _cell(base, 0, base.COL_WCRIT)
