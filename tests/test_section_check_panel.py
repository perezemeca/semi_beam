import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QApplication, QAbstractItemView, QSizePolicy

from semi_beam.ui.section_check_panel import (
    FLEX_NO_DEMAND_TEXT,
    BASTIDOR_LATERAL_DISTANCIA_MM,
    FRAME_REINFORCEMENT_OFFSET_FROM_BASTIDOR_MM,
    SectionCheckPanel,
    build_web_rects,
)


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


def _is_editable(panel: SectionCheckPanel, row: int, col: int) -> bool:
    item = panel.tbl.item(row, col)
    assert item is not None
    return bool(item.flags() & Qt.ItemIsEditable)


def test_section_panel_editable_flags_survive_refresh_and_clear_results():
    _app()
    panel = SectionCheckPanel()
    expected_triggers = (
        QAbstractItemView.DoubleClicked
        | QAbstractItemView.EditKeyPressed
        | QAbstractItemView.AnyKeyPressed
        | QAbstractItemView.SelectedClicked
    )

    assert panel.tbl.editTriggers() == expected_triggers
    assert _is_editable(panel, 0, panel.COL_X) is True
    assert _is_editable(panel, 0, panel.COL_HWEB) is True
    assert _is_editable(panel, 0, panel.COL_DOUBLE_WEB_OFFSET) is False
    assert _is_editable(panel, 0, panel.COL_SEC) is False
    assert _is_editable(panel, 0, panel.COL_M) is False
    assert _is_editable(panel, 0, panel.COL_FS) is False

    panel._double_web_widgets[0].setChecked(True)
    assert _is_editable(panel, 0, panel.COL_DOUBLE_WEB_OFFSET) is True

    panel.set_moment_provider(lambda _x_mm: 125000.0)
    panel.tbl.item(0, panel.COL_X).setText("1000")
    panel.tbl.item(0, panel.COL_HWEB).setText("450")
    panel.refresh_results_from_context()

    assert panel.tbl.editTriggers() == expected_triggers
    assert _cell(panel, 0, panel.COL_X) == "1000"
    assert _cell(panel, 0, panel.COL_HWEB) == "450"
    assert _cell(panel, 0, panel.COL_DOUBLE_WEB_OFFSET) == ""
    assert _is_editable(panel, 0, panel.COL_X) is True
    assert _is_editable(panel, 0, panel.COL_HWEB) is True
    assert _is_editable(panel, 0, panel.COL_DOUBLE_WEB_OFFSET) is True
    assert _is_editable(panel, 0, panel.COL_M) is False
    assert _is_editable(panel, 0, panel.COL_FS) is False
    assert _is_editable(panel, 0, panel.COL_JX) is False

    panel.clear_results_only()

    assert panel.tbl.editTriggers() == expected_triggers
    assert _cell(panel, 0, panel.COL_X) == "1000"
    assert _cell(panel, 0, panel.COL_HWEB) == "450"
    assert _cell(panel, 0, panel.COL_DOUBLE_WEB_OFFSET) == ""
    assert _is_editable(panel, 0, panel.COL_X) is True
    assert _is_editable(panel, 0, panel.COL_HWEB) is True
    assert _is_editable(panel, 0, panel.COL_DOUBLE_WEB_OFFSET) is True
    assert _is_editable(panel, 0, panel.COL_M) is False
    assert _is_editable(panel, 0, panel.COL_FS) is False


def test_section_panel_defers_refresh_while_user_is_editing_input_cell(monkeypatch):
    _app()
    panel = SectionCheckPanel()
    panel._double_web_widgets[0].setChecked(True)
    panel.tbl.item(0, panel.COL_X).setText("1000")
    panel.tbl.item(0, panel.COL_HWEB).setText("450")
    panel.tbl.item(0, panel.COL_DOUBLE_WEB_OFFSET).setText("20")
    panel.set_moment_provider(lambda _x_mm: 125000.0)
    panel.tbl.item(0, panel.COL_M).setText("111")
    panel.tbl.setCurrentCell(0, panel.COL_X)
    monkeypatch.setattr(panel, "_is_user_editing_input_cell", lambda: True)

    panel.refresh_results_from_context()

    assert panel._pending_recompute_after_edit is True
    assert _cell(panel, 0, panel.COL_X) == "1000"
    assert _cell(panel, 0, panel.COL_HWEB) == "450"
    assert _cell(panel, 0, panel.COL_DOUBLE_WEB_OFFSET) == "20"
    assert _cell(panel, 0, panel.COL_M) == "111"
    assert _cell(panel, 0, panel.COL_FS) == ""

    monkeypatch.setattr(panel, "_is_user_editing_input_cell", lambda: False)
    panel._flush_pending_recompute_after_edit()

    assert panel._pending_recompute_after_edit is False
    assert _cell(panel, 0, panel.COL_X) == "1000"
    assert _cell(panel, 0, panel.COL_HWEB) == "450"
    assert _cell(panel, 0, panel.COL_DOUBLE_WEB_OFFSET) == "20"
    assert _cell(panel, 0, panel.COL_M) == "125000"
    assert _cell(panel, 0, panel.COL_FS)
    assert "ERR" not in _cell(panel, 0, panel.COL_FS)


def test_section_panel_thickness_combo_uses_cell_width_policy():
    _app()
    panel = SectionCheckPanel()
    combo = panel._tweb_widgets[0]

    assert combo.minimumWidth() == 0
    assert combo.maximumWidth() > 100000
    assert combo.sizePolicy().horizontalPolicy() == QSizePolicy.Expanding
    assert combo.currentData() == "1/4"


def test_section_panel_frame_reinforcement_options_are_available_and_disabled_by_default():
    _app()
    panel = SectionCheckPanel()

    options = {
        panel.cmb_frame_reinforcement_thickness.itemData(i)
        for i in range(panel.cmb_frame_reinforcement_thickness.count())
    }

    assert panel.chk_frame_reinforcement.text() == "Refuerzo de bastidor"
    assert options == {"3/16", "1/4", "5/16", "3/8"}
    assert panel.chk_frame_reinforcement.isChecked() is False
    assert panel.cmb_frame_reinforcement_thickness.isEnabled() is False

    panel.chk_frame_reinforcement.setChecked(True)

    assert panel.cmb_frame_reinforcement_thickness.isEnabled() is True


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
    panel.n_chapon_length.setValue(0.0)
    _load_valid_row(panel)

    panel._recompute_all()
    cards = panel._collect_section_export_cards()

    assert _cell(panel, 0, panel.COL_FS) == "ERR CHAPÓN"
    assert cards
    card = cards[0]
    assert card["chapon_context_error"] is True
    assert "chapon_length_mm" in card["chapon_context_missing_fields"]
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
    assert panel.chk_frame_reinforcement.isChecked() is False
    assert panel.cmb_frame_reinforcement_thickness.currentData() == "1/4"


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


def test_section_panel_export_import_preserves_frame_reinforcement_state():
    _app()
    panel = SectionCheckPanel()
    panel.chk_frame_reinforcement.setChecked(True)
    idx = panel.cmb_frame_reinforcement_thickness.findData("3/8")
    assert idx >= 0
    panel.cmb_frame_reinforcement_thickness.setCurrentIndex(idx)

    state = panel.export_state()
    restored = SectionCheckPanel()
    restored.import_state(state)

    assert state["frame_reinforcement_enabled"] is True
    assert state["frame_reinforcement_thickness_in"] == "3/8"
    assert "inner_web_enabled" not in state
    assert restored.chk_frame_reinforcement.isChecked() is True
    assert restored.cmb_frame_reinforcement_thickness.currentData() == "3/8"


def test_section_panel_import_accepts_legacy_inner_web_state():
    _app()
    panel = SectionCheckPanel()

    panel.import_state({"inner_web_enabled": True, "inner_web_thickness_in": "3/8", "rows": []})

    assert panel.chk_frame_reinforcement.isChecked() is True
    assert panel.cmb_frame_reinforcement_thickness.currentData() == "3/8"


def test_section_panel_frame_reinforcement_duplicates_bastidor_vertical_rect_and_changes_properties():
    _app()
    panel = SectionCheckPanel()
    panel.chk_bastidor_lateral.setChecked(True)
    base = panel._make_section(450.0, 0.25)

    panel.chk_frame_reinforcement.setChecked(True)
    idx = panel.cmb_frame_reinforcement_thickness.findData("5/16")
    assert idx >= 0
    panel.cmb_frame_reinforcement_thickness.setCurrentIndex(idx)
    reinforced = panel._make_section(450.0, 0.25)

    assert hasattr(reinforced, "rects")
    rects = [rect for rect in reinforced.rects if rect.label.endswith("refuerzo")]
    bastidor_rects = [
        rect
        for rect in reinforced.rects
        if rect.label.startswith("bastidor_") and rect.label.endswith("alma")
    ]
    assert len(rects) == 2
    assert len(bastidor_rects) == 2
    t = float(rects[0].b_mm)
    assert rects[0].h_mm == bastidor_rects[0].h_mm == panel.n_bastidor_lateral_altura.value()
    assert rects[1].h_mm == bastidor_rects[1].h_mm == panel.n_bastidor_lateral_altura.value()
    assert rects[0].x0_mm + rects[0].b_mm / 2.0 == -BASTIDOR_LATERAL_DISTANCIA_MM + FRAME_REINFORCEMENT_OFFSET_FROM_BASTIDOR_MM
    assert rects[1].x0_mm + rects[1].b_mm / 2.0 == BASTIDOR_LATERAL_DISTANCIA_MM - FRAME_REINFORCEMENT_OFFSET_FROM_BASTIDOR_MM
    assert t == rects[1].b_mm
    assert reinforced.props_mm()["Ix_mm4"] > base.props_mm()["Ix_mm4"]


def test_section_panel_frame_reinforcement_updates_fs_and_export_payload():
    _app()
    panel = SectionCheckPanel()
    panel.set_moment_provider(lambda _x_mm: 125000.0)
    panel.chk_bastidor_lateral.setChecked(True)
    _load_valid_row(panel)
    panel._recompute_all()
    base_jx = float(_cell(panel, 0, panel.COL_JX))
    base_fs = float(_cell(panel, 0, panel.COL_FS))

    panel.chk_frame_reinforcement.setChecked(True)
    panel._recompute_all()
    reinforced_jx = float(_cell(panel, 0, panel.COL_JX))
    reinforced_fs = float(_cell(panel, 0, panel.COL_FS))
    cards = panel._collect_section_export_cards()

    assert reinforced_jx > base_jx
    assert reinforced_fs > base_fs
    assert cards[0]["frame_reinforcement_enabled"] is True
    assert cards[0]["frame_reinforcement_offset_from_bastidor_mm"] == 40.0
    assert cards[0]["frame_reinforcement_thickness_in"] == "1/4"
    assert cards[0]["web_configuration_label"] == "Simple + refuerzo de bastidor"


def test_section_panel_export_import_preserves_floor_width_without_structural_flag():
    _app()
    panel = SectionCheckPanel()
    panel.chk_piso.setChecked(True)
    panel.n_piso_width.setValue(1850.0)

    state = panel.export_state()

    assert state["include_piso"] is True
    assert state["piso_width_mm"] == 1850.0
    assert "piso_structural" not in state

    restored = SectionCheckPanel()
    restored.import_state({**state, "piso_structural": False})

    assert restored.chk_piso.isChecked() is True
    assert restored.n_piso_width.value() == 1850.0
    assert "piso_structural" not in restored.export_state()


def test_section_panel_import_uses_default_floor_width_when_missing():
    _app()
    panel = SectionCheckPanel()

    panel.import_state({"include_piso": True, "piso_structural": False})

    assert panel.chk_piso.isChecked() is True
    assert panel.n_piso_width.value() == 2430.0
    sec = panel._make_section(450.0, 0.25)
    piso_rects = [rect for rect in sec.rects if rect.label == "piso"]
    assert len(piso_rects) == 1
    assert piso_rects[0].b_mm == 2430.0


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


def test_section_panel_valid_double_web_calculates_fs():
    _app()
    panel = SectionCheckPanel()
    panel.set_moment_provider(lambda _x_mm: 125000.0)
    panel._double_web_widgets[0].setChecked(True)
    panel.tbl.item(0, panel.COL_DOUBLE_WEB_OFFSET).setText("20")
    panel.tbl.item(0, panel.COL_X).setText("1000")
    panel.tbl.item(0, panel.COL_HWEB).setText("450")

    panel._recompute_all()

    fs_text = _cell(panel, 0, panel.COL_FS)
    assert fs_text
    assert "ERR" not in fs_text
    assert float(fs_text) > 0.0


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
