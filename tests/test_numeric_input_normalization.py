import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication, QLineEdit

from semi_beam.domain.loads import PointForce
from semi_beam.ui.main_window import UnitTab
from semi_beam.ui.number_parsing import (
    normalize_decimal_text,
    normalize_line_edit_text,
    parse_user_float,
)
from semi_beam.ui.reactions_tab import SemiTrailerReactionsTab
from semi_beam.ui.section_check_panel import SectionCheckPanel


def _app():
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


def test_normalize_decimal_text_accepts_common_user_inputs():
    assert normalize_decimal_text("12.5") == "12.5"
    assert normalize_decimal_text("12,5") == "12.5"
    assert normalize_decimal_text("50,,,,,,,,,,,,,,,,3") == "50.3"
    assert normalize_decimal_text("459....................8") == "459.8"
    assert normalize_decimal_text("12,,..,,5") == "12.5"
    assert normalize_decimal_text("abc12,5mm") == "12.5"
    assert normalize_decimal_text("--12,5", allow_negative=True) == "-12.5"
    assert normalize_decimal_text("--12,5", allow_negative=False) == "12.5"
    assert normalize_decimal_text("-1200", allow_negative=False) == "1200"
    assert normalize_decimal_text("--1200,5", allow_negative=False) == "1200.5"
    assert normalize_decimal_text("-1200", allow_negative=True) == "-1200"
    assert normalize_decimal_text("1200,5") == "1200.5"
    assert normalize_decimal_text("1200,,,,5") == "1200.5"


def test_parse_user_float_accepts_comma_dot_and_repeated_separators():
    assert parse_user_float("12.5") == 12.5
    assert parse_user_float("12,5") == 12.5
    assert parse_user_float("50,,,,3") == 50.3
    assert parse_user_float("459....8") == 459.8
    assert parse_user_float("-3,25") == -3.25


def test_line_edit_normalizer_keeps_useful_partial_text():
    _app()
    line = QLineEdit()

    for raw, expected in (
        ("", ""),
        ("-", "-"),
        ("12.", "12."),
        ("12,", "12."),
        ("50,,,,,,,,3", "50.3"),
        ("459................8", "459.8"),
        ("12,,..,,5", "12.5"),
    ):
        line.setText(raw)
        line.setCursorPosition(len(raw))
        normalize_line_edit_text(line, allow_negative=True)
        assert line.text() == expected


def test_section_verifier_accepts_comma_decimal_inputs_without_clearing_inputs():
    _app()
    panel = SectionCheckPanel()
    panel.set_moment_provider(lambda _x_mm: 125000.0)
    panel.chk_piso.setChecked(True)
    panel.n_piso_width.lineEdit().setText("2430,5")
    panel.n_piso_width.interpretText()
    panel._double_web_widgets[0].setChecked(True)
    panel.tbl.item(0, panel.COL_X).setText("1000,5")
    panel.tbl.item(0, panel.COL_HWEB).setText("450,5")
    panel.tbl.item(0, panel.COL_DOUBLE_WEB_OFFSET).setText("20,5")

    panel._recompute_all()
    panel.refresh_results_from_context()
    panel.clear_results_only()

    assert panel.n_piso_width.value() == 2430.5
    assert panel.tbl.item(0, panel.COL_X).text() == "1000,5"
    assert panel.tbl.item(0, panel.COL_HWEB).text() == "450,5"
    assert panel.tbl.item(0, panel.COL_DOUBLE_WEB_OFFSET).text() == "20,5"
    assert panel.tbl.item(0, panel.COL_FS).text() == ""
    panel._recompute_all()
    assert panel.tbl.item(0, panel.COL_FS).text()
    assert "ERR" not in panel.tbl.item(0, panel.COL_FS).text()


def test_reactions_load_table_accepts_comma_decimal_magnitude_and_position():
    _app()
    tab = SemiTrailerReactionsTab()
    tab.tbl.setRowCount(0)
    tab._add_load_row(load_type="Puntual")
    tab.tbl.item(0, tab.COL_MAG).setText("123,5")
    tab.tbl.item(0, tab.COL_POS).setText("1000,5")

    loads, errors, _ = tab._build_loads()

    assert errors == []
    assert len(loads) == 1
    assert isinstance(loads[0], PointForce)
    assert loads[0].value_user == 123.5
    assert loads[0].x_mm == 1000.5


def test_main_load_tables_accept_comma_decimal_inputs():
    _app()
    tab = UnitTab("Test")
    tab.Lc.setValue(10000.0)
    tab.tbl_points.setRowCount(0)
    tab._add_point_row()
    tab.tbl_points.item(0, 1).setText("1000,5")
    tab.tbl_points.item(0, 2).setText("500,25")

    beam, points, dists, moms, notes = tab.parse_inputs()

    assert beam.L_mm > 0.0
    assert notes == []
    assert len(points) == 1
    assert points[0].x_mm == 1000.5
    assert points[0].value_user == 500.25
    assert dists == []
    assert moms == []


def _paste_into_spinbox(sp, text: str) -> str:
    line_edit = sp.lineEdit()
    line_edit.clear()
    line_edit.insert(text)
    return line_edit.text()


def test_non_negative_kingpin_and_front_train_inputs_strip_minus_sign():
    _app()
    semi = UnitTab("Semirremolque")
    acoplado = UnitTab("Acoplado", is_acoplado=True)

    assert _paste_into_spinbox(semi.x_front_or_kp, "--1200,5") == "1200.5"
    assert _paste_into_spinbox(acoplado.x_front_or_kp, "-1200") == "1200"
    assert _paste_into_spinbox(semi.x_front_or_kp, "1200,,,,5") == "1200.5"


def test_non_negative_editable_reaction_inputs_strip_minus_sign():
    _app()
    bitren = UnitTab("Bitren", is_bitren=True)

    for spin in (
        bitren.R_front_or_kp,
        bitren.Rt,
        bitren.x_rp2_rel,
        bitren.Rp2,
    ):
        text = _paste_into_spinbox(spin, "abc-1200,5")
        assert text == "1200.5"
        assert "-" not in text
