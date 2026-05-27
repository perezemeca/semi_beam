import os
import zipfile
from types import SimpleNamespace

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from docx import Document
from PySide6.QtWidgets import QApplication, QLabel, QMessageBox, QPushButton, QScrollArea, QTableWidget, QToolButton

from semi_beam.domain.loads import DistUniform, PointForce, PointMoment
from semi_beam.sections.i_section import ISection
from semi_beam.services.study_storage import load_study_file, save_study_file
from semi_beam.ui.main_window import (
    DIRECTIONAL_OFFSET_MAX_MM,
    DIRECTIONAL_OFFSET_MIN_MM,
    FBDApp,
)
from semi_beam.ui.section_check_panel import SectionCheckPanel


def _app():
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


def test_section_panel_exposes_new_thickness_options():
    _app()
    panel = SectionCheckPanel()

    top_options = {panel.cmb_t_top.itemData(i) for i in range(panel.cmb_t_top.count())}
    web_options = {panel._tweb_widgets[0].itemData(i) for i in range(panel._tweb_widgets[0].count())}

    assert {"5/16", "3/8", "7/16"}.issubset(top_options)
    assert "7/16" in web_options
    assert panel.cmb_t_top.itemText(panel.cmb_t_top.findData("7/16")) == "7/16 - 11.11 mm"
    assert not hasattr(panel, "btn_export_report")
    assert panel.chk_bastidor_lateral.text() == "Agregar bastidor lateral"
    assert panel.chk_bastidor_lateral.isChecked() is False
    assert panel.chk_bastidor_lateral_structural.isChecked() is True
    assert panel.chk_bastidor_lateral_structural.isVisible() is False
    assert panel.n_bastidor_lateral_altura.minimum() == 130.0
    assert panel.n_bastidor_lateral_altura.maximum() == 170.0
    assert panel.n_bastidor_lateral_altura.value() == 170.0
    piso_options = {panel.cmb_espesor_piso.itemData(i) for i in range(panel.cmb_espesor_piso.count())}
    assert panel.chk_piso.text() == "Agregar piso"
    assert panel.chk_piso.isChecked() is False
    assert not hasattr(panel, "chk_piso_structural")
    assert panel.n_piso_width.value() == 2430.0
    assert panel.n_piso_width.isEnabled() is False
    assert {2.0, 3.0, 4.0, 3.175, 4.7625}.issubset(piso_options)
    assert panel.cmb_mat_piso.count() == panel.cmb_mat_top.count()
    chapon_options = {panel.cmb_espesor_chapon.itemData(i) for i in range(panel.cmb_espesor_chapon.count())}
    assert panel.chk_chapon.text() == "Agregar chapón"
    assert panel.chk_chapon.isChecked() is False
    assert {6.35, 7.9375, 9.525}.issubset(chapon_options)


def test_section_panel_lateral_frame_is_always_structural():
    _app()
    panel = SectionCheckPanel()
    panel.set_moment_provider(lambda _x_mm: 125000.0)
    panel.tbl.item(0, panel.COL_X).setText("1000")
    panel.tbl.item(0, panel.COL_HWEB).setText("450")
    panel._recompute_all()
    base_jx = float(panel.tbl.item(0, panel.COL_JX).text())

    panel.chk_bastidor_lateral.setChecked(True)
    panel.chk_bastidor_lateral_structural.setChecked(False)
    panel._recompute_all()
    legacy_false_jx = float(panel.tbl.item(0, panel.COL_JX).text())

    assert panel.chk_bastidor_lateral_structural.isChecked() is True
    assert legacy_false_jx > base_jx


def test_section_panel_floor_included_always_contributes_and_width_changes_properties():
    _app()
    panel = SectionCheckPanel()
    panel.set_moment_provider(lambda _x_mm: 125000.0)
    panel.tbl.item(0, panel.COL_X).setText("1000")
    panel.tbl.item(0, panel.COL_HWEB).setText("450")
    panel._recompute_all()
    base_jx = float(panel.tbl.item(0, panel.COL_JX).text())

    panel.chk_piso.setChecked(True)
    panel.n_piso_width.setValue(1000.0)
    panel._recompute_all()
    narrow_sec = panel._make_section(450.0, 0.25)
    narrow_area = sum(rect.area_mm2 for rect in narrow_sec.rects)
    narrow_jx = float(panel.tbl.item(0, panel.COL_JX).text())

    panel.n_piso_width.setValue(2400.0)
    panel.cmb_espesor_piso.setCurrentIndex(panel.cmb_espesor_piso.findData(4.7625))
    panel._recompute_all()
    wide_sec = panel._make_section(450.0, 0.25)
    wide_area = sum(rect.area_mm2 for rect in wide_sec.rects)
    wide_jx = float(panel.tbl.item(0, panel.COL_JX).text())
    piso_rects = [r for r in wide_sec.rects if r.label == "piso"]

    assert narrow_jx > base_jx
    assert wide_area > narrow_area
    assert wide_jx != narrow_jx
    assert wide_sec.includes_piso is True
    assert len(piso_rects) == 1
    assert piso_rects[0].x0_mm == -1200.0
    assert piso_rects[0].b_mm == 2400.0
    assert piso_rects[0].y0_mm == wide_sec.base_section.H_mm
    assert piso_rects[0].h_mm == 4.7625


def test_section_panel_floor_not_included_does_not_contribute():
    _app()
    panel = SectionCheckPanel()
    panel.set_moment_provider(lambda _x_mm: 125000.0)
    panel.tbl.item(0, panel.COL_X).setText("1000")
    panel.tbl.item(0, panel.COL_HWEB).setText("450")
    panel.n_piso_width.setValue(2400.0)

    sec = panel._make_section(450.0, 0.25)
    panel._recompute_all()

    assert isinstance(sec, ISection)
    assert float(panel.tbl.item(0, panel.COL_JX).text()) > 0.0
    assert all(getattr(rect, "label", "") != "piso" for rect in getattr(sec, "rects", ()))


def test_section_panel_adds_chapon_only_inside_longitudinal_range():
    _app()
    panel = SectionCheckPanel()
    panel.set_moment_provider(lambda _x_mm: 125000.0)
    panel.tbl.item(0, panel.COL_X).setText("1500")
    panel.tbl.item(0, panel.COL_HWEB).setText("450")
    panel.tbl.item(1, panel.COL_X).setText("2500")
    panel.tbl.item(1, panel.COL_HWEB).setText("450")
    panel._recompute_all()
    base_inside_jx = float(panel.tbl.item(0, panel.COL_JX).text())
    base_outside_jx = float(panel.tbl.item(1, panel.COL_JX).text())

    panel.chk_chapon.setChecked(True)
    panel.n_chapon_length.setValue(2000.0)
    panel.cmb_espesor_chapon.setCurrentIndex(panel.cmb_espesor_chapon.findData(9.525))
    panel._recompute_all()
    inside_jx = float(panel.tbl.item(0, panel.COL_JX).text())
    outside_jx = float(panel.tbl.item(1, panel.COL_JX).text())
    inside_sec = panel._make_section(450.0, 0.25, station_mm=1500.0)
    outside_sec = panel._make_section(450.0, 0.25, station_mm=2500.0)
    chapon_rects = [r for r in inside_sec.rects if r.label == "chapon"]

    assert inside_jx > base_inside_jx
    assert outside_jx == base_outside_jx
    assert inside_sec.includes_chapon is True
    assert isinstance(outside_sec, ISection)
    assert len(chapon_rects) == 1
    assert chapon_rects[0].x0_mm == -525.0
    assert chapon_rects[0].b_mm == 1050.0
    assert chapon_rects[0].y0_mm == -9.525
    assert chapon_rects[0].h_mm == 9.525

    panel.n_chapon_length.setValue(3000.0)
    panel._recompute_all()
    extended_sec = panel._make_section(450.0, 0.25, station_mm=2500.0)
    assert extended_sec.includes_chapon is True
    assert extended_sec.chapon_end_mm == 3000.0


def test_section_panel_governs_by_component_when_all_reinforcements_are_active():
    _app()
    panel = SectionCheckPanel()
    panel.set_moment_provider(lambda _x_mm: 250000.0)
    for combo, mat_id in (
        (panel.cmb_mat_top, "STR900MC"),
        (panel.cmb_mat_bot, "STR900MC"),
        (panel.cmb_mat_web, "STR900MC"),
        (panel.cmb_mat_piso, "STR900MC"),
    ):
        idx = combo.findData(mat_id)
        if idx >= 0:
            combo.setCurrentIndex(idx)
    panel.chk_bastidor_lateral.setChecked(True)
    panel.chk_piso.setChecked(True)
    panel.chk_chapon.setChecked(True)
    panel.tbl.item(0, panel.COL_X).setText("1500")
    panel.tbl.item(0, panel.COL_HWEB).setText("450")

    panel._recompute_all()
    card = panel._collect_section_export_cards()[0]
    checks = card["component_checks"]
    governing = card["governing_component"]
    fs_cell = float(panel.tbl.item(0, panel.COL_FS).text())

    components = {(row["component"], row["material"]) for row in checks}
    expected_min_fs = min(float(row["fs"]) for row in checks if row["fs"] is not None)

    assert ("Bastidor lateral", "SAE1010") in components
    assert ("Chapón", "SAE1010") in components
    assert ("Piso", "STR900MC") in components
    assert governing is not None
    assert abs(float(governing["fs"]) - expected_min_fs) < 1e-9
    assert abs(fs_cell - expected_min_fs) < 0.02


def test_section_panel_exports_verification_report(tmp_path):
    _app()
    panel = SectionCheckPanel()
    panel.set_moment_provider(lambda _x_mm: 125000.0)
    panel.set_shear_provider(lambda _x_mm: 4200.0)
    panel.set_deflection_context(
        SimpleNamespace(
            vmin_mm=-12.0,
            x_vmin_mm=850.0,
            utilized_mm=42.0,
            allowable_mm=60.0,
            limit_y_mm=-30.0,
            camber_mid_mm=30.0,
            ok=True,
        ),
        i_source="tabla de secciones",
    )
    panel.tbl.item(0, panel.COL_X).setText("1000")
    panel.tbl.item(0, panel.COL_HWEB).setText("450")
    panel._recompute_all()

    out = tmp_path / "memoria_calculo.docx"
    panel.export_verification_report(str(out))

    assert out.exists()
    assert out.stat().st_size > 0
    doc = Document(out)
    assert len(doc.inline_shapes) == 1
    assert len(doc.tables) >= 3
    assert doc.paragraphs[0].text == "Verificación de viga"
    all_text = "\n".join(p.text for p in doc.paragraphs if p.text)
    assert "Base teórica y metodología aplicada" in all_text
    assert "Memoria de cálculo de la verificación de sección" in all_text
    assert "Verificación al cortante" in all_text
    assert "Verificación de flecha" in all_text
    assert round(doc.sections[0].page_width.mm) == 210
    assert round(doc.sections[0].page_height.mm) == 297
    assert round(doc.sections[0].left_margin.mm, 1) == 25.4
    with zipfile.ZipFile(out) as zf:
        document_xml = zf.read("word/document.xml").decode("utf-8")
        styles_xml = zf.read("word/styles.xml").decode("utf-8")
        settings_xml = zf.read("word/settings.xml").decode("utf-8")
        core_xml = zf.read("docProps/core.xml").decode("utf-8")
    assert "es-AR" in document_xml
    assert "es-AR" in styles_xml
    assert "es-AR" in settings_xml
    assert "es-AR" in core_xml
    assert "en-US" not in document_xml
    assert "<m:oMath" in document_xml


def test_study_file_roundtrip_restores_ui_state(tmp_path):
    _app()
    window = FBDApp()
    window.tabs.setCurrentWidget(window.tab_semi)

    semi = window.tab_semi
    semi.cmb_semi_tipo.setCurrentIndex(1)
    semi.cmb_config.setCurrentIndex(2)
    semi.Lc.setValue(13600.0)
    semi.x_front_or_kp.setValue(1250.0)
    semi.R_front_or_kp.setValue(9400.0)
    semi.Rt.setValue(22200.0)
    semi.chk_show_deflection.setChecked(False)
    semi._add_load_row(load_type="Puntual")
    semi.tbl.item(0, semi.COL_MAG).setText("5000")
    semi.tbl.item(0, semi.COL_POS).setText("2500")
    semi.section_panel.cmb_t_top.setCurrentIndex(semi.section_panel.cmb_t_top.findData("7/16"))
    semi.section_panel.chk_bastidor_lateral.setChecked(True)
    semi.section_panel.n_bastidor_lateral_altura.setValue(150.0)
    semi.section_panel.chk_piso.setChecked(True)
    semi.section_panel.n_piso_width.setValue(1800.0)
    semi.section_panel.cmb_espesor_piso.setCurrentIndex(semi.section_panel.cmb_espesor_piso.findData(3.175))
    semi.section_panel.cmb_mat_piso.setCurrentIndex(semi.section_panel.cmb_mat_piso.findData("F24"))
    semi.section_panel.chk_chapon.setChecked(True)
    semi.section_panel.n_chapon_length.setValue(2750.0)
    semi.section_panel.cmb_espesor_chapon.setCurrentIndex(semi.section_panel.cmb_espesor_chapon.findData(7.9375))
    semi.section_panel._double_web_widgets[0].setChecked(True)
    semi.section_panel.tbl.item(0, semi.section_panel.COL_DOUBLE_WEB_OFFSET).setText("20")
    semi.section_panel.tbl.item(0, semi.section_panel.COL_X).setText("1000")
    semi.section_panel.tbl.item(0, semi.section_panel.COL_HWEB).setText("450")

    reactions = window.tab_reactions
    reactions.mode.setCurrentIndex(1)
    reactions.offset.setValue(3700.0)

    payload = window._export_study_state()
    study_path = tmp_path / "demo.sbeam"
    save_study_file(study_path, payload)

    restored = FBDApp()
    restored._apply_study_state(load_study_file(study_path))

    restored_semi = restored.tab_semi
    restored_reactions = restored.tab_reactions

    assert restored.tabs.currentIndex() == 1
    assert restored_semi.cmb_semi_tipo.currentIndex() == 1
    assert restored_semi.cmb_config.currentText() == semi.cmb_config.currentText()
    assert restored_semi.Lc.value() == 13600.0
    assert restored_semi.x_front_or_kp.value() == 1250.0
    assert restored_semi.chk_show_deflection.isChecked() is True
    assert restored_semi.deflection_enabled() is True
    assert restored_semi._type_combo(0).currentText() == "Puntual"
    assert restored_semi.tbl.item(0, restored_semi.COL_POS).text() == "2500"
    assert restored_semi.tbl.item(0, restored_semi.COL_MAG).text() == "5000"
    assert restored_semi.section_panel.cmb_t_top.currentData() == "7/16"
    assert restored_semi.section_panel.chk_bastidor_lateral.isChecked() is True
    assert restored_semi.section_panel.chk_bastidor_lateral_structural.isChecked() is True
    assert restored_semi.section_panel.n_bastidor_lateral_altura.value() == 150.0
    assert restored_semi.section_panel.chk_piso.isChecked() is True
    assert restored_semi.section_panel.n_piso_width.value() == 1800.0
    assert restored_semi.section_panel.cmb_espesor_piso.currentData() == 3.175
    assert restored_semi.section_panel.cmb_mat_piso.currentData() == "F24"
    assert restored_semi.section_panel.chk_chapon.isChecked() is True
    assert restored_semi.section_panel.n_chapon_length.value() == 2750.0
    assert restored_semi.section_panel.cmb_espesor_chapon.currentData() == 7.9375
    assert restored_semi.section_panel._row_double_web_enabled(0) is True
    assert restored_semi.section_panel._row_double_web_offset_mm(0) == 20.0
    assert restored_semi.section_panel.tbl.item(0, restored_semi.section_panel.COL_X).text() == "1000"
    assert restored_semi.section_panel.tbl.item(0, restored_semi.section_panel.COL_HWEB).text() == "450"
    assert restored_reactions.mode.currentIndex() == 1
    assert restored_reactions.offset.value() == 3700.0


def test_axis_lift_state_roundtrip_and_old_default(tmp_path):
    _app()
    window = FBDApp()
    semi = window.tab_semi
    semi.chk_axis_lift.setChecked(True)

    payload = window._export_study_state()
    assert payload["tabs"]["semirremolque"]["axis_lift_enabled"] is True

    study_path = tmp_path / "axis_lift.sbeam"
    save_study_file(study_path, payload)

    restored = FBDApp()
    restored._apply_study_state(load_study_file(study_path))
    assert restored.tab_semi.chk_axis_lift.isChecked() is True

    old_payload = window._export_study_state()
    del old_payload["tabs"]["semirremolque"]["axis_lift_enabled"]
    restored_old = FBDApp()
    restored_old._apply_study_state(old_payload)
    assert restored_old.tab_semi.chk_axis_lift.isChecked() is False

    window.close()
    restored.close()
    restored_old.close()


def test_section_panel_import_old_state_uses_default_chapon_length():
    _app()
    panel = SectionCheckPanel()

    panel.import_state({"include_chapon": True, "espesor_chapon": 6.35, "rows": []})

    assert panel.chk_chapon.isChecked() is True
    assert panel.n_chapon_length.value() == 2000.0
    state = panel.export_state()
    assert state["chapon_length_mm"] == 2000.0


def test_section_panel_import_old_lateral_frame_false_structural_forces_true():
    _app()
    panel = SectionCheckPanel()
    panel.import_state({
        "include_bastidor_lateral": True,
        "bastidor_lateral_structural": False,
        "bastidor_lateral_height_mm": 150.0,
    })

    assert panel.chk_bastidor_lateral.isChecked() is True
    assert panel.chk_bastidor_lateral_structural.isChecked() is True
    assert panel.export_state()["bastidor_lateral_structural"] is True


def _configure_solvable_new_study(window: FBDApp):
    tab = window.tab_semi
    window.tabs.setCurrentWidget(tab)
    tab.Lc.setValue(13600.0)
    tab.x_front_or_kp.setValue(1250.0)
    tab.R_front_or_kp.setValue(9400.0)
    tab.Rt.setValue(22200.0)
    tab.chk_show_deflection.setChecked(False)
    tab._add_load_row(load_type="Puntual")
    tab.tbl.item(0, tab.COL_MAG).setText("5000")
    tab.tbl.item(0, tab.COL_POS).setText("2500")
    return tab


def _point_by_label(points, label: str):
    matches = [p for p in points if p.label == label]
    assert len(matches) == 1
    return matches[0]


def test_unit_tab_single_load_table_parses_all_load_types():
    _app()
    window = FBDApp()
    tab = window.tab_semi
    tab.Lc.setValue(12000.0)
    tab.tbl.setRowCount(0)

    tab._add_load_row(load_type="Puntual")
    tab.tbl.item(0, tab.COL_MAG).setText("500")
    tab.tbl.item(0, tab.COL_POS).setText("1000")

    tab._add_load_row(load_type="Distribuida")
    tab.tbl.item(1, tab.COL_MAG).setText("1200")
    tab.tbl.item(1, tab.COL_POS).setText("4000")
    tab.tbl.item(1, tab.COL_LEN).setText("600")

    tab._add_load_row(load_type="Momento")
    tab.tbl.item(2, tab.COL_MAG).setText("25000")
    tab.tbl.item(2, tab.COL_POS).setText("7000")

    _beam, points, dists, moms, notes = tab.parse_inputs()

    assert notes == []
    assert len(points) == 1
    assert isinstance(points[0], PointForce)
    assert points[0].label == "P1"
    assert points[0].value_user == 500.0
    assert points[0].x_mm == 1000.0
    assert len(dists) == 1
    assert isinstance(dists[0], DistUniform)
    assert dists[0].label == "q1"
    assert dists[0].x0_mm == 3700.0
    assert dists[0].Lq_mm == 600.0
    assert dists[0].q_user == 2.0
    assert len(moms) == 1
    assert isinstance(moms[0], PointMoment)
    assert moms[0].label == "M1"
    assert moms[0].M_user_kgmm == 25000.0
    assert moms[0].x_mm == 7000.0
    window.close()


def test_unit_tab_imports_legacy_split_load_tables_into_single_load_table():
    _app()
    window = FBDApp()
    tab = window.tab_semi
    legacy_state = tab.export_state()
    legacy_state.pop("loads", None)
    legacy_state["tbl_points"] = [["P1", "1000", "500"]]
    legacy_state["tbl_dists"] = [["q1", "3700", "600", "2"]]
    legacy_state["tbl_moms"] = [["M1", "7000", "25000"]]

    restored = FBDApp()
    restored.tab_semi.import_state(legacy_state)
    restored_tab = restored.tab_semi

    assert restored_tab.tbl.rowCount() == 3
    assert restored_tab._type_combo(0).currentText() == "Puntual"
    assert restored_tab.tbl.item(0, restored_tab.COL_MAG).text() == "500"
    assert restored_tab.tbl.item(0, restored_tab.COL_POS).text() == "1000"
    assert restored_tab._type_combo(1).currentText() == "Distribuida"
    assert restored_tab.tbl.item(1, restored_tab.COL_MAG).text() == "1200"
    assert restored_tab.tbl.item(1, restored_tab.COL_POS).text() == "4000"
    assert restored_tab.tbl.item(1, restored_tab.COL_LEN).text() == "600"
    assert restored_tab._type_combo(2).currentText() == "Momento"
    assert restored_tab.tbl.item(2, restored_tab.COL_MAG).text() == "25000"
    assert restored_tab.tbl.item(2, restored_tab.COL_POS).text() == "7000"
    window.close()
    restored.close()


def test_axis_lift_adds_single_first_axle_load_without_moving_tandem():
    _app()
    window = FBDApp()
    tab = _configure_solvable_new_study(window)

    window._solve_for_tab(tab)
    base_cache = tab.get_cache()
    assert base_cache is not None
    base_rp1 = _point_by_label(base_cache.points, "Rp1")
    base_rt = _point_by_label(base_cache.points, "Rt")
    sample_x = float(base_rt.x_mm) - 500.0
    base_v = float(tab.get_diag().eval_V(sample_x))
    base_m = float(tab.get_diag().eval_M(sample_x))
    assert [p for p in base_cache.points if p.label.startswith("P_levante")] == []

    tab.chk_axis_lift.setChecked(True)
    window._replot_active_tab()
    lifted_cache = tab.get_cache()
    assert lifted_cache is not None
    lifted_rp1 = _point_by_label(lifted_cache.points, "Rp1")
    lifted_rt = _point_by_label(lifted_cache.points, "Rt")
    lift_loads = [p for p in lifted_cache.points if p.label == "P_levante_primer_eje"]

    assert len(lift_loads) == 1
    assert lift_loads[0].value_user == 1200.0
    assert lift_loads[0].x_mm == lifted_rt.x_mm - 625.0
    assert lifted_rt.x_mm == base_rt.x_mm
    assert lifted_rt.value_user != base_rt.value_user
    assert (lifted_rp1.value_user, lifted_rt.value_user) != (base_rp1.value_user, base_rt.value_user)
    assert tab.view_mode() == "solved"
    assert float(tab.get_diag().eval_V(sample_x)) != base_v
    assert float(tab.get_diag().eval_M(sample_x)) != base_m
    assert abs(float(tab.get_diag().eval_V(lifted_cache.beam_plot.L_mm))) < 1e-6
    assert abs(float(tab.get_diag().eval_M(lifted_cache.beam_plot.L_mm))) < 1e-3

    tab.chk_axis_lift.setChecked(False)
    window._replot_active_tab()
    disabled_cache = tab.get_cache()
    assert disabled_cache is not None
    assert [p for p in disabled_cache.points if p.label.startswith("P_levante")] == []
    assert float(tab.get_diag().eval_V(sample_x)) == base_v
    assert float(tab.get_diag().eval_M(sample_x)) == base_m

    tab.chk_axis_lift.setChecked(True)
    window._replot_active_tab()
    window._solve_for_tab(tab)
    solved_again = tab.get_cache()
    assert solved_again is not None
    assert len([p for p in solved_again.points if p.label == "P_levante_primer_eje"]) == 1
    window.close()


def test_axis_lift_directional_load_uses_directional_position():
    _app()
    window = FBDApp()
    tab = _configure_solvable_new_study(window)
    tab.cmb_config.setCurrentText("1 + 2 ejes — Rd 9200 (offset 3075) + Rt 15800")
    tab.Rd.setValue(9200.0)
    tab.dir_offset.setValue(3075.0)
    tab.chk_axis_lift.setChecked(True)

    window._solve_for_tab(tab)

    cache = tab.get_cache()
    assert cache is not None
    rt = _point_by_label(cache.points, "Rt")
    lift_load = _point_by_label(cache.points, "P_levante_direccional")
    assert "Levantar direccional" in tab.chk_axis_lift.text()
    assert lift_load.value_user == 1300.0
    assert tab.directional_offset_for_solver() == tab.dir_offset.value() + 625.0
    assert lift_load.x_mm == rt.x_mm - tab.directional_offset_for_solver()
    assert [p for p in cache.points if p.label == "Rd"] == []
    assert abs(float(tab.get_diag().eval_V(cache.beam_plot.L_mm))) < 1e-6
    assert abs(float(tab.get_diag().eval_M(cache.beam_plot.L_mm))) < 1e-3
    window.close()


def test_axis_lift_treats_1_1_1_as_directional_lift():
    _app()
    window = FBDApp()
    tab = window.tab_semi
    tab.cmb_config.setCurrentText("1 + 1 + 1 ejes — Rd 9200 (offset 2450) + Rt 18800")
    tab._apply_config_defaults()

    assert tab.chk_axis_lift.isEnabled() is True
    assert "Levantar direccional" in tab.chk_axis_lift.text()
    assert tab.Rd.value() == 9200.0
    assert tab.Rt.value() == 18800.0
    assert tab.Rd.value() + tab.Rt.value() == 28000.0
    assert tab.dir_offset.value() == 2450.0
    window.close()


def test_directional_offset_is_entered_from_second_axis_and_validated():
    _app()
    window = FBDApp()
    tab = _configure_solvable_new_study(window)
    tab.cmb_config.setCurrentText("1 + 2 ejes — Rd 9200 (offset 3075) + Rt 15800")
    tab._apply_config_defaults()

    assert tab.dir_offset.minimum() == DIRECTIONAL_OFFSET_MIN_MM
    assert tab.dir_offset.maximum() == DIRECTIONAL_OFFSET_MAX_MM
    assert "segundo eje" in tab.dir_offset.toolTip()

    for value in (2400.0, 2450.0, 2475.0, 4000.0):
        tab.dir_offset.setValue(value)
        assert tab._validate_required_inputs() == []

    for invalid in ("0", "1000", "4500"):
        tab.dir_offset.lineEdit().setText(invalid)
        assert any("2400 y 4000" in error for error in tab._validate_required_inputs())

    tab.dir_offset.setValue(3075.0)
    assert tab.directional_offset_from_second_axis() == 3075.0
    assert tab.directional_offset_for_solver() == 3700.0
    window.close()


def test_semi_1_1_1_solves_with_directional_lift_and_fixed_tandem():
    _app()
    window = FBDApp()
    tab = _configure_solvable_new_study(window)
    tab.cmb_config.setCurrentText("1 + 1 + 1 ejes — Rd 9200 (offset 2450) + Rt 18800")
    tab._apply_config_defaults()

    window._solve_for_tab(tab)
    base_cache = tab.get_cache()
    assert base_cache is not None
    base_rt = _point_by_label(base_cache.points, "Rt")

    tab.chk_axis_lift.setChecked(True)
    window._replot_active_tab()
    lifted_cache = tab.get_cache()
    assert lifted_cache is not None
    lifted_rt = _point_by_label(lifted_cache.points, "Rt")
    lift_load = _point_by_label(lifted_cache.points, "P_levante_direccional")

    assert tab.Rd.value() + tab.Rt.value() == 28000.0
    assert tab.dir_offset.value() == 2450.0
    assert lift_load.value_user == 1300.0
    assert tab.directional_offset_for_solver() == 3675.0
    assert lift_load.x_mm == lifted_rt.x_mm - tab.directional_offset_for_solver()
    assert lifted_rt.x_mm == base_rt.x_mm
    assert lifted_rt.value_user != base_rt.value_user
    assert abs(float(tab.get_diag().eval_V(lifted_cache.beam_plot.L_mm))) < 1e-6
    assert abs(float(tab.get_diag().eval_M(lifted_cache.beam_plot.L_mm))) < 1e-3
    window.close()


def test_unit_tabs_have_unified_load_add_controls_and_fixed_tab_bar():
    _app()
    window = FBDApp()

    assert isinstance(window.tab_acoplado.content_scroll, QScrollArea)
    assert isinstance(window.tab_semi.content_scroll, QScrollArea)
    assert isinstance(window.tab_bitren.content_scroll, QScrollArea)
    assert isinstance(window.tab_reactions.content_scroll, QScrollArea)
    assert not isinstance(window.tabs.parentWidget(), QScrollArea)

    for tab in (window.tab_acoplado, window.tab_semi, window.tab_bitren):
        sections = [button for button in tab.findChildren(QToolButton) if button.text() == "Cargas"]
        assert len(sections) == 1
        assert tab.btn_add_load.text() == "Agregar carga"
        assert tab.btn_del_load.text() == "Eliminar seleccionadas"
        assert not hasattr(tab, "cmb_add_load_type")
        assert not hasattr(tab, "btn_add_p")
        assert not hasattr(tab, "btn_add_q")
        assert not hasattr(tab, "btn_add_m")
        assert not hasattr(tab, "tbl_points")
        assert not hasattr(tab, "tbl_dists")
        assert not hasattr(tab, "tbl_moms")

        headers = [
            tab.tbl.horizontalHeaderItem(col).text()
            for col in range(tab.tbl.columnCount())
        ]
        assert headers == ["Tipo", "Magnitud", "Posición / centro [mm]", "Longitud [mm]"]

        for load_type in ("Puntual", "Distribuida", "Momento"):
            before = tab.tbl.rowCount()
            tab._add_load_row(load_type=load_type)
            assert tab.tbl.rowCount() == before + 1
            assert tab._type_combo(before).currentText() == load_type

    assert window.tab_reactions.btn_add.text() == "Agregar carga"
    window.close()


def test_unit_tabs_render_single_general_load_table_without_old_collapsibles():
    app = _app()
    window = FBDApp()
    window.resize(1400, 900)
    window.show()
    app.processEvents()

    forbidden_sections = {
        "Fuerzas puntuales conocidas (P1, P2, ...)",
        "Cargas distribuidas conocidas (kg/mm)",
        "Momentos puntuales (kg·mm, CCW+)",
    }
    general_headers = ("Tipo", "Magnitud", "Posición / centro [mm]", "Longitud [mm]")

    for tab in (window.tab_acoplado, window.tab_semi, window.tab_bitren):
        window.tabs.setCurrentWidget(tab)
        app.processEvents()

        visible_labels = {label.text() for label in tab.findChildren(QLabel) if label.isVisible()}
        visible_buttons = [button for button in tab.findChildren(QToolButton) if button.isVisible()]
        visible_sections = {button.text() for button in visible_buttons}
        assert "Cargas" in visible_sections
        assert visible_sections.isdisjoint(forbidden_sections)
        assert "Bastidor lateral estructural" not in visible_labels
        assert "Deformada" not in visible_sections
        assert "Mostrar deformada" not in visible_labels

        load_button = next(button for button in visible_buttons if button.text() == "Cargas")
        load_button.click()
        app.processEvents()
        visible_labels = {label.text() for label in tab.findChildren(QLabel) if label.isVisible()}
        assert "Magnitud: kg para puntuales y distribuidas (P total), kg·mm para momentos." in visible_labels

        visible_headers = []
        for table in tab.findChildren(QTableWidget):
            if not table.isVisible():
                continue
            visible_headers.append(tuple(
                table.horizontalHeaderItem(col).text()
                for col in range(table.columnCount())
            ))

        assert visible_headers.count(general_headers) == 1
        assert ("label", "x_mm", "valor_kg") not in visible_headers
        assert ("label", "x0_mm", "Lq_mm", "q_kg_per_mm") not in visible_headers
        assert ("label", "x_mm", "M_kgmm") not in visible_headers

    bitren_labels = {
        label.text()
        for label in window.tab_bitren.findChildren(QLabel)
        if label.isVisible()
    }
    bitren_buttons = {
        button.text()
        for button in window.tab_bitren.findChildren(QToolButton)
        if button.isVisible()
    }
    assert not any("direccional" in text.lower() for text in bitren_labels | bitren_buttons)

    window.close()


def test_about_action_is_single_button_at_top_bar_right(monkeypatch):
    app = _app()
    window = FBDApp()
    window.show()
    app.processEvents()

    assert [action.text() for action in window.menuBar().actions()] == []

    about_buttons = [
        button
        for button in window.findChildren(QPushButton)
        if button.text() == "Acerca de Calculeitor"
    ]
    assert about_buttons == [window.btn_about]
    assert window.btn_about.parentWidget() is window.btn_export_memoria_docx.parentWidget()
    assert window.btn_about.geometry().x() > window.btn_export_memoria_docx.geometry().x()
    assert window.btn_about.geometry().right() > window.btn_export_memoria_docx.geometry().right()

    tab_texts = [window.tabs.tabText(index) for index in range(window.tabs.count())]
    assert tab_texts == ["Acoplado", "Semirremolque", "Bitren", "Cálculo y verificación"]
    assert not any(
        button.text() == "Ayuda"
        for button in window.findChildren(QPushButton)
    )

    calls = []

    def fake_about(parent, title, text):
        calls.append((parent, title, text))

    monkeypatch.setattr(QMessageBox, "about", fake_about)
    window.btn_about.click()

    assert len(calls) == 1
    assert calls[0][0] is window
    assert calls[0][1] == "Acerca de Calculeitor"
    assert "Calculeitor" in calls[0][2]
    window.close()


def test_axis_lift_disabled_for_non_applicable_configuration():
    _app()
    window = FBDApp()
    tab = window.tab_acoplado
    tab.cmb_config.setCurrentText("4 ejes conv — 15800 / 15800")
    tab.chk_axis_lift.setChecked(True)
    tab._apply_config_defaults()

    assert tab.chk_axis_lift.isEnabled() is False
    assert tab.chk_axis_lift.isChecked() is False
    window.close()


def test_new_study_solve_refreshes_section_check_results_without_import_state():
    _app()
    window = FBDApp()
    tab = _configure_solvable_new_study(window)
    panel = tab.section_panel
    panel.tbl.item(0, panel.COL_X).setText("1000")
    panel.tbl.item(0, panel.COL_HWEB).setText("450")

    window._solve_for_tab(tab)

    fs_text = panel.tbl.item(0, panel.COL_FS).text()
    assert panel.tbl.item(0, panel.COL_M).text()
    assert fs_text
    assert "ERR" not in fs_text
    assert float(fs_text) > 0.0
    window.close()


def test_new_study_solve_refreshes_double_web_section_results_without_import_state():
    _app()
    window = FBDApp()
    tab = _configure_solvable_new_study(window)
    panel = tab.section_panel
    panel._double_web_widgets[0].setChecked(True)
    panel.tbl.item(0, panel.COL_DOUBLE_WEB_OFFSET).setText("20")
    panel.tbl.item(0, panel.COL_X).setText("1000")
    panel.tbl.item(0, panel.COL_HWEB).setText("450")

    window._solve_for_tab(tab)

    fs_text = panel.tbl.item(0, panel.COL_FS).text()
    assert panel.tbl.item(0, panel.COL_M).text()
    assert fs_text
    assert "ERR" not in fs_text
    assert float(fs_text) > 0.0
    window.close()


def test_window_title_shows_current_study_filename(tmp_path):
    _app()
    window = FBDApp()

    assert window.windowTitle() == "Calculeitor - Acoplado / Semirremolque / Bitren"

    window._current_study_path = "C:/tmp/Forestal.sbeam"
    window._update_window_title()

    assert "Forestal" in window.windowTitle()
    assert ".sbeam" not in window.windowTitle()
    assert window.windowTitle() == "Calculeitor - Acoplado / Semirremolque / Bitren — Forestal"
