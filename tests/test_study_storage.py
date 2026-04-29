import os
import zipfile
from types import SimpleNamespace

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from docx import Document
from PySide6.QtWidgets import QApplication

from semi_beam.sections.i_section import ISection
from semi_beam.services.study_storage import load_study_file, save_study_file
from semi_beam.ui.main_window import FBDApp
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


def test_section_panel_adds_lateral_frame_only_when_structural():
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
    non_structural_jx = float(panel.tbl.item(0, panel.COL_JX).text())

    panel.chk_bastidor_lateral_structural.setChecked(True)
    panel.n_bastidor_lateral_altura.setValue(150.0)
    panel._recompute_all()
    structural_jx = float(panel.tbl.item(0, panel.COL_JX).text())

    assert non_structural_jx == base_jx
    assert structural_jx > base_jx


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
    panel.set_beam_context(largo_viga_mm=5000.0, posicion_perno_mm=1000.0)
    panel.set_moment_provider(lambda _x_mm: 125000.0)
    panel.tbl.item(0, panel.COL_X).setText("1500")
    panel.tbl.item(0, panel.COL_HWEB).setText("450")
    panel.tbl.item(1, panel.COL_X).setText("2500")
    panel.tbl.item(1, panel.COL_HWEB).setText("450")
    panel._recompute_all()
    base_inside_jx = float(panel.tbl.item(0, panel.COL_JX).text())
    base_outside_jx = float(panel.tbl.item(1, panel.COL_JX).text())

    panel.chk_chapon.setChecked(True)
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


def test_section_panel_governs_by_component_when_all_reinforcements_are_active():
    _app()
    panel = SectionCheckPanel()
    panel.set_beam_context(largo_viga_mm=5000.0, posicion_perno_mm=1000.0)
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
    semi._add_point_row()
    semi.tbl_points.item(0, 1).setText("2500")
    semi.tbl_points.item(0, 2).setText("5000")
    semi.section_panel.cmb_t_top.setCurrentIndex(semi.section_panel.cmb_t_top.findData("7/16"))
    semi.section_panel.chk_bastidor_lateral.setChecked(True)
    semi.section_panel.n_bastidor_lateral_altura.setValue(150.0)
    semi.section_panel.chk_piso.setChecked(True)
    semi.section_panel.n_piso_width.setValue(1800.0)
    semi.section_panel.cmb_espesor_piso.setCurrentIndex(semi.section_panel.cmb_espesor_piso.findData(3.175))
    semi.section_panel.cmb_mat_piso.setCurrentIndex(semi.section_panel.cmb_mat_piso.findData("F24"))
    semi.section_panel.chk_chapon.setChecked(True)
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
    assert restored_semi.chk_show_deflection.isChecked() is False
    assert restored_semi.tbl_points.item(0, 1).text() == "2500"
    assert restored_semi.tbl_points.item(0, 2).text() == "5000"
    assert restored_semi.section_panel.cmb_t_top.currentData() == "7/16"
    assert restored_semi.section_panel.chk_bastidor_lateral.isChecked() is True
    assert restored_semi.section_panel.chk_bastidor_lateral_structural.isChecked() is True
    assert restored_semi.section_panel.n_bastidor_lateral_altura.value() == 150.0
    assert restored_semi.section_panel.chk_piso.isChecked() is True
    assert restored_semi.section_panel.n_piso_width.value() == 1800.0
    assert restored_semi.section_panel.cmb_espesor_piso.currentData() == 3.175
    assert restored_semi.section_panel.cmb_mat_piso.currentData() == "F24"
    assert restored_semi.section_panel.chk_chapon.isChecked() is True
    assert restored_semi.section_panel.cmb_espesor_chapon.currentData() == 7.9375
    assert restored_semi.section_panel._row_double_web_enabled(0) is True
    assert restored_semi.section_panel._row_double_web_offset_mm(0) == 20.0
    assert restored_semi.section_panel.tbl.item(0, restored_semi.section_panel.COL_X).text() == "1000"
    assert restored_semi.section_panel.tbl.item(0, restored_semi.section_panel.COL_HWEB).text() == "450"
    assert restored_reactions.mode.currentIndex() == 1
    assert restored_reactions.offset.value() == 3700.0


def _configure_solvable_new_study(window: FBDApp):
    tab = window.tab_semi
    window.tabs.setCurrentWidget(tab)
    tab.Lc.setValue(13600.0)
    tab.x_front_or_kp.setValue(1250.0)
    tab.R_front_or_kp.setValue(9400.0)
    tab.Rt.setValue(22200.0)
    tab.chk_show_deflection.setChecked(False)
    tab._add_point_row()
    tab.tbl_points.item(0, 1).setText("2500")
    tab.tbl_points.item(0, 2).setText("5000")
    return tab


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
