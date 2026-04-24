import os
import zipfile
from types import SimpleNamespace

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from docx import Document
from PySide6.QtWidgets import QApplication

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
    assert restored_semi.section_panel.tbl.item(0, restored_semi.section_panel.COL_X).text() == "1000"
    assert restored_semi.section_panel.tbl.item(0, restored_semi.section_panel.COL_HWEB).text() == "450"
    assert restored_reactions.mode.currentIndex() == 1
    assert restored_reactions.offset.value() == 3700.0
