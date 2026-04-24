import base64
import os
import tempfile
import zipfile
from datetime import datetime

from docx import Document

from semi_beam.services.memoria_calculo_docx import (
    export_memoria_docx,
    MemoriaCaso,
    MemoriaHeader,
    MemoriaResultados,
)


_PNG_1X1 = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO2p6Z8AAAAASUVORK5CYII="
)


def _write_png(path: str) -> None:
    with open(path, "wb") as fh:
        fh.write(_PNG_1X1)


def test_export_memoria_docx_includes_deflection_image():
    with tempfile.TemporaryDirectory() as td:
        out = os.path.join(td, "memoria.docx")

        img_fbd = os.path.join(td, "fbd.png")
        img_v = os.path.join(td, "v.png")
        img_m = os.path.join(td, "m.png")
        img_defl = os.path.join(td, "deflection.png")
        for path in (img_fbd, img_v, img_m, img_defl):
            _write_png(path)

        header = MemoriaHeader(titulo="Memoria Test", fecha=datetime.now())
        caso = MemoriaCaso(
            unidad="Caso 1",
            L_carrozable_mm=1000.0,
            L_viga_total_mm=1200.0,
            descripcion_config="Dummy",
            apoyos=[("Rp1", "x=0 mm; R=0 kg")],
            cargas=[("P1", "x=200 mm; P=100 kg")],
        )
        resultados = MemoriaResultados(
            q_user_kgmm=0.1,
            x_t_mm=500.0,
            x_d_mm=None,
            residual_Fy=0.0,
            residual_M0=0.0,
            extremos_V=[],
            extremos_M=[],
            vmin_mm=-12.0,
            x_vmin_mm=450.0,
            utilized_mm=42.0,
            allowable_mm=60.0,
            deflection_ok=True,
            i_source="tabla de secciones",
        )

        export_memoria_docx(
            out,
            header=header,
            caso=caso,
            resultados=resultados,
            seccion=None,
            imagenes={
                "fbd": img_fbd,
                "v": img_v,
                "m": img_m,
                "deflection": img_defl,
            },
            extras={},
        )

        doc = Document(out)
        assert len(doc.inline_shapes) == 4
        text = "\n".join(p.text for p in doc.paragraphs if p.text)
        assert "Memoria de Cálculo" in text
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
        assert "en-US" not in styles_xml
