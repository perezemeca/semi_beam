from __future__ import annotations

from datetime import datetime
from pathlib import Path
import sys
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple
from xml.sax.saxutils import escape as _xml_escape

from semi_beam.services.memoria_calculo_pdf import (
    MemoriaCaso,
    MemoriaHeader,
    MemoriaResultados,
    MemoriaSeccion,
)

try:
    from docx import Document
    from docx.shared import Mm
    from docx.oxml import parse_xml
    from docx.oxml.ns import nsdecls
except Exception:  # pragma: no cover
    Document = None
    Mm = None
    parse_xml = None
    nsdecls = None


def _ensure_docx_lib():
    if Document is None:
        raise RuntimeError(
            "Falta dependencia 'python-docx'. Instale requirements.txt y reintente."
        )


def _fmt_num(v: Any, decimals: int = 2) -> str:
    try:
        x = float(v)
    except Exception:
        return str(v)
    s = f"{x:.{int(decimals)}f}"
    if "." in s:
        s = s.rstrip("0").rstrip(".")
    return s.replace(".", ",")


def _clear_paragraph(paragraph):
    p = paragraph._p
    for child in list(p):
        p.remove(child)


def _append_omml_number(paragraph, text: str):
    _ensure_docx_lib()
    t = _xml_escape((text or "").strip())
    if not t:
        t = "0"
    omml = parse_xml(
        f'<m:oMath {nsdecls("m", "w")}>'
        f"<m:r><w:rPr><w:rStyle w:val=\"Equation\"/></w:rPr><m:t>{t}</m:t></m:r>"
        f"</m:oMath>"
    )
    paragraph._p.append(omml)


def _set_labeled_number(paragraph, label: str, value: Any, *, unit: str = "", decimals: int = 2):
    _clear_paragraph(paragraph)
    paragraph.add_run(label)
    _append_omml_number(paragraph, _fmt_num(value, decimals=decimals))
    if unit:
        paragraph.add_run(f" {unit}")


def _parse_support_x_mm(detail: str) -> Optional[float]:
    # Formato esperado: "x=2200 mm; R=..."
    if not detail:
        return None
    txt = detail.replace(",", ".")
    k = txt.find("x=")
    if k < 0:
        return None
    tail = txt[k + 2 :]
    end = tail.find("mm")
    if end < 0:
        return None
    raw = tail[:end].replace(";", " ").strip()
    try:
        return float(raw)
    except Exception:
        return None


def _replace_or_append_line(doc, *, token_matches: Sequence[str], line_builder) -> None:
    keyset = [x.lower() for x in token_matches]
    for p in doc.paragraphs:
        t = (p.text or "").strip().lower()
        if any(k in t for k in keyset):
            line_builder(p)
            return
    p = doc.add_paragraph("")
    line_builder(p)


def _set_table_cell_text(cell, value: str):
    if not cell.paragraphs:
        cell.add_paragraph("")
    p = cell.paragraphs[0]
    _clear_paragraph(p)
    p.add_run(value)


def _set_table_cell_num(cell, value: Any, *, decimals: int = 2):
    if not cell.paragraphs:
        cell.add_paragraph("")
    p = cell.paragraphs[0]
    _clear_paragraph(p)
    _append_omml_number(p, _fmt_num(value, decimals=decimals))


def _replace_inline_images(doc, image_paths: Sequence[str]):
    _ensure_docx_lib()
    img_iter = [p for p in image_paths if p and Path(p).exists()]
    if not img_iter:
        return

    idx = 0
    for p in doc.paragraphs:
        if idx >= len(img_iter):
            break
        txt = (p.text or "").strip()
        if txt != "":
            continue
        _clear_paragraph(p)
        run = p.add_run()
        run.add_picture(img_iter[idx], width=Mm(165))
        idx += 1

    while idx < len(img_iter):
        p = doc.add_paragraph("")
        run = p.add_run()
        run.add_picture(img_iter[idx], width=Mm(165))
        idx += 1


def ensure_memoria_template(path: str) -> str:
    p = Path(path)
    if p.exists():
        return str(p)
    _ensure_docx_lib()
    p.parent.mkdir(parents=True, exist_ok=True)
    doc = Document()
    doc.add_heading("Memoria de Cálculo", level=1)
    doc.add_paragraph("Largo de viga: ")
    doc.add_paragraph("Configuración: ")
    doc.add_paragraph("Distancia perno de enganche: ")
    doc.add_paragraph("Peso 1º eje completo: ")
    doc.add_paragraph("Peso 2º eje completo: ")
    doc.add_paragraph("Por lo que la distancia 1º eje: ... 2º eje: ...")
    doc.add_paragraph("momento flector máximo: ")
    doc.add_paragraph("Alas: ")
    doc.add_paragraph("Alma: ")
    doc.add_paragraph("Tensión de fluencia: ")
    doc.add_paragraph("Conclusión FS: ")
    for _ in range(11):
        doc.add_paragraph("")
    doc.add_heading("Verificación por Flexión", level=2)
    t = doc.add_table(rows=6, cols=6)
    headers = [
        "Sección",
        "Momento máximo aplicado [kg·cm]",
        "Tensión material [kg/cm²]",
        "Momento resistente necesario",
        "Momento resistente de sección",
        "Coeficiente de seguridad",
    ]
    for c, h in enumerate(headers):
        _set_table_cell_text(t.cell(0, c), h)
    for r in range(1, 6):
        _set_table_cell_text(t.cell(r, 0), ["A-A'", "B-B'", "C-C'", "D-D'", "E-E'"][r - 1])
    doc.save(str(p))
    return str(p)


def default_template_path() -> str:
    here = Path(__file__).resolve()
    candidates: List[Path] = []

    meipass = getattr(sys, "_MEIPASS", None)
    if meipass:
        base = Path(meipass)
        candidates.append(base / "assets" / "templates" / "memoria_base.docx")

    candidates.append(here.parents[3] / "assets" / "templates" / "memoria_base.docx")
    candidates.append(Path.cwd() / "assets" / "templates" / "memoria_base.docx")

    for c in candidates:
        if c.exists():
            return str(c)
    return str(candidates[-1])


def export_memoria_docx(
    out_docx_path: str,
    *,
    template_path: str,
    header: MemoriaHeader,
    caso: MemoriaCaso,
    resultados: MemoriaResultados,
    seccion: Optional[MemoriaSeccion] = None,
    imagenes: Optional[Dict[str, str]] = None,
    extras: Optional[Dict[str, Any]] = None,
) -> None:
    _ensure_docx_lib()
    tpl = ensure_memoria_template(template_path)
    doc = Document(tpl)
    extras = extras or {}
    imgs = imagenes or {}

    # Datos principales
    _replace_or_append_line(
        doc,
        token_matches=["largo de viga"],
        line_builder=lambda p: _set_labeled_number(
            p, "Largo de viga: ", caso.L_viga_total_mm, unit="mm", decimals=0
        ),
    )
    _replace_or_append_line(
        doc,
        token_matches=["configuración"],
        line_builder=lambda p: (_clear_paragraph(p), p.add_run(f"Configuración: {caso.descripcion_config}")),
    )

    kp_x = extras.get("dist_perno_mm")
    if kp_x is None:
        for name, detail in caso.apoyos:
            if (name or "").strip().upper() == "RP1":
                kp_x = _parse_support_x_mm(detail)
                break
    _replace_or_append_line(
        doc,
        token_matches=["distancia perno de enganche", "perno de enganche"],
        line_builder=lambda p: _set_labeled_number(
            p, "Distancia perno de enganche: ", kp_x if kp_x is not None else 0.0, unit="mm", decimals=0
        ),
    )

    _replace_or_append_line(
        doc,
        token_matches=["peso 1º eje", "peso 1° eje", "peso 1er eje"],
        line_builder=lambda p: _set_labeled_number(
            p, "Peso 1º eje completo: ", extras.get("peso_eje1_kg", 0.0), unit="Kg", decimals=2
        ),
    )
    _replace_or_append_line(
        doc,
        token_matches=["peso 2º eje", "peso 2° eje", "peso 2do eje"],
        line_builder=lambda p: _set_labeled_number(
            p, "Peso 2º eje completo: ", extras.get("peso_eje2_kg", 0.0), unit="Kg", decimals=2
        ),
    )

    def _build_dist_line(p):
        _clear_paragraph(p)
        p.add_run("Por lo que la distancia 1º eje: ")
        _append_omml_number(p, _fmt_num(extras.get("dist_eje1_mm", 0.0), 0))
        p.add_run(" mm  2º eje: ")
        _append_omml_number(p, _fmt_num(extras.get("dist_eje2_mm", 0.0), 0))
        p.add_run(" mm")

    _replace_or_append_line(doc, token_matches=["por lo que la distancia"], line_builder=_build_dist_line)

    def _build_mmax_line(p):
        _clear_paragraph(p)
        p.add_run("momento flector máximo: ")
        _append_omml_number(p, _fmt_num(extras.get("mmax_kgcm", 0.0), 2))
        p.add_run(" Kgcm a ")
        _append_omml_number(p, _fmt_num(extras.get("mmax_x_mm", 0.0), 0))
        p.add_run(" mm")

    _replace_or_append_line(doc, token_matches=["momento flector máximo"], line_builder=_build_mmax_line)
    _replace_or_append_line(
        doc,
        token_matches=["alas:"],
        line_builder=lambda p: (_clear_paragraph(p), p.add_run(f"Alas: {extras.get('alas', '—')}")),
    )
    _replace_or_append_line(
        doc,
        token_matches=["alma:"],
        line_builder=lambda p: (_clear_paragraph(p), p.add_run(f"Alma: {extras.get('alma', '—')}")),
    )
    _replace_or_append_line(
        doc,
        token_matches=["tensión de fluencia"],
        line_builder=lambda p: _set_labeled_number(
            p, "Tensión de fluencia: ", extras.get("fy_kgcm2", 0.0), unit="kg/cm²", decimals=2
        ),
    )
    _replace_or_append_line(
        doc,
        token_matches=["conclusión fs", "coeficiente de seguridad"],
        line_builder=lambda p: _set_labeled_number(
            p, "Conclusión FS (mínimo real): ", extras.get("fs_min_real", 0.0), unit="", decimals=2
        ),
    )

    # Tabla de verificación por flexión
    flex_rows = list(extras.get("flex_rows", []))
    target_table = None
    for t in doc.tables:
        if t.rows and t.columns and len(t.columns) >= 6:
            h0 = (t.cell(0, 0).text or "").lower()
            h1 = (t.cell(0, 1).text or "").lower()
            if "sección" in h0 and "momento" in h1:
                target_table = t
                break

    if target_table is None:
        target_table = doc.add_table(rows=6, cols=6)
        headers = [
            "Sección",
            "Momento máximo aplicado [kg·cm]",
            "Tensión material [kg/cm²]",
            "Momento resistente necesario",
            "Momento resistente de sección",
            "Coeficiente de seguridad",
        ]
        for c, h in enumerate(headers):
            _set_table_cell_text(target_table.cell(0, c), h)
        for r in range(1, 6):
            _set_table_cell_text(target_table.cell(r, 0), ["A-A'", "B-B'", "C-C'", "D-D'", "E-E'"][r - 1])

    for i in range(5):
        rr = i + 1
        data = flex_rows[i] if i < len(flex_rows) else {}
        _set_table_cell_text(target_table.cell(rr, 0), data.get("sec", ["A-A'", "B-B'", "C-C'", "D-D'", "E-E'"][i]))
        _set_table_cell_num(target_table.cell(rr, 1), data.get("M_kgcm", 0.0), decimals=2)
        _set_table_cell_num(target_table.cell(rr, 2), data.get("sigma_material_kgcm2", data.get("sigma_max", 0.0)), decimals=2)
        _set_table_cell_num(target_table.cell(rr, 3), data.get("Wreq_cm3", 0.0), decimals=2)
        _set_table_cell_num(target_table.cell(rr, 4), data.get("Wcrit_cm3", 0.0), decimals=2)
        _set_table_cell_num(target_table.cell(rr, 5), data.get("FS", 0.0), decimals=2)

    # Reemplazo de imágenes en placeholders vacíos (11 esperadas, o las que haya)
    ordered_keys = [
        "fbd",
        "v",
        "m",
        "sec_a",
        "sec_b",
        "sec_c",
        "sec_d",
        "sec_e",
        "stab_long",
        "stab_lat",
        "secciones",
    ]
    ordered_paths = [imgs.get(k, "") for k in ordered_keys]
    _replace_inline_images(doc, ordered_paths)

    # Pie con metadata
    doc.add_paragraph("")
    p = doc.add_paragraph("")
    _clear_paragraph(p)
    p.add_run(f"Documento: {header.titulo} | ")
    p.add_run(f"Unidad: {caso.unidad} | ")
    p.add_run(f"Fecha: {(header.fecha or datetime.now()).strftime('%Y-%m-%d %H:%M')}")

    out = Path(out_docx_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    doc.save(str(out))
