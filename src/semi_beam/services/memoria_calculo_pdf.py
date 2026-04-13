# path: src/semi_beam/services/memoria_calculo_pdf.py
from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional, Sequence, Tuple

from reportlab.lib import colors, pagesizes
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import Image, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

# Nota: este módulo NO depende de Qt. Acepta paths a imágenes ya generadas
# (FBD, V, M y tabla de secciones) y datos pre-calculados desde el motor/UI.


@dataclass(frozen=True)
class MemoriaHeader:
    titulo: str
    cliente_proyecto: str = ""
    autor: str = ""
    fecha: Optional[datetime] = None
    revision: str = "A"


@dataclass(frozen=True)
class MemoriaCaso:
    unidad: str
    L_carrozable_mm: float
    L_viga_total_mm: float
    descripcion_config: str
    apoyos: Sequence[Tuple[str, str]]  # (nombre, detalle)
    cargas: Sequence[Tuple[str, str]]  # (tipo/label, detalle)


@dataclass(frozen=True)
class MemoriaResultados:
    q_user_kgmm: float
    x_t_mm: float
    x_d_mm: Optional[float]
    residual_Fy: float
    residual_M0: float
    extremos_V: Sequence[Tuple[str, float, float]]  # (tipo max/min, x_mm, V)
    extremos_M: Sequence[Tuple[str, float, float]]  # (tipo max/min, x_mm, M_kgcm)


@dataclass(frozen=True)
class MemoriaSeccion:
    materiales: Sequence[Tuple[str, str]]  # (componente, material/σadm)
    fs_min: float
    n_vigas: int
    parametros: Sequence[Tuple[str, str]]  # (nombre, valor)
    tabla: Sequence[Sequence[str]]          # filas (strings) incluyendo encabezados opcionalmente


def export_memoria_pdf(
    out_pdf_path: str,
    header: MemoriaHeader,
    caso: MemoriaCaso,
    resultados: MemoriaResultados,
    seccion: Optional[MemoriaSeccion] = None,
    imagenes: Optional[Dict[str, str]] = None,
    page_size=pagesizes.A4,
) -> None:
    """
    Genera una Memoria de Cálculo en PDF (A4).

    Compatibilidad:
    - Acepta llamada posicional: export_memoria_pdf(path, header, caso, resultados, seccion, imgs)
    - O por keyword: export_memoria_pdf(path, header=..., caso=..., resultados=..., seccion=..., imagenes=...)
    """

    # Normalizar dict de imágenes (claves y valores)
    imgs = _normalize_images_dict(imagenes)

    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name="H1c", parent=styles["Heading1"], alignment=TA_CENTER))
    styles.add(ParagraphStyle(name="Small", parent=styles["BodyText"], fontSize=9, leading=11))
    styles.add(ParagraphStyle(name="MonoSmall", parent=styles["BodyText"], fontName="Courier", fontSize=8, leading=10))

    doc = SimpleDocTemplate(
        out_pdf_path,
        pagesize=page_size,
        leftMargin=15 * mm,
        rightMargin=15 * mm,
        topMargin=12 * mm,
        bottomMargin=12 * mm,
        title=header.titulo,
    )

    story: List[object] = []

    # ----------------- Portada -----------------
    story.append(Paragraph(header.titulo, styles["H1c"]))
    story.append(Spacer(1, 4 * mm))

    fecha = header.fecha or datetime.now()
    meta_rows = [
        ["Proyecto / Cliente:", header.cliente_proyecto or "-"],
        ["Autor:", header.autor or "-"],
        ["Fecha:", fecha.strftime("%Y-%m-%d %H:%M")],
        ["Revisión:", header.revision],
        ["Unidad / Caso:", caso.unidad],
    ]
    t = Table(meta_rows, colWidths=[40 * mm, 140 * mm])
    t.setStyle(
        TableStyle(
            [
                ("FONTNAME", (0, 0), (-1, -1), "Helvetica"),
                ("FONTSIZE", (0, 0), (-1, -1), 10),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.black),
                ("BACKGROUND", (0, 0), (0, -1), colors.whitesmoke),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("LEFTPADDING", (0, 0), (-1, -1), 6),
                ("RIGHTPADDING", (0, 0), (-1, -1), 6),
                ("TOPPADDING", (0, 0), (-1, -1), 4),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ]
        )
    )
    story.append(t)
    story.append(Spacer(1, 6 * mm))

    story.append(Paragraph("Alcance", styles["Heading2"]))
    story.append(
        Paragraph(
            "Memoria de cálculo para viga carrozable modelada como viga de Euler-Bernoulli, "
            "con cargas puntuales, carga distribuida uniforme y momentos puntuales, "
            "resolviendo equilibrio estático y verificando tensiones a flexión en sección tipo doble T.",
            styles["BodyText"],
        )
    )
    story.append(Spacer(1, 4 * mm))

    # ----------------- Base teórica -----------------
    story.append(Paragraph("Base teórica y supuestos", styles["Heading2"]))
    base = [
        "Hipótesis: material elástico lineal, pequeñas deformaciones, secciones planas permanecen planas (Euler-Bernoulli), "
        "viga prismática y comportamiento cuasi-estático.",
        "Convención interna: fuerzas verticales internas Fy > 0 hacia arriba. Para el usuario, cargas 'P' y 'q' se ingresan con positivo hacia abajo.",
        "Equilibrio: se impone ΣFy = 0 y ΣM0 = 0 (respecto al origen x=0) para resolver incógnitas.",
        "Diagramas: V(x) y M(x) por superposición, usando funciones escalón (Heaviside) para puntuales y expresiones cerradas para distribuidas uniformes.",
        "Verificación: σ = M / W (o σ = M·c/I) y FS = σ_adm / σ, con posibilidad de duplicar inercia por número de vigas en paralelo (n_vigas).",
    ]
    story.extend(_bullets(base, styles))
    story.append(Spacer(1, 3 * mm))

    story.append(Paragraph("Ecuaciones principales", styles["Heading3"]))
    eq = [
        "ΣFy = 0  ⇒  Fy_conocidas + Fy_tándem + Fy_direccional + Fy_q = 0",
        "Fy_q = w_up · L_tramo,   con w_up = -q_user (porque el usuario ingresa q positivo hacia abajo)",
        "⇒ q_user = (Fy_conocidas + Fy_tándem + Fy_direccional) / L_tramo",
        "ΣM0 = 0  ⇒  M0_conocido + Fy_t·x_t + Fy_d·(x_t - off) + M_q = 0",
        "M_q = Fy_q · x_centroide_tramo",
        "⇒ x_t = -(M0_conocido - Fy_d·off + M_q) / (Fy_t + Fy_d)",
    ]
    story.extend(_mono_block(eq, styles))
    story.append(Spacer(1, 3 * mm))

    story.append(PageBreak())

    # ----------------- Datos del caso -----------------
    story.append(Paragraph("Datos del caso", styles["Heading2"]))
    dims = [
        ["L carrozable [mm]", f"{caso.L_carrozable_mm:,.0f}".replace(",", ".")],
        ["L viga total [mm]", f"{caso.L_viga_total_mm:,.0f}".replace(",", ".")],
        ["Configuración", caso.descripcion_config or "-"],
    ]
    t = Table(dims, colWidths=[55 * mm, 125 * mm])
    t.setStyle(_kv_table_style())
    story.append(t)
    story.append(Spacer(1, 3 * mm))

    story.append(Paragraph("Apoyos y reacciones", styles["Heading3"]))
    arows = [["Elemento", "Detalle"]] + [[a, d] for a, d in caso.apoyos]
    t = Table(arows, colWidths=[40 * mm, 140 * mm])
    t.setStyle(_grid_table_style(header_rows=1))
    story.append(t)
    story.append(Spacer(1, 3 * mm))

    story.append(Paragraph("Cargas aplicadas", styles["Heading3"]))
    crows = [["Carga", "Detalle"]] + [[a, d] for a, d in caso.cargas]
    t = Table(crows, colWidths=[40 * mm, 140 * mm])
    t.setStyle(_grid_table_style(header_rows=1))
    story.append(t)
    story.append(Spacer(1, 4 * mm))

    # ----------------- Resultados -----------------
    story.append(Paragraph("Resultados", styles["Heading2"]))
    rrows = [
        ["q calculada [kg/mm] (usuario: down+)", _f(resultados.q_user_kgmm, 6)],
        ["x_t [mm]", _f(resultados.x_t_mm, 0)],
        ["x_d [mm]", "-" if resultados.x_d_mm is None else _f(resultados.x_d_mm, 0)],
        ["Residual ΣFy", _f(resultados.residual_Fy, 6)],
        ["Residual ΣM0", _f(resultados.residual_M0, 6)],
    ]
    t = Table(rrows, colWidths=[80 * mm, 100 * mm])
    t.setStyle(_kv_table_style())
    story.append(t)
    story.append(Spacer(1, 3 * mm))

    if resultados.extremos_V:
        story.append(Paragraph("Extremos locales de V(x)", styles["Heading3"]))
        rows = [["Tipo", "x [mm]", "V"]]
        for kind, x, v in resultados.extremos_V:
            rows.append([kind, _f(x, 0), _f(v, 2)])
        t = Table(rows, colWidths=[30 * mm, 55 * mm, 95 * mm])
        t.setStyle(_grid_table_style(header_rows=1))
        story.append(t)
        story.append(Spacer(1, 2 * mm))

    if resultados.extremos_M:
        story.append(Paragraph("Extremos locales de M(x) [kg·cm]", styles["Heading3"]))
        rows = [["Tipo", "x [mm]", "M [kg·cm]"]]
        for kind, x, m in resultados.extremos_M:
            rows.append([kind, _f(x, 0), _f(m, 2)])
        t = Table(rows, colWidths=[30 * mm, 55 * mm, 95 * mm])
        t.setStyle(_grid_table_style(header_rows=1))
        story.append(t)
        story.append(Spacer(1, 4 * mm))

    # ----------------- Figuras -----------------
    story.append(PageBreak())
    story.append(Paragraph("Figuras", styles["Heading2"]))

    _append_figure(story, styles, "fbd", "Diagrama de cuerpo libre (FBD)", imgs, max_w=180 * mm, max_h=95 * mm)
    _append_figure(story, styles, "v", "Diagrama de cortante V(x)", imgs, max_w=180 * mm, max_h=95 * mm)
    _append_figure(story, styles, "m", "Diagrama de momento M(x)", imgs, max_w=180 * mm, max_h=95 * mm)

    # ----------------- Sección -----------------
    if seccion is not None:
        story.append(PageBreak())
        story.append(Paragraph("Verificación de sección", styles["Heading2"]))

        mr = [["Componente", "Material / σ_adm [kg/cm²]"]] + [[a, b] for a, b in seccion.materiales]
        t = Table(mr, colWidths=[55 * mm, 125 * mm])
        t.setStyle(_grid_table_style(header_rows=1))
        story.append(t)
        story.append(Spacer(1, 2 * mm))

        pr = [
            ["FS mínimo", _f(seccion.fs_min, 2)],
            ["Cantidad de vigas (n)", str(seccion.n_vigas)],
        ]
        for k, v in seccion.parametros:
            pr.append([k, v])
        t = Table(pr, colWidths=[55 * mm, 125 * mm])
        t.setStyle(_kv_table_style())
        story.append(t)
        story.append(Spacer(1, 3 * mm))

        _append_figure(story, styles, "secciones", "Tabla de chequeo (resumen)", imgs, max_w=180 * mm, max_h=110 * mm)

        if seccion.tabla:
            story.append(Paragraph("Detalle tabular", styles["Heading3"]))
            rows = [list(r) for r in seccion.tabla]
            t = Table(rows, repeatRows=1)
            t.setStyle(_grid_table_style(header_rows=1, font_size=8))
            story.append(t)

    doc.build(story)


# ----------------- helpers -----------------

def _normalize_images_dict(imagenes: Optional[Dict[str, str]]) -> Dict[str, str]:
    if not imagenes:
        return {}
    out: Dict[str, str] = {}
    for k, v in imagenes.items():
        kk = (k or "").strip().lower()
        vv = (v or "").strip()
        if not kk or not vv:
            continue
        out[kk] = vv
    # alias comunes
    if "fbd.jpg" in out and "fbd" not in out:
        out["fbd"] = out["fbd.jpg"]
    return out


def _append_figure(story: List[object], styles, key: str, title: str, imgs: Dict[str, str], *, max_w: float, max_h: float):
    story.append(Paragraph(title, styles["Heading3"]))
    path = (imgs.get(key) or "").strip()
    if path and os.path.exists(path):
        story.append(_img(path, max_w=max_w, max_h=max_h))
    else:
        # Dejar evidencia en el PDF si no se insertó la imagen
        story.append(Paragraph(f"(Sin imagen: '{key}' no disponible o no existe en disco)", styles["Small"]))
    story.append(Spacer(1, 3 * mm))


def _f(v: float, dec: int) -> str:
    s = f"{float(v):.{dec}f}"
    if "." in s:
        s = s.rstrip("0").rstrip(".")
    return s


def _bullets(items: List[str], styles):
    out: List[object] = []
    for it in items:
        out.append(Paragraph(f"• {it}", styles["BodyText"]))
        out.append(Spacer(1, 1.2 * mm))
    return out


def _mono_block(lines: List[str], styles):
    out: List[object] = []
    for ln in lines:
        out.append(Paragraph(ln.replace(" ", "&nbsp;"), styles["MonoSmall"]))
    return out


def _kv_table_style():
    return TableStyle(
        [
            ("FONTNAME", (0, 0), (-1, -1), "Helvetica"),
            ("FONTSIZE", (0, 0), (-1, -1), 10),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.black),
            ("BACKGROUND", (0, 0), (0, -1), colors.whitesmoke),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("LEFTPADDING", (0, 0), (-1, -1), 6),
            ("RIGHTPADDING", (0, 0), (-1, -1), 6),
            ("TOPPADDING", (0, 0), (-1, -1), 4),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ]
    )


def _grid_table_style(header_rows: int = 1, font_size: int = 9):
    ts = [
        ("FONTNAME", (0, 0), (-1, -1), "Helvetica"),
        ("FONTSIZE", (0, 0), (-1, -1), font_size),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.black),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("LEFTPADDING", (0, 0), (-1, -1), 4),
        ("RIGHTPADDING", (0, 0), (-1, -1), 4),
        ("TOPPADDING", (0, 0), (-1, -1), 3),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
    ]
    if header_rows > 0:
        ts += [
            ("BACKGROUND", (0, 0), (-1, header_rows - 1), colors.lightgrey),
            ("FONTNAME", (0, 0), (-1, header_rows - 1), "Helvetica-Bold"),
        ]
    return TableStyle(ts)


def _img(path: str, *, max_w: float, max_h: float):
    img = Image(path)
    iw, ih = img.imageWidth, img.imageHeight
    if iw <= 0 or ih <= 0:
        return img
    scale = min(max_w / iw, max_h / ih, 1.0)
    img.drawWidth = iw * scale
    img.drawHeight = ih * scale
    return img
