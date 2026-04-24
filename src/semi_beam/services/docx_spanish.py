from __future__ import annotations

import re

from docx.oxml import OxmlElement
from docx.oxml.ns import qn


LANGUAGE_CODE = "es-AR"

_ACCENT_REPLACEMENTS = {
    "analisis": "análisis",
    "calculo": "cálculo",
    "calculos": "cálculos",
    "configuracion": "configuración",
    "configuraciones": "configuraciones",
    "conclusion": "conclusión",
    "conclusiones": "conclusiones",
    "deflexion": "deflexión",
    "distribucion": "distribución",
    "flexion": "flexión",
    "geometria": "geometría",
    "geometrias": "geometrías",
    "grafico": "gráfico",
    "graficos": "gráficos",
    "lamina": "lámina",
    "laminas": "láminas",
    "maximo": "máximo",
    "maximos": "máximos",
    "memoria de calculo": "memoria de cálculo",
    "minimo": "mínimo",
    "minimos": "mínimos",
    "numero": "número",
    "numeros": "números",
    "pagina": "página",
    "paginas": "páginas",
    "parametro": "parámetro",
    "parametros": "parámetros",
    "seccion": "sección",
    "secciones de viga analizadas": "secciones de viga analizadas",
    "subtitulo": "subtítulo",
    "subtitulos": "subtítulos",
    "tension": "tensión",
    "teorica": "teórica",
    "titulo": "título",
    "titulos": "títulos",
    "verificacion": "verificación",
}

_ACCENT_PATTERNS = [
    (re.compile(rf"\b{re.escape(src)}\b", re.IGNORECASE), dst)
    for src, dst in sorted(_ACCENT_REPLACEMENTS.items(), key=lambda item: len(item[0]), reverse=True)
]


def normalize_spanish_text(text: str) -> str:
    if not text:
        return text

    fixed = text
    for _ in range(2):
        if any(ch in fixed for ch in ("Ã", "Â", "â")):
            try:
                repaired = fixed.encode("latin1").decode("utf-8")
            except UnicodeError:
                break
            if repaired == fixed:
                break
            fixed = repaired

    fixed = fixed.replace("–", "-").replace("\u00a0", " ")
    for pattern, replacement in _ACCENT_PATTERNS:
        fixed = pattern.sub(lambda m: _preserve_case(m.group(0), replacement), fixed)
    return fixed


def localize_docx_to_spanish_ar(doc) -> None:
    _set_core_language(doc)

    seen_parts: set[str] = set()
    for part in doc.part.package.parts:
        root = getattr(part, "element", None)
        if root is None:
            continue
        part_name = str(getattr(part, "partname", ""))
        if part_name in seen_parts:
            continue
        seen_parts.add(part_name)
        _localize_xml_root(root)


def _preserve_case(source: str, replacement: str) -> str:
    if source.isupper():
        return replacement.upper()
    if source[:1].isupper():
        return replacement[:1].upper() + replacement[1:]
    return replacement


def _set_core_language(doc) -> None:
    core = getattr(doc, "core_properties", None)
    if core is None:
        return
    try:
        core.language = LANGUAGE_CODE
    except Exception:
        pass


def _localize_xml_root(root) -> None:
    local_name = root.tag.rsplit("}", 1)[-1]
    if local_name == "styles":
        _localize_styles_root(root)
    elif local_name == "settings":
        _localize_settings_root(root)

    for text_node in root.xpath(".//*[local-name()='t']"):
        if text_node.text:
            text_node.text = normalize_spanish_text(text_node.text)

    for r_pr in root.xpath(".//*[local-name()='rPr']"):
        _ensure_lang_element(r_pr)


def _localize_styles_root(root) -> None:
    doc_defaults = root.find(qn("w:docDefaults"))
    if doc_defaults is None:
        doc_defaults = OxmlElement("w:docDefaults")
        root.insert(0, doc_defaults)

    r_pr_default = doc_defaults.find(qn("w:rPrDefault"))
    if r_pr_default is None:
        r_pr_default = OxmlElement("w:rPrDefault")
        doc_defaults.append(r_pr_default)

    r_pr = r_pr_default.find(qn("w:rPr"))
    if r_pr is None:
        r_pr = OxmlElement("w:rPr")
        r_pr_default.append(r_pr)
    _ensure_lang_element(r_pr)

    for style in root.xpath(".//*[local-name()='style']"):
        style_r_pr = style.find(qn("w:rPr"))
        if style_r_pr is None:
            style_r_pr = OxmlElement("w:rPr")
            style.append(style_r_pr)
        _ensure_lang_element(style_r_pr)


def _localize_settings_root(root) -> None:
    theme_font_lang = root.find(qn("w:themeFontLang"))
    if theme_font_lang is None:
        theme_font_lang = OxmlElement("w:themeFontLang")
        root.append(theme_font_lang)
    theme_font_lang.set(qn("w:val"), LANGUAGE_CODE)
    theme_font_lang.set(qn("w:eastAsia"), LANGUAGE_CODE)
    theme_font_lang.set(qn("w:bidi"), LANGUAGE_CODE)


def _ensure_lang_element(parent) -> None:
    lang = parent.find(qn("w:lang"))
    if lang is None:
        lang = OxmlElement("w:lang")
        parent.append(lang)
    lang.set(qn("w:val"), LANGUAGE_CODE)
    lang.set(qn("w:eastAsia"), LANGUAGE_CODE)
    lang.set(qn("w:bidi"), LANGUAGE_CODE)
