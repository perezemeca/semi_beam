from __future__ import annotations

from pathlib import Path
import sys
import tempfile

from PySide6.QtCore import Qt
from PySide6.QtGui import QColor, QFont, QImage, QPainter, QPen


def default_icon_path() -> Path:
    here = Path(__file__).resolve()
    meipass = getattr(sys, "_MEIPASS", None)
    if meipass:
        p = Path(meipass) / "assets" / "branding" / "calculeitor.ico"
        if p.exists():
            return p
    return here.parents[3] / "assets" / "branding" / "calculeitor.ico"


def ensure_calculeitor_icon(path: str | Path | None = None) -> str:
    target = Path(path) if path is not None else default_icon_path()
    if target.exists():
        return str(target)

    try:
        target.parent.mkdir(parents=True, exist_ok=True)
    except Exception:
        target = Path(tempfile.gettempdir()) / "calculeitor.ico"

    size = 256
    img = QImage(size, size, QImage.Format_ARGB32)
    img.fill(QColor("#113355"))

    p = QPainter(img)
    p.setRenderHint(QPainter.Antialiasing)
    p.setPen(QPen(QColor("#EAF3FF"), 10))
    p.setBrush(Qt.NoBrush)
    p.drawRoundedRect(10, 10, size - 20, size - 20, 36, 36)

    f = QFont("Cambria Math", 170)
    f.setBold(True)
    p.setFont(f)
    p.setPen(QColor("#FFFFFF"))
    p.drawText(img.rect(), Qt.AlignCenter, "Σ")
    p.end()

    ok = img.save(str(target), "ICO")
    if not ok:
        # fallback raro de plugin: guardar PNG como respaldo visual
        png_target = target.with_suffix(".png")
        img.save(str(png_target), "PNG")
        return str(png_target)
    return str(target)
