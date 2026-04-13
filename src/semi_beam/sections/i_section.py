from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

IN_TO_MM = 25.4


def _rect_Ix_about_centroid(b: float, h: float) -> float:
    """Ix de un rectángulo b (ancho) x h (alto), respecto a su centroide (eje horizontal)."""
    return (b * h**3) / 12.0


@dataclass(frozen=True)
class ISection:
    """
    Sección doble T idealizada: 3 rectángulos (planchuela inf + alma + planchuela sup).
    Todas las dimensiones en mm.

    Convención de y: y=0 en la cara inferior, y positivo hacia arriba.
    """
    b_f_mm: float       # ancho planchuelas
    t_top_mm: float     # espesor planchuela superior
    t_bot_mm: float     # espesor planchuela inferior
    h_web_mm: float     # altura libre del alma entre planchuelas
    t_web_mm: float     # espesor del alma

    @property
    def H_mm(self) -> float:
        return float(self.t_bot_mm + self.h_web_mm + self.t_top_mm)

    def props_mm(self) -> Dict[str, float]:
        """
        Devuelve propiedades geométricas (mm / mm^4 / mm^3):
          - H_mm
          - ybar_mm (desde base)
          - Ix_mm4 (sobre eje x que pasa por el centroide)
          - c_top_mm, c_bot_mm, c_max_mm
          - W_top_mm3, W_bot_mm3, Wcrit_mm3  (elástico)
        """
        b = float(self.b_f_mm)
        ttop = float(self.t_top_mm)
        tbot = float(self.t_bot_mm)
        h = float(self.h_web_mm)
        tw = float(self.t_web_mm)

        H = self.H_mm

        # Áreas
        A_bot = b * tbot
        A_web = tw * h
        A_top = b * ttop
        A_tot = A_bot + A_web + A_top

        # Centroides (y desde base)
        y_bot = tbot / 2.0
        y_web = tbot + h / 2.0
        y_top = tbot + h + ttop / 2.0

        # Eje neutro
        ybar = (A_bot * y_bot + A_web * y_web + A_top * y_top) / A_tot

        # Ix por teorema de ejes paralelos
        Ix_bot = _rect_Ix_about_centroid(b, tbot) + A_bot * (y_bot - ybar) ** 2
        Ix_web = _rect_Ix_about_centroid(tw, h) + A_web * (y_web - ybar) ** 2
        Ix_top = _rect_Ix_about_centroid(b, ttop) + A_top * (y_top - ybar) ** 2
        Ix = Ix_bot + Ix_web + Ix_top

        # Distancias a fibras extremas
        c_bot = ybar
        c_top = H - ybar
        c_max = max(c_bot, c_top)

        # Módulos resistentes elásticos
        W_bot = Ix / c_bot if c_bot > 0 else 0.0
        W_top = Ix / c_top if c_top > 0 else 0.0
        Wcrit = Ix / c_max if c_max > 0 else 0.0

        return {
            "H_mm": H,
            "ybar_mm": ybar,
            "Ix_mm4": Ix,
            "c_bot_mm": c_bot,
            "c_top_mm": c_top,
            "c_max_mm": c_max,
            "W_bot_mm3": W_bot,
            "W_top_mm3": W_top,
            "Wcrit_mm3": Wcrit,
        }
