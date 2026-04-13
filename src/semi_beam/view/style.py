from __future__ import annotations
from dataclasses import dataclass

@dataclass(frozen=True)
class RenderStyle:
    beam_lw: float = 2

    arrow_lw: float = 1.0
    arrow_scale: float = 11.0

    dist_rect_lw: float = 0.9
    dist_rect_alpha: float = 0.12

    moment_arc_lw: float = 1.0
    moment_arrow_lw: float = 1.0
    moment_arrow_scale: float = 11.0

    # Alturas (fijas, ajustables por UI)
    arrow_height_pctL: float = 12.0
    dist_height_pctL: float = 12.0
    moment_radius_pctL: float = 8.0

    moment_delta_deg: float = 18.0
    font_size: int = 10

