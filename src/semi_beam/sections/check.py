from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Tuple

from semi_beam.sections.i_section import ISection


def _ceil_dec(v: float, dec: int = 2) -> float:
    """Redondeo hacia arriba con 'dec' decimales."""
    p = 10 ** dec
    return math.ceil(float(v) * p) / p


@dataclass(frozen=True)
class SectionCheckRow:
    idx: int
    x_cm: float
    M_kgcm: float
    Jx_cm4: float
    Yc_cm: float
    Wd_cm3: float
    Wn_cm3: float
    n: float


def check_sections(
    *,
    section: ISection,
    sections_input: List[Tuple[float, float]],  # [(x_mm, M_kgcm), ...]
    sigma_adm_kgcm2: float = 3600.0,            # F36 (como tu tabla)
    ceil_decimals: int = 2,
    n_beams: int = 2,                           # ✅ dos vigas idénticas
) -> List[SectionCheckRow]:
    """
    Tabla como referencia:
    - Jx [cm4]
    - Yc [cm]
    - Wd [cm3]
    - Wn [cm3] = |M|/sigma_adm
    - n = Wd/Wn

    Considera n_beams vigas en paralelo:
    - Jx_total = n_beams * Jx_single
    - Wd_total = n_beams * Wd_single
    """
    if n_beams < 1:
        n_beams = 1

    p = section.props_mm()
    Ix_mm4_single = float(p["Ix_mm4"])
    ybar_mm = float(p["ybar_mm"])
    Wd_mm3_single = float(p["Wd_mm3"])

    # ✅ total (dos vigas)
    Ix_mm4 = Ix_mm4_single * float(n_beams)
    Wd_mm3 = Wd_mm3_single * float(n_beams)

    # conversiones a cm
    Jx_cm4 = Ix_mm4 / 1e4        # mm4 -> cm4
    Yc_cm = ybar_mm / 10.0       # mm -> cm
    Wd_cm3 = Wd_mm3 / 1e3        # mm3 -> cm3

    out: List[SectionCheckRow] = []
    for k, (x_mm, M_kgcm) in enumerate(sections_input, start=1):
        x_cm = float(x_mm) / 10.0
        M = float(M_kgcm)

        if abs(sigma_adm_kgcm2) < 1e-12:
            Wn = float("inf")
        else:
            Wn = abs(M) / float(sigma_adm_kgcm2)

        n = float("inf") if Wn == 0.0 else float(Wd_cm3) / float(Wn)

        out.append(
            SectionCheckRow(
                idx=k,
                x_cm=_ceil_dec(x_cm, ceil_decimals),
                M_kgcm=_ceil_dec(M, ceil_decimals),
                Jx_cm4=_ceil_dec(Jx_cm4, ceil_decimals),
                Yc_cm=_ceil_dec(Yc_cm, ceil_decimals),
                Wd_cm3=_ceil_dec(Wd_cm3, ceil_decimals),
                Wn_cm3=_ceil_dec(Wn, ceil_decimals),
                n=_ceil_dec(n, ceil_decimals),
            )
        )

    return out
