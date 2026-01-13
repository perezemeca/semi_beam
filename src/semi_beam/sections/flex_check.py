from __future__ import annotations

from dataclasses import dataclass
from typing import Optional
import math

from semi_beam.sections.i_section import ISection


@dataclass
class FlexRowResult:
    # geom / propiedades (para mostrar)
    Jx_cm4: float
    ybar_cm: float
    cmax_cm: float
    Wcrit_cm3: float

    # requerimientos
    Wreq_cm3: float

    # tensiones
    sigma_max_kgcm2: float

    # factor seguridad
    FS: float

    # extras
    sigma_top_kgcm2: Optional[float] = None
    sigma_bot_kgcm2: Optional[float] = None
    FS_top: Optional[float] = None
    FS_bot: Optional[float] = None
    govern_side: str = ""  # "TOP" / "BOT"


def _round_up(x: float, decimals: int) -> float:
    if decimals <= 0:
        return float(math.ceil(x))
    f = 10.0 ** decimals
    return float(math.ceil(x * f) / f)


def _get_first_key(props: dict, keys: list[str]) -> float:
    for k in keys:
        if k in props:
            return float(props[k])
    raise KeyError(f"No se encontró ninguna de las claves {keys} en props_mm(). Keys disponibles: {list(props.keys())}")


def compute_flex_row(
    *,
    section: ISection,
    M_kgcm: float,
    sigma_adm_kgcm2: float,
    n_beams: int = 2,
    round_up_decimals: int = 2,
    sigma_adm_top_kgcm2: Optional[float] = None,
    sigma_adm_bot_kgcm2: Optional[float] = None,
) -> FlexRowResult:
    """
    Verificación elástica a flexión (tensiones normales):
      σ = M*y / I

    Unidades:
      - M_kgcm en kg·cm
      - I en cm^4
      - y en cm
      - σ en kg/cm²

    Si sigma_adm_top/bot se proveen, calcula FS = min(FS_top, FS_bot).
    """
    p = section.props_mm()

    # ✅ Compatibilidad: distintos nombres posibles para I, ybar, H
    I_mm4 = _get_first_key(p, ["Ixx_mm4", "Ix_mm4", "I_mm4", "Ixx", "Ix"])
    ybar_mm = _get_first_key(p, ["ybar_mm", "y_bar_mm", "ybar", "y_mm"])
    H_mm = _get_first_key(p, ["H_mm", "Htot_mm", "H_total_mm", "height_mm"])

    # duplicar por cantidad de vigas
    I_mm4_total = I_mm4 * float(n_beams)

    # conversiones
    I_cm4 = I_mm4_total / (10.0 ** 4)  # mm^4 -> cm^4
    ybar_cm = ybar_mm / 10.0
    H_cm = H_mm / 10.0

    # distancias a fibra extrema
    c_bot_cm = ybar_cm
    c_top_cm = max(0.0, H_cm - ybar_cm)
    cmax_cm = max(c_top_cm, c_bot_cm)

    # módulo resistente crítico global
    Wcrit_cm3 = I_cm4 / max(cmax_cm, 1e-12)

    M = float(M_kgcm)

    # tensión máx global (usando cmax)
    sigma_max = abs(M) / max(Wcrit_cm3, 1e-12)

    # tensiones en fibras extremas (robusto)
    sigma_top = abs(M) * c_top_cm / max(I_cm4, 1e-12)
    sigma_bot = abs(M) * c_bot_cm / max(I_cm4, 1e-12)

    # W requerido
    if sigma_adm_top_kgcm2 is not None and sigma_adm_bot_kgcm2 is not None:
        Wreq_top = abs(M) / max(float(sigma_adm_top_kgcm2), 1e-12)
        Wreq_bot = abs(M) / max(float(sigma_adm_bot_kgcm2), 1e-12)
        Wreq = max(Wreq_top, Wreq_bot)
    else:
        Wreq = abs(M) / max(float(sigma_adm_kgcm2), 1e-12)

    # FS
    govern = ""
    if sigma_adm_top_kgcm2 is not None and sigma_adm_bot_kgcm2 is not None:
        FS_top = float(sigma_adm_top_kgcm2) / max(sigma_top, 1e-12)
        FS_bot = float(sigma_adm_bot_kgcm2) / max(sigma_bot, 1e-12)
        FS = min(FS_top, FS_bot)
        govern = "TOP" if FS_top <= FS_bot else "BOT"
    else:
        FS_top = None
        FS_bot = None
        FS = float(sigma_adm_kgcm2) / max(sigma_max, 1e-12)

    # redondeo hacia arriba (mostrar)
    return FlexRowResult(
        Jx_cm4=_round_up(I_cm4, round_up_decimals),
        ybar_cm=_round_up(ybar_cm, round_up_decimals),
        cmax_cm=_round_up(cmax_cm, round_up_decimals),
        Wcrit_cm3=_round_up(Wcrit_cm3, round_up_decimals),
        Wreq_cm3=_round_up(Wreq, round_up_decimals),
        sigma_max_kgcm2=_round_up(sigma_max, round_up_decimals),
        FS=_round_up(FS, round_up_decimals),

        sigma_top_kgcm2=_round_up(sigma_top, round_up_decimals),
        sigma_bot_kgcm2=_round_up(sigma_bot, round_up_decimals),
        FS_top=_round_up(FS_top, round_up_decimals) if FS_top is not None else None,
        FS_bot=_round_up(FS_bot, round_up_decimals) if FS_bot is not None else None,
        govern_side=govern,
    )
