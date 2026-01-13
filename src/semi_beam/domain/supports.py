from __future__ import annotations
from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class FixedSupport:
    """
    Soporte con posición conocida.
    reaction_user: valor ingresado por usuario según convención del label (kg)
    """
    label: str
    x_mm: float
    reaction_user: float


@dataclass(frozen=True)
class TandemSupport:
    """
    Tándem: reacción conocida, posición desconocida (x_t).
    """
    label: str  # típico "Rt"
    reaction_user: float
    x_min_mm: Optional[float] = None
    x_max_mm: Optional[float] = None


@dataclass(frozen=True)
class DirectionalSupport:
    """
    Direccional: reacción conocida, posición dependiente del tándem.
    Por regla: x_d = x_t - offset_mm
    """
    label: str  # típico "Rd"
    reaction_user: float
    offset_mm: float = 3075.0
