from __future__ import annotations
from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class UnknownUniformLoad:
    """
    Carga distribuida uniforme faltante (q), con magnitud desconocida.

    Por ahora:
      - Se aplica sobre todo el carrozable: [0, L]
    Preparado para offset:
      - span_start_mm permite desplazar el inicio (futuro).
      - span_len_mm si es None, se asume = L_beam (cubre todo el carrozable).
    """
    label: str = "q"
    span_start_mm: float = 0.0
    span_len_mm: Optional[float] = None
