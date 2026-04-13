from __future__ import annotations
from dataclasses import dataclass
from typing import List, Optional

from semi_beam.domain.loads import PointForce, DistUniform, PointMoment


@dataclass(frozen=True)
class EquilibriumResult:
    x_t_mm: float
    q_user_kg_per_mm: float

    x_d_mm: Optional[float]  # direccional, si aplica

    residual_Fy: float       # debería ~0
    residual_M0: float       # debería ~0 (momento respecto a x=0)

    notes: List[str]

    # Cargas completas "resueltas" (listas para normalizar/diagramas)
    solved_point_forces: List[PointForce]
    solved_dist_loads: List[DistUniform]
    solved_moments: List[PointMoment]
