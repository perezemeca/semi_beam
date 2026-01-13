from __future__ import annotations
from dataclasses import dataclass

@dataclass(frozen=True)
class PointForce:
    label: str
    x_mm: float
    value_user: float  # kg (según convención de label)


@dataclass(frozen=True)
class DistUniform:
    label: str
    x0_mm: float
    Lq_mm: float
    q_user: float      # kg/mm (según convención de label; para q: down+)


@dataclass(frozen=True)
class PointMoment:
    label: str
    x_mm: float
    M_user_kgmm: float  # kg·mm (+ CCW / sagging+)


@dataclass(frozen=True)
class NormalizedPointForce:
    label: str
    x_mm: float
    Fy_internal: float
    value_user: float


@dataclass(frozen=True)
class NormalizedDistUniform:
    label: str
    x1_mm: float
    x2_mm: float
    w_up_internal: float
    q_user: float


@dataclass(frozen=True)
class NormalizedPointMoment:
    label: str
    x_mm: float
    M_internal: float
    M_user_kgmm: float
