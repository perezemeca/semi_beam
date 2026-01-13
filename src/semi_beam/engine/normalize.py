from __future__ import annotations

from typing import List

from semi_beam.domain.beam import Beam
from semi_beam.domain.loads import (
    PointForce, DistUniform, PointMoment,
    NormalizedPointForce, NormalizedDistUniform, NormalizedPointMoment
)
from semi_beam.domain.cases import FBDData
from semi_beam.domain.labels import (
    to_internal_Fy, to_internal_w_up, label_kind, OUTSIDE_AS_EDGE_MOMENT_LABELS
)

def _clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def normalize_inputs(
    beam: Beam,
    point_forces: List[PointForce],
    dist_loads: List[DistUniform],
    moments: List[PointMoment],
) -> FBDData:
    L = float(beam.L_mm)
    notes: List[str] = []

    n_points: List[NormalizedPointForce] = []
    n_dists: List[NormalizedDistUniform] = []
    n_moms: List[NormalizedPointMoment] = []

    # 1) Fuerzas puntuales (NO convertir a momentos equivalentes; mantener x tal cual)
    for pf in point_forces:
        label = (pf.label or "P").strip()
        x = float(pf.x_mm)
        val = float(pf.value_user)
        Fy = to_internal_Fy(label, val)

        n_points.append(NormalizedPointForce(
            label=label,
            x_mm=x,
            Fy_internal=Fy,
            value_user=val
        ))


    # 2) Distribuidas uniformes: recorte a [0, L]
    for dl in dist_loads:
        label = (dl.label or "q").strip()
        x0 = float(dl.x0_mm)
        Lq = float(dl.Lq_mm)
        q_user = float(dl.q_user)  # kg/mm

        if Lq <= 0:
            notes.append(f'Distribuida inválida (label="{label}"): Lq<=0 (Lq={Lq:g} mm). Se ignoró.')
            continue

        x1 = x0
        x2 = x0 + Lq

        x1c = _clamp(x1, 0.0, L)
        x2c = _clamp(x2, 0.0, L)
        if x2c <= x1c + 1e-9:
            notes.append(f'Distribuida fuera de la viga (label="{label}", [{x1:g},{x2:g}] mm): se ignoró.')
            continue

        w_up = to_internal_w_up(label, q_user)  # kg/mm interno (up+)
        n_dists.append(NormalizedDistUniform(
            label=label,
            x1_mm=x1c,
            x2_mm=x2c,
            w_up_internal=w_up,
            q_user=q_user
        ))

        if abs(x1c - x1) > 1e-9 or abs(x2c - x2) > 1e-9:
            notes.append(f'Distribuida recortada (label="{label}") de [{x1:g},{x2:g}] mm a [{x1c:g},{x2c:g}] mm.')

    # 3) Momentos puntuales (mantener x tal cual)
    for pm in moments:
        label = (pm.label or "M").strip()
        x = float(pm.x_mm)
        M_user = float(pm.M_user_kgmm)
        M_internal = M_user

        n_moms.append(NormalizedPointMoment(
            label=label,
            x_mm=x,
            M_internal=M_internal,
            M_user_kgmm=M_user
        ))


    return FBDData(
        beam=beam,
        point_forces=n_points,
        dist_loads=n_dists,
        moments=n_moms,
        notes=notes
    )
