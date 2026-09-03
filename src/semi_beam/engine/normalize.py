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

def normalize_inputs(
    beam: Beam,
    point_forces: List[PointForce],
    dist_loads: List[DistUniform],
    moments: List[PointMoment],
) -> FBDData:
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


    # 2) Distribuidas uniformes: mantener el tramo real, aunque exceda [0, L].
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

        w_up = to_internal_w_up(label, q_user)  # kg/mm interno (up+)
        n_dists.append(NormalizedDistUniform(
            label=label,
            x1_mm=x1,
            x2_mm=x2,
            w_up_internal=w_up,
            q_user=q_user
        ))

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
