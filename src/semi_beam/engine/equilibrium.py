from __future__ import annotations

from typing import List, Optional, Tuple

from semi_beam.domain.cases import BeamCase
from semi_beam.domain.loads import PointForce, DistUniform, PointMoment
from semi_beam.domain.results import EquilibriumResult
from semi_beam.domain.labels import (
    to_internal_Fy,
    to_internal_w_up,
)


def _clip_dist_to_beam(L: float, x0: float, Lq: float) -> Optional[Tuple[float, float]]:
    if Lq <= 0:
        return None
    x1 = x0
    x2 = x0 + Lq
    x1c = max(0.0, min(L, x1))
    x2c = max(0.0, min(L, x2))
    if x2c <= x1c:
        return None
    return x1c, x2c


def _sum_known_contributions(case: BeamCase) -> Tuple[float, float, float, List[str]]:
    """
    Devuelve:
      Fy_known_internal: suma de fuerzas internas (sin q faltante, sin tándem/direccional/hitch/kingpin)
      M0_known_internal: suma de momentos respecto a x=0 (sin q faltante, sin tándem/direccional/hitch/kingpin)
      M_point_total: suma de momentos puntuales ingresados manualmente (solo para info)
    Incluye:
      - point_forces conocidos (aplicando la regla P fuera de [0,L] -> momento en borde, sin fuerza)
      - dist_loads conocidos (recortados a la viga)
      - moments conocidos (M_user ya es interno CCW+)
    """
    L = float(case.beam.L_mm)
    notes: List[str] = []

    Fy = 0.0
    M0 = 0.0
    M_point_total = 0.0

    # Puntuales conocidas: se tratan como fuerzas aunque estén fuera de [0,L]
    for pf in case.point_forces:
        lbl = (pf.label or "").strip()
        x = float(pf.x_mm)
        f = to_internal_Fy(lbl, pf.value_user)

        Fy += f
        M0 += f * x


    # Distribuidas conocidas
    for dl in case.dist_loads:
        clipped = _clip_dist_to_beam(L, float(dl.x0_mm), float(dl.Lq_mm))
        if clipped is None:
            notes.append(f'Distribuida ignorada (no intersecta viga): {dl.label} [{dl.x0_mm:g},{dl.x0_mm + dl.Lq_mm:g}]')
            continue
        x1, x2 = clipped
        w_up = to_internal_w_up(dl.label, float(dl.q_user))  # kg/mm (up+)
        length = x2 - x1
        x_cent = 0.5 * (x1 + x2)
        F_res = w_up * length
        Fy += F_res
        M0 += F_res * x_cent

    # Momentos conocidos (puntuales)
    for pm in case.moments:
        M0 += float(pm.M_user_kgmm)
        M_point_total += float(pm.M_user_kgmm)

    return Fy, M0, M_point_total, notes


def solve_equilibrium(case: BeamCase) -> EquilibriumResult:
    """
    Resuelve:
      - q faltante (uniforme) aplicada sobre [span_start, span_start+span_len]
      - x_t (posición del centro del tándem)

    Ecuaciones:
      ΣFy = 0
      ΣM0 = 0  (respecto a x=0)

    Direccional:
      x_d = x_t - 3075 mm (si existe)
    """
    L = float(case.beam.L_mm)
    notes: List[str] = []

    Fy_known, M0_known, M_point_total, n0 = _sum_known_contributions(case)
    notes.extend(n0)
    if abs(M_point_total) > 0:
        notes.append(f"Momento(s) puntual(es) manual(es) total = {M_point_total:g} kg·mm (incluido en ΣM).")

    # Soportes fijos: kingpin y opcional hitch
    Fy_kp = to_internal_Fy(case.kingpin.label, case.kingpin.reaction_user)
    Fy_known += Fy_kp
    M0_known += Fy_kp * float(case.kingpin.x_mm)

    if case.hitch is not None:
        Fy_hitch = to_internal_Fy(case.hitch.label, case.hitch.reaction_user)
        Fy_known += Fy_hitch
        M0_known += Fy_hitch * float(case.hitch.x_mm)

    # Tándem y direccional (valores conocidos, posiciones dependen de x_t)
    Fy_t = to_internal_Fy(case.tandem.label, case.tandem.reaction_user)
    Fy_d = 0.0
    off = 0.0
    if case.directional is not None:
        Fy_d = to_internal_Fy(case.directional.label, case.directional.reaction_user)
        off = float(case.directional.offset_mm)

    # Tramo de la q faltante
    span_start = float(case.unknown_uniform.span_start_mm)
    span_len = float(case.unknown_uniform.span_len_mm) if case.unknown_uniform.span_len_mm is not None else L
    span_end = span_start + span_len

    span_start_c = max(0.0, min(L, span_start))
    span_end_c = max(0.0, min(L, span_end))
    if span_end_c <= span_start_c:
        raise ValueError("Tramo de la carga uniforme faltante inválido (no intersecta la viga).")

    span_len_c = span_end_c - span_start_c
    x_cent = 0.5 * (span_start_c + span_end_c)

    if abs(span_start_c - span_start) > 1e-9 or abs(span_end_c - span_end) > 1e-9:
        notes.append(f"Tramo q faltante recortado a [{span_start_c:g}, {span_end_c:g}] mm.")

    # Resolver q por ΣFy=0
    Fy_wo_q = Fy_known + Fy_t + Fy_d
    q_user = Fy_wo_q / span_len_c  # para label q (down+): w_up = -q_user

    # Resolver x_t por ΣM0=0
    w_up_q = to_internal_w_up(case.unknown_uniform.label, q_user)
    Fy_q = w_up_q * span_len_c
    M_q = Fy_q * x_cent

    denom = (Fy_t + Fy_d)
    if abs(denom) < 1e-9:
        raise ValueError("No se puede resolver x_t: Fy_t + Fy_d ≈ 0 (denominador nulo).")

    const = M0_known - Fy_d * off + M_q
    x_t = -const / denom

    # Chequeo límites tándem
    if case.tandem.x_min_mm is not None and x_t < float(case.tandem.x_min_mm) - 1e-6:
        notes.append(f"ATENCIÓN: x_t={x_t:g} mm < x_min={case.tandem.x_min_mm:g} mm.")
    if case.tandem.x_max_mm is not None and x_t > float(case.tandem.x_max_mm) + 1e-6:
        notes.append(f"ATENCIÓN: x_t={x_t:g} mm > x_max={case.tandem.x_max_mm:g} mm.")

    x_d = None
    if case.directional is not None:
        x_d = x_t - off

    # Residuales
    Fy_total = Fy_known + Fy_t + Fy_d + Fy_q
    M0_total = M0_known + Fy_t * x_t + (Fy_d * (x_t - off)) + M_q

    # Cargas resueltas para FBD/diagramas
    solved_points: List[PointForce] = []
    solved_points.extend(case.point_forces)
    solved_points.append(PointForce(label=case.kingpin.label, x_mm=case.kingpin.x_mm, value_user=case.kingpin.reaction_user))
    if case.hitch is not None:
        solved_points.append(PointForce(label=case.hitch.label, x_mm=case.hitch.x_mm, value_user=case.hitch.reaction_user))
    solved_points.append(PointForce(label=case.tandem.label, x_mm=x_t, value_user=case.tandem.reaction_user))
    if case.directional is not None and x_d is not None:
        solved_points.append(PointForce(label=case.directional.label, x_mm=x_d, value_user=case.directional.reaction_user))

    solved_dists: List[DistUniform] = []
    solved_dists.extend(case.dist_loads)
    solved_dists.append(DistUniform(
        label=case.unknown_uniform.label,
        x0_mm=span_start_c,
        Lq_mm=span_len_c,
        q_user=q_user
    ))

    solved_moms: List[PointMoment] = []
    solved_moms.extend(case.moments)

    notes.append(f"Solución: q={q_user:g} kg/mm (usuario: down+), x_t={x_t:g} mm.")
    if x_d is not None:
        notes.append(f"Direccional: x_d = x_t - {off:g} => {x_d:g} mm.")

    return EquilibriumResult(
        x_t_mm=x_t,
        q_user_kg_per_mm=q_user,
        x_d_mm=x_d,
        residual_Fy=Fy_total,
        residual_M0=M0_total,
        notes=notes,
        solved_point_forces=solved_points,
        solved_dist_loads=solved_dists,
        solved_moments=solved_moms,
    )
