from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np

from semi_beam.domain.loads import DistUniform, PointForce, PointMoment
from semi_beam.domain.labels import to_internal_Fy, to_internal_w_up
from semi_beam.engine.diagrams import VMDiagram


ReactionLoad = Union[PointForce, DistUniform, PointMoment]


@dataclass(frozen=True)
class ReactionsResult:
    reacciones: Dict[str, float]
    Fy_total_residual: float
    M0_residual: float
    notes: List[str]


@dataclass(frozen=True)
class _InternalLoadSet:
    point_forces: List[Tuple[float, float]]
    dist_loads: List[Tuple[float, float, float]]
    moments: List[Tuple[float, float]]
    notes: List[str]


def _clip_dist_to_beam(L: float, x0: float, Lq: float) -> Optional[Tuple[float, float]]:
    if float(Lq) <= 0.0:
        return None
    x1 = float(x0)
    x2 = float(x0) + float(Lq)
    x1c = max(0.0, min(float(L), x1))
    x2c = max(0.0, min(float(L), x2))
    if x2c <= x1c:
        return None
    return x1c, x2c


def _as_internal_loads(L: float, loads: Sequence[ReactionLoad]) -> _InternalLoadSet:
    point_forces: List[Tuple[float, float]] = []
    dist_loads: List[Tuple[float, float, float]] = []
    moments: List[Tuple[float, float]] = []
    notes: List[str] = []

    for load in loads:
        if isinstance(load, PointForce):
            point_forces.append((float(load.x_mm), float(to_internal_Fy(load.label, load.value_user))))
            continue

        if isinstance(load, DistUniform):
            clipped = _clip_dist_to_beam(float(L), float(load.x0_mm), float(load.Lq_mm))
            if clipped is None:
                notes.append(f'Distribuida ignorada: "{load.label}" no intersecta la viga o tiene longitud inválida.')
                continue
            x1, x2 = clipped
            if abs(x1 - float(load.x0_mm)) > 1e-9 or abs(x2 - (float(load.x0_mm) + float(load.Lq_mm))) > 1e-9:
                notes.append(f'Distribuida "{load.label}" recortada a [{x1:g}, {x2:g}] mm para el solver.')
            dist_loads.append((x1, x2, float(to_internal_w_up(load.label, load.q_user))))
            continue

        if isinstance(load, PointMoment):
            moments.append((float(load.x_mm), float(load.M_user_kgmm)))
            continue

        raise TypeError(f"Tipo de carga no soportado: {type(load)!r}")

    return _InternalLoadSet(
        point_forces=point_forces,
        dist_loads=dist_loads,
        moments=moments,
        notes=notes,
    )


def _sum_internal_contributions(
    point_forces: Sequence[Tuple[float, float]],
    dist_loads: Sequence[Tuple[float, float, float]],
    moments: Sequence[Tuple[float, float]],
) -> Tuple[float, float]:
    fy = 0.0
    m0 = 0.0

    for x, force in point_forces:
        fy += float(force)
        m0 += float(force) * float(x)

    for x1, x2, w_up in dist_loads:
        length = float(x2) - float(x1)
        fres = float(w_up) * length
        fy += fres
        m0 += fres * (0.5 * (float(x1) + float(x2)))

    for x, moment in moments:
        m0 += float(moment)

    return fy, m0


def _solve_two_support_reactions(x_a: float, x_b: float, fy_loads: float, m0_loads: float) -> Tuple[float, float]:
    denom = float(x_b) - float(x_a)
    if abs(denom) < 1e-9:
        raise ValueError("Los apoyos están coincidentes o demasiado próximos.")
    rb = (float(fy_loads) * float(x_a) - float(m0_loads)) / denom
    ra = -float(fy_loads) - rb
    return ra, rb


def _build_internal_diagram(
    *,
    L: float,
    point_forces: Sequence[Tuple[float, float]],
    dist_loads: Sequence[Tuple[float, float, float]],
    moments: Sequence[Tuple[float, float]],
) -> VMDiagram:
    pf_x = np.asarray([float(x) for x, _ in point_forces], dtype=float)
    pf_fy = np.asarray([float(fy) for _, fy in point_forces], dtype=float)
    dl_a = np.asarray([float(x1) for x1, _, _ in dist_loads], dtype=float)
    dl_b = np.asarray([float(x2) for _, x2, _ in dist_loads], dtype=float)
    dl_w = np.asarray([float(w_up) for _, _, w_up in dist_loads], dtype=float)
    pm_x = np.asarray([float(x) for x, _ in moments], dtype=float)
    pm_m = np.asarray([float(m) for _, m in moments], dtype=float)
    return VMDiagram(
        x_start=0.0,
        x_end=float(L),
        pf_x=pf_x,
        pf_Fy=pf_fy,
        dl_a=dl_a,
        dl_b=dl_b,
        dl_w=dl_w,
        pm_x=pm_x,
        pm_M=pm_m,
    )


def _integration_grid(
    L: float,
    *,
    point_forces: Sequence[Tuple[float, float]],
    dist_loads: Sequence[Tuple[float, float, float]],
    moments: Sequence[Tuple[float, float]],
    extras: Sequence[float] = (),
    n_per_segment: int = 120,
) -> np.ndarray:
    pts = {0.0, float(L)}
    for x, _ in point_forces:
        pts.add(float(x))
    for x1, x2, _ in dist_loads:
        pts.add(float(x1))
        pts.add(float(x2))
    for x, _ in moments:
        pts.add(float(x))
    for x in extras:
        pts.add(float(x))

    ordered = sorted(pts)
    if len(ordered) == 1:
        return np.asarray(ordered, dtype=float)

    xs: List[float] = [ordered[0]]
    for i in range(len(ordered) - 1):
        a = ordered[i]
        b = ordered[i + 1]
        if b <= a:
            continue
        seg = np.linspace(a, b, int(max(2, n_per_segment)), endpoint=False, dtype=float)
        xs.extend(seg[1:].tolist())
        xs.append(float(b))
    return np.asarray(xs, dtype=float)


def _flexibility_integral(actual: VMDiagram, unit: VMDiagram, grid: np.ndarray) -> float:
    if grid.size < 2:
        return 0.0
    m_actual = actual._eval_M_array(grid)
    m_unit = unit._eval_M_array(grid)
    y = m_actual * m_unit
    dx = np.diff(grid)
    return float(np.sum(0.5 * (y[:-1] + y[1:]) * dx))


def solve_reactions_2support(
    L: float,
    supports: Tuple[float, float],
    loads: Sequence[ReactionLoad],
) -> ReactionsResult:
    L_v = float(L)
    x_a, x_b = float(supports[0]), float(supports[1])
    if L_v <= 0.0:
        raise ValueError("La longitud de la viga debe ser mayor a 0.")
    if not (0.0 <= x_a <= L_v and 0.0 <= x_b <= L_v):
        raise ValueError("Las posiciones de los apoyos deben estar dentro de [0, L].")
    if x_b <= x_a:
        raise ValueError("Se requiere x_A < x_B para el caso de 2 apoyos.")

    internal = _as_internal_loads(L_v, loads)
    fy_loads, m0_loads = _sum_internal_contributions(internal.point_forces, internal.dist_loads, internal.moments)
    ra, rb = _solve_two_support_reactions(x_a, x_b, fy_loads, m0_loads)

    fy_res = ra + rb + fy_loads
    m0_res = (ra * x_a) + (rb * x_b) + m0_loads

    notes = list(internal.notes)
    if ra < 0.0:
        notes.append(f"Advertencia: la reacción R_A resultó negativa ({ra:g} kg).")
    if rb < 0.0:
        notes.append(f"Advertencia: la reacción R_B resultó negativa ({rb:g} kg).")

    return ReactionsResult(
        reacciones={"R_A": ra, "R_B": rb},
        Fy_total_residual=fy_res,
        M0_residual=m0_res,
        notes=notes,
    )


def solve_reactions_3support(
    L: float,
    x_k: float,
    x_d: float,
    x_t: float,
    loads: Sequence[ReactionLoad],
) -> ReactionsResult:
    L_v = float(L)
    xk = float(x_k)
    xd = float(x_d)
    xt = float(x_t)
    if L_v <= 0.0:
        raise ValueError("La longitud de la viga debe ser mayor a 0.")
    if not (0.0 <= xk <= L_v and 0.0 <= xd <= L_v and 0.0 <= xt <= L_v):
        raise ValueError("Las posiciones de los apoyos deben estar dentro de [0, L].")
    if not (xk < xd < xt):
        raise ValueError("Se requiere x_k < x_d < x_t para el caso de 3 apoyos.")

    internal = _as_internal_loads(L_v, loads)
    fy_loads, m0_loads = _sum_internal_contributions(internal.point_forces, internal.dist_loads, internal.moments)

    r_k_base, r_t_base = _solve_two_support_reactions(xk, xt, fy_loads, m0_loads)
    pf_base = list(internal.point_forces) + [(xk, r_k_base), (xt, r_t_base)]
    diag_base = _build_internal_diagram(
        L=L_v,
        point_forces=pf_base,
        dist_loads=internal.dist_loads,
        moments=internal.moments,
    )

    unit_point_force = [(xd, 1.0)]
    fy_unit, m0_unit = _sum_internal_contributions(unit_point_force, [], [])
    r_k_unit, r_t_unit = _solve_two_support_reactions(xk, xt, fy_unit, m0_unit)
    pf_unit = unit_point_force + [(xk, r_k_unit), (xt, r_t_unit)]
    diag_unit = _build_internal_diagram(
        L=L_v,
        point_forces=pf_unit,
        dist_loads=[],
        moments=[],
    )

    grid = _integration_grid(
        L_v,
        point_forces=pf_base + pf_unit,
        dist_loads=internal.dist_loads,
        moments=internal.moments,
        extras=[xd],
    )
    delta_load = _flexibility_integral(diag_base, diag_unit, grid)
    flexibility = _flexibility_integral(diag_unit, diag_unit, grid)
    if abs(flexibility) < 1e-12:
        raise ValueError("No se pudo resolver el apoyo redundante: flexibilidad ~ 0.")

    r_d = -delta_load / flexibility
    r_k, r_t = _solve_two_support_reactions(xk, xt, fy_loads + r_d, m0_loads + (r_d * xd))

    fy_res = r_k + r_d + r_t + fy_loads
    m0_res = (r_k * xk) + (r_d * xd) + (r_t * xt) + m0_loads

    notes = list(internal.notes)
    if r_k < 0.0:
        notes.append(f"Advertencia: la reacción R_k resultó negativa ({r_k:g} kg).")
    if r_d < 0.0:
        notes.append(f"Advertencia: la reacción R_d resultó negativa ({r_d:g} kg).")
    if r_t < 0.0:
        notes.append(f"Advertencia: la reacción R_t resultó negativa ({r_t:g} kg).")

    return ReactionsResult(
        reacciones={"R_k": r_k, "R_d": r_d, "R_t": r_t},
        Fy_total_residual=fy_res,
        M0_residual=m0_res,
        notes=notes,
    )
