from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Callable, Dict, List, Optional, Sequence, Tuple

from semi_beam.domain.loads import DistUniform, PointForce, PointMoment
from semi_beam.engine.constraints import check_no_overlap, dist_interval
from semi_beam.engine.reactions import (
    ReactionLoad,
    ReactionsResult,
    solve_reactions_2support,
    solve_reactions_3support,
)


@dataclass(frozen=True)
class SearchVariable:
    name: str
    current: float
    lo: float
    hi: float


@dataclass(frozen=True)
class OptimizerConfig:
    L_mm: float
    support_mode: str
    x_a_mm: Optional[float] = None
    x_b_mm: Optional[float] = None
    x_k_mm: Optional[float] = None
    x_t_mm: Optional[float] = None
    offset_mm: Optional[float] = None
    x_t_min_mm: Optional[float] = None
    x_t_max_mm: Optional[float] = None
    offset_min_mm: float = 3075.0
    offset_max_mm: float = 4000.0
    support_limits: Dict[str, float] = None
    loads: Sequence[ReactionLoad] = ()
    coarse_step_mm: float = 50.0
    refine_step_mm: float = 10.0


@dataclass(frozen=True)
class OptimizerSolution:
    loads: List[ReactionLoad]
    x_t_mm: Optional[float]
    offset_mm: Optional[float]
    reactions: ReactionsResult
    min_margin: float
    feasible: bool
    notes: List[str]


def _frange(lo: float, hi: float, step: float) -> List[float]:
    lo_v = float(lo)
    hi_v = float(hi)
    if hi_v < lo_v:
        lo_v, hi_v = hi_v, lo_v
    if step <= 0.0:
        return [lo_v, hi_v] if hi_v > lo_v else [lo_v]
    out: List[float] = []
    cur = lo_v
    while cur <= hi_v + 1e-9:
        out.append(round(cur, 8))
        cur += float(step)
    if not out or abs(out[-1] - hi_v) > 1e-6:
        out.append(hi_v)
    return out


def _limit_margin(name: str, value: float, limits: Dict[str, float]) -> float:
    limit = float(limits.get(name, 0.0) or 0.0)
    if limit <= 0.0:
        return float("inf")
    return (limit - float(value)) / limit


def _clone_load_with_position(load: ReactionLoad, pos_mm: float) -> ReactionLoad:
    if isinstance(load, PointForce):
        return PointForce(label=load.label, x_mm=float(pos_mm), value_user=float(load.value_user))
    if isinstance(load, DistUniform):
        return DistUniform(label=load.label, x0_mm=float(pos_mm), Lq_mm=float(load.Lq_mm), q_user=float(load.q_user))
    if isinstance(load, PointMoment):
        return PointMoment(label=load.label, x_mm=float(pos_mm), M_user_kgmm=float(load.M_user_kgmm))
    raise TypeError(f"Tipo de carga no soportado: {type(load)!r}")


def _build_dist_intervals(loads: Sequence[ReactionLoad]) -> List[Tuple[float, float]]:
    intervals: List[Tuple[float, float]] = []
    for load in loads:
        if isinstance(load, DistUniform):
            x1 = float(load.x0_mm)
            x2 = float(load.x0_mm) + float(load.Lq_mm)
            intervals.append((x1, x2))
    return intervals


def _evaluate_config(config: OptimizerConfig, loads: Sequence[ReactionLoad], x_t_mm: Optional[float], offset_mm: Optional[float]) -> OptimizerSolution:
    limits = dict(config.support_limits or {})
    notes: List[str] = []

    intervals = _build_dist_intervals(loads)
    ok, pairs = check_no_overlap(intervals)
    if not ok:
        notes.append(f"Hay distribuidas solapadas: {pairs}.")
        return OptimizerSolution(
            loads=list(loads),
            x_t_mm=x_t_mm,
            offset_mm=offset_mm,
            reactions=ReactionsResult(reacciones={}, Fy_total_residual=0.0, M0_residual=0.0, notes=[]),
            min_margin=float("-inf"),
            feasible=False,
            notes=notes,
        )

    if config.support_mode == "2":
        result = solve_reactions_2support(
            config.L_mm,
            (float(config.x_a_mm), float(config.x_b_mm)),
            loads,
        )
        min_margin = min(_limit_margin(name, value, limits) for name, value in result.reacciones.items()) if result.reacciones else float("-inf")
        feasible = all(_limit_margin(name, value, limits) >= 0.0 for name, value in result.reacciones.items())
        return OptimizerSolution(
            loads=list(loads),
            x_t_mm=None,
            offset_mm=None,
            reactions=result,
            min_margin=min_margin,
            feasible=feasible,
            notes=notes + list(result.notes),
        )

    xd = float(x_t_mm) - float(offset_mm)
    result = solve_reactions_3support(
        config.L_mm,
        float(config.x_k_mm),
        xd,
        float(x_t_mm),
        loads,
    )
    min_margin = min(_limit_margin(name, value, limits) for name, value in result.reacciones.items()) if result.reacciones else float("-inf")
    feasible = all(_limit_margin(name, value, limits) >= 0.0 for name, value in result.reacciones.items())
    return OptimizerSolution(
        loads=list(loads),
        x_t_mm=float(x_t_mm),
        offset_mm=float(offset_mm),
        reactions=result,
        min_margin=min_margin,
        feasible=feasible,
        notes=notes + list(result.notes),
    )


def _candidate_variables(config: OptimizerConfig, loads: Sequence[ReactionLoad]) -> List[SearchVariable]:
    vars_out: List[SearchVariable] = []
    L = float(config.L_mm)
    for idx, load in enumerate(loads):
        if isinstance(load, PointForce):
            vars_out.append(SearchVariable(name=f"load:{idx}", current=float(load.x_mm), lo=0.0, hi=L))
        elif isinstance(load, PointMoment):
            vars_out.append(SearchVariable(name=f"load:{idx}", current=float(load.x_mm), lo=0.0, hi=L))
        elif isinstance(load, DistUniform):
            hi = max(0.0, L - float(load.Lq_mm))
            vars_out.append(SearchVariable(name=f"load:{idx}", current=float(load.x0_mm), lo=0.0, hi=hi))

    if config.support_mode == "3":
        vars_out.append(SearchVariable(
            name="x_t",
            current=float(config.x_t_mm),
            lo=float(config.x_t_min_mm),
            hi=float(config.x_t_max_mm),
        ))
        vars_out.append(SearchVariable(
            name="offset",
            current=float(config.offset_mm),
            lo=float(config.offset_min_mm),
            hi=float(config.offset_max_mm),
        ))
    return vars_out


def _apply_variable(loads: Sequence[ReactionLoad], x_t_mm: Optional[float], offset_mm: Optional[float], var: SearchVariable, value: float) -> Tuple[List[ReactionLoad], Optional[float], Optional[float]]:
    next_loads = list(loads)
    next_xt = x_t_mm
    next_offset = offset_mm
    if var.name.startswith("load:"):
        idx = int(var.name.split(":", 1)[1])
        next_loads[idx] = _clone_load_with_position(next_loads[idx], value)
    elif var.name == "x_t":
        next_xt = float(value)
    elif var.name == "offset":
        next_offset = float(value)
    return next_loads, next_xt, next_offset


def search_configuration(
    config: OptimizerConfig,
    *,
    maximize_margin: bool = False,
    progress_callback: Optional[Callable[[int, str], None]] = None,
    is_cancelled: Optional[Callable[[], bool]] = None,
) -> OptimizerSolution:
    limits = dict(config.support_limits or {})
    base_loads = list(config.loads or [])
    vars_out = _candidate_variables(config, base_loads)

    if config.support_mode == "3":
        if config.x_t_min_mm is None or config.x_t_max_mm is None or config.x_t_mm is None or config.offset_mm is None:
            raise ValueError("La búsqueda en 3 apoyos requiere x_t actual, x_t_min, x_t_max y offset actual.")

    initial = _evaluate_config(config, base_loads, config.x_t_mm, config.offset_mm)
    best = initial
    feasible_best = initial if initial.feasible else None

    steps = [float(config.coarse_step_mm), float(config.refine_step_mm)]
    total_iters = max(1, len(vars_out) * len(steps) * 3)
    iter_count = 0

    current_loads = list(base_loads)
    current_xt = config.x_t_mm
    current_offset = config.offset_mm

    for step in steps:
        for _pass in range(3):
            improved = False
            for var in vars_out:
                if is_cancelled is not None and is_cancelled():
                    raise RuntimeError("Búsqueda cancelada por el usuario.")

                choices = _frange(var.lo, var.hi, step)
                current_best = _evaluate_config(config, current_loads, current_xt, current_offset)
                chosen_value = getattr(current_best, "x_t_mm", None)
                local_best = current_best
                local_loads = list(current_loads)
                local_xt = current_xt
                local_offset = current_offset

                for candidate in choices:
                    try:
                        cand_loads, cand_xt, cand_offset = _apply_variable(current_loads, current_xt, current_offset, var, candidate)
                        if config.support_mode == "3":
                            xd = float(cand_xt) - float(cand_offset)
                            if not (float(config.x_k_mm) + 1e-6 < xd < float(cand_xt) - 1e-6):
                                continue
                        evaluated = _evaluate_config(config, cand_loads, cand_xt, cand_offset)
                    except Exception:
                        continue

                    better = False
                    if maximize_margin:
                        better = evaluated.min_margin > local_best.min_margin
                    else:
                        if local_best.feasible:
                            better = evaluated.feasible and evaluated.min_margin > local_best.min_margin
                        else:
                            better = evaluated.feasible or evaluated.min_margin > local_best.min_margin

                    if better:
                        local_best = evaluated
                        local_loads = list(cand_loads)
                        local_xt = cand_xt
                        local_offset = cand_offset

                current_loads = local_loads
                current_xt = local_xt
                current_offset = local_offset

                if feasible_best is None or (local_best.feasible and local_best.min_margin > feasible_best.min_margin):
                    feasible_best = local_best
                if local_best.min_margin > best.min_margin or (local_best.feasible and not best.feasible):
                    best = local_best
                    improved = True

                iter_count += 1
                if progress_callback is not None:
                    pct = min(100, int(round((iter_count / total_iters) * 100.0)))
                    progress_callback(pct, f"Explorando {var.name} con paso {step:g} mm")

            if not improved:
                break

    final_solution = feasible_best if feasible_best is not None else best
    final_notes = list(final_solution.notes)
    if final_solution.feasible:
        final_notes.append("Se encontró una configuración que cumple los límites ingresados.")
    else:
        final_notes.append("No se encontró una configuración factible con la grilla definida.")
    return OptimizerSolution(
        loads=list(final_solution.loads),
        x_t_mm=final_solution.x_t_mm,
        offset_mm=final_solution.offset_mm,
        reactions=final_solution.reactions,
        min_margin=final_solution.min_margin,
        feasible=final_solution.feasible,
        notes=final_notes,
    )

