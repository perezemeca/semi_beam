from __future__ import annotations

from typing import Iterable, List, Sequence, Tuple


def dist_interval(center: float, length: float) -> Tuple[float, float]:
    if float(length) <= 0.0:
        raise ValueError("La longitud de la carga distribuida debe ser mayor a 0.")
    half = 0.5 * float(length)
    return float(center) - half, float(center) + half


def clip_interval(a: float, b: float, lo: float, hi: float) -> Tuple[float, float]:
    aa = max(float(lo), min(float(hi), float(a)))
    bb = max(float(lo), min(float(hi), float(b)))
    if bb < aa:
        aa, bb = bb, aa
    return aa, bb


def check_no_overlap(
    intervals: Sequence[Tuple[float, float]],
    tol: float = 0.0,
) -> Tuple[bool, List[Tuple[int, int]]]:
    pairs: List[Tuple[int, int]] = []
    tol_v = max(0.0, float(tol))
    ordered = sorted(
        [(idx, float(a), float(b)) for idx, (a, b) in enumerate(intervals)],
        key=lambda item: (item[1], item[2]),
    )
    for i in range(len(ordered)):
        idx_i, a_i, b_i = ordered[i]
        for j in range(i + 1, len(ordered)):
            idx_j, a_j, b_j = ordered[j]
            if a_j >= b_i - tol_v:
                break
            pairs.append((idx_i, idx_j))
    return (len(pairs) == 0), pairs

