from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np


@dataclass(frozen=True)
class DeflectionResult:
    x_mm: np.ndarray
    v_precamber_mm: np.ndarray
    v_load_mm: np.ndarray
    v_total_mm: np.ndarray
    theta_rad: np.ndarray
    vmin_mm: float
    x_vmin_mm: float
    utilized_mm: float
    allowable_mm: float
    limit_y_mm: float
    camber_mid_mm: float
    ok: bool


def _as_strict_arrays(x: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    x_arr = np.asarray(x, dtype=float).reshape(-1)
    y_arr = np.asarray(y, dtype=float).reshape(-1)
    if x_arr.size != y_arr.size:
        raise ValueError("x y M deben tener la misma longitud.")
    if x_arr.size < 2:
        raise ValueError("Se requieren al menos dos puntos para calcular deformada.")
    if not np.all(np.diff(x_arr) > 0.0):
        raise ValueError("x debe ser estrictamente creciente.")
    return x_arr, y_arr


def _cumtrapz(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    out = np.zeros_like(y, dtype=float)
    dx = np.diff(x)
    out[1:] = np.cumsum(0.5 * (y[1:] + y[:-1]) * dx)
    return out


def _coerce_inertia_profile(x: np.ndarray, I: float | np.ndarray) -> np.ndarray:
    if np.isscalar(I):
        i_val = float(I)
        if i_val <= 0.0:
            raise ValueError("I debe ser mayor a 0.")
        return np.full_like(x, i_val, dtype=float)

    i_arr = np.asarray(I, dtype=float).reshape(-1)
    if i_arr.size != x.size:
        raise ValueError("I(x) debe tener la misma longitud que x.")
    if np.any(i_arr <= 0.0):
        raise ValueError("Todos los valores de I(x) deben ser mayores a 0.")
    return i_arr


def precamber_profile(x: np.ndarray, L: float, camber_mid_mm: float = 30.0) -> np.ndarray:
    x_arr = np.asarray(x, dtype=float).reshape(-1)
    L_mm = float(L)
    if L_mm <= 0.0:
        raise ValueError("L debe ser mayor a 0.")
    x0 = float(x_arr[0]) if x_arr.size else 0.0
    xi = x_arr - x0
    return float(camber_mid_mm) * np.sin(np.pi * xi / L_mm)


def compute_deflection_from_moment(
    x: np.ndarray,
    M: np.ndarray,
    E: float,
    I: float | np.ndarray,
    supports: tuple[float, float],
) -> tuple[np.ndarray, np.ndarray]:
    x_arr, m_arr = _as_strict_arrays(x, M)
    e_val = float(E)
    if e_val <= 0.0:
        raise ValueError("E debe ser mayor a 0.")
    i_arr = _coerce_inertia_profile(x_arr, I)

    xa = float(supports[0])
    xb = float(supports[1])
    if not xa < xb:
        raise ValueError("supports debe cumplir xa < xb.")

    kappa = m_arr / (e_val * i_arr)
    theta_raw = _cumtrapz(x_arr, kappa)
    v_raw = _cumtrapz(x_arr, theta_raw)

    va = float(np.interp(xa, x_arr, v_raw))
    vb = float(np.interp(xb, x_arr, v_raw))
    slope_corr = (vb - va) / (xb - xa)
    baseline = va + slope_corr * (x_arr - xa)

    v_load = v_raw - baseline
    theta = theta_raw - slope_corr
    return v_load, theta


def compute_total_deflection(
    x: np.ndarray,
    M: np.ndarray,
    E: float,
    I: float | np.ndarray,
    supports: tuple[float, float],
    camber_mid_mm: float = 30.0,
) -> DeflectionResult:
    x_arr, m_arr = _as_strict_arrays(x, M)
    L_mm = float(x_arr[-1] - x_arr[0])
    if L_mm <= 0.0:
        raise ValueError("La longitud de x debe ser mayor a 0.")

    v_pre = precamber_profile(x_arr, L=L_mm, camber_mid_mm=camber_mid_mm)
    v_load, theta = compute_deflection_from_moment(x_arr, m_arr, E=E, I=I, supports=supports)
    v_total = v_pre + v_load

    idx_min = int(np.argmin(v_total))
    vmin = float(v_total[idx_min])
    x_vmin = float(x_arr[idx_min])
    allowable_mm = 2.0 * float(camber_mid_mm)
    limit_y_mm = -float(camber_mid_mm)
    utilized_mm = float(camber_mid_mm) - vmin
    ok = bool(vmin >= limit_y_mm and utilized_mm <= allowable_mm + 1e-9)

    return DeflectionResult(
        x_mm=x_arr,
        v_precamber_mm=v_pre,
        v_load_mm=v_load,
        v_total_mm=v_total,
        theta_rad=theta,
        vmin_mm=vmin,
        x_vmin_mm=x_vmin,
        utilized_mm=utilized_mm,
        allowable_mm=allowable_mm,
        limit_y_mm=limit_y_mm,
        camber_mid_mm=float(camber_mid_mm),
        ok=ok,
    )
