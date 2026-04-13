from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple
import numpy as np

from semi_beam.domain.beam import Beam
from semi_beam.domain.loads import PointForce, DistUniform, PointMoment
from semi_beam.engine.normalize import normalize_inputs


@dataclass(frozen=True)
class VMDiagram:
    """
    Diagrama V(x) y M(x) basado en superposición.

    Convención interna:
    - Fuerzas + arriba (Fy_internal)
    - Distribuida interna w_up (kg/mm) + arriba
    - Momento puntual interno M_internal (kg*mm), CCW+ => salto + en M(x)

    Dominio para graficar:
    - x_start .. x_end (puede incluir x<0 y x>L)
    """
    x_start: float
    x_end: float

    # cargas internas
    pf_x: np.ndarray          # posiciones
    pf_Fy: np.ndarray         # Fy_internal (kg)

    dl_a: np.ndarray          # inicio
    dl_b: np.ndarray          # fin
    dl_w: np.ndarray          # w_up_internal (kg/mm)

    pm_x: np.ndarray          # posición
    pm_M: np.ndarray          # M_internal (kg*mm)

    def eval_V(self, x: float) -> float:
        return float(self._eval_V_array(np.asarray([x], dtype=float))[0])

    def eval_M(self, x: float) -> float:
        return float(self._eval_M_array(np.asarray([x], dtype=float))[0])

    # -------------------------
    # Evaluadores vectorizados
    # -------------------------
    def _eval_V_array(self, x: np.ndarray) -> np.ndarray:
        V = np.zeros_like(x, dtype=float)

        # puntuales: V += Fy * H(x-xi)
        if self.pf_x.size:
            H = (x[:, None] >= self.pf_x[None, :]).astype(float)
            V += H @ self.pf_Fy

        # distribuidas uniformes: V += w * clip(x-a, 0, (b-a))
        if self.dl_a.size:
            a = self.dl_a[None, :]
            b = self.dl_b[None, :]
            w = self.dl_w[None, :]
            lx = np.clip(x[:, None] - a, 0.0, b - a)
            V += np.sum(w * lx, axis=1)

        return V

    def _eval_M_array(self, x: np.ndarray) -> np.ndarray:
        M = np.zeros_like(x, dtype=float)

        # puntuales: M += Fy*(x-xi)*H(x-xi)
        if self.pf_x.size:
            dx = (x[:, None] - self.pf_x[None, :])
            H = (dx >= 0.0).astype(float)
            M += np.sum(self.pf_Fy[None, :] * dx * H, axis=1)

        # distribuidas: M += w * integral_a^{min(x,b)} (x-ξ) dξ
        # - x<a: 0
        # - a<=x<=b: w*(x-a)^2/2
        # - x>b: w*(b-a)*(x - (a+b)/2)
        if self.dl_a.size:
            a = self.dl_a[None, :]
            b = self.dl_b[None, :]
            w = self.dl_w[None, :]

            # regiones
            xcol = x[:, None]
            in1 = (xcol >= a) & (xcol <= b)
            in2 = (xcol > b)

            # a<=x<=b
            t = (xcol - a)
            M += np.sum(w * (t * t) * 0.5 * in1, axis=1)

            # x>b
            L = (b - a)
            xc = 0.5 * (a + b)
            M += np.sum(w * L * (xcol - xc) * in2, axis=1)

        # momentos puntuales: M += M0 * H(x-xk)
        if self.pm_x.size:
            Hm = (x[:, None] >= self.pm_x[None, :]).astype(float)
            M += Hm @ self.pm_M

        return M

    # -------------------------
    # Muestreo con discontinuidades (saltos)
    # -------------------------
    def _breakpoints(self) -> np.ndarray:
        xs = [float(self.x_start), float(self.x_end)]
        xs += list(self.pf_x.astype(float))
        xs += list(self.pm_x.astype(float))
        xs += list(self.dl_a.astype(float))
        xs += list(self.dl_b.astype(float))
        xs = np.array(sorted(set(xs)), dtype=float)
        return xs

    def _jump_at(self, x0: float) -> Tuple[float, float]:
        """Saltos exactos en V y M en x0 por puntuales."""
        jV = 0.0
        jM = 0.0
        if self.pf_x.size:
            mask = np.isclose(self.pf_x, x0, atol=1e-9)
            if np.any(mask):
                jV = float(np.sum(self.pf_Fy[mask]))
        if self.pm_x.size:
            mask = np.isclose(self.pm_x, x0, atol=1e-9)
            if np.any(mask):
                jM = float(np.sum(self.pm_M[mask]))
        return jV, jM

    def sample(self, n_per_segment: int = 80) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Devuelve (x, V, M) listos para plot:
        - Incluye duplicación de x en discontinuidades para dibujar saltos verticales.
        """
        bps = self._breakpoints()
        if len(bps) < 2:
            x = np.array([self.x_start, self.x_end], dtype=float)
            V = self._eval_V_array(x)
            M = self._eval_M_array(x)
            return x, V, M

        span = float(max(1.0, self.x_end - self.x_start))
        eps = 1e-6 * span  # ~0.01mm si span=10m, suficientemente chico

        x_out: List[float] = []
        V_out: List[float] = []
        M_out: List[float] = []

        for i in range(len(bps) - 1):
            a = float(bps[i])
            b = float(bps[i + 1])
            if b <= a:
                continue

            xs = np.linspace(a, b, int(n_per_segment), endpoint=False, dtype=float)
            if xs.size:
                Vs = self._eval_V_array(xs)
                Ms = self._eval_M_array(xs)
                x_out.extend(xs.tolist())
                V_out.extend(Vs.tolist())
                M_out.extend(Ms.tolist())

            # en el breakpoint b: agregamos valor izquierdo y, si corresponde, salto
            xL = b - eps
            xL = max(a + 0.5 * eps, xL)  # por seguridad
            VL = float(self.eval_V(xL))
            ML = float(self.eval_M(xL))

            jV, jM = self._jump_at(b)

            if abs(jV) > 1e-12 or abs(jM) > 1e-12:
                # punto izquierdo
                x_out.append(b); V_out.append(VL);      M_out.append(ML)
                # punto derecho (aplicando saltos exactos)
                x_out.append(b); V_out.append(VL + jV); M_out.append(ML + jM)
            else:
                x_out.append(b); V_out.append(VL); M_out.append(ML)

        return np.asarray(x_out, dtype=float), np.asarray(V_out, dtype=float), np.asarray(M_out, dtype=float)


def build_V_M(
    *,
    beam_L_mm: float,
    point_forces: List[PointForce],
    dist_loads: List[DistUniform],
    moments: List[PointMoment],
    x_start: Optional[float] = None,
    x_end: Optional[float] = None,
) -> VMDiagram:
    """
    Construye el diagrama usando las MISMAS reglas de normalización que el FBD.
    """
    beam = Beam(L_mm=float(beam_L_mm))
    data = normalize_inputs(beam, point_forces, dist_loads, moments)

    # Dominio
    xs = [0.0, float(beam_L_mm)]
    xs += [float(p.x_mm) for p in data.point_forces]
    xs += [float(m.x_mm) for m in data.moments]
    for d in data.dist_loads:
        xs += [float(d.x1_mm), float(d.x2_mm)]
    x_min = float(min(xs))
    x_max = float(max(xs))

    if x_start is None:
        x_start = x_min
    if x_end is None:
        x_end = x_max

    # Puntuales internos
    pf_x = np.array([float(p.x_mm) for p in data.point_forces], dtype=float)
    pf_Fy = np.array([float(p.Fy_internal) for p in data.point_forces], dtype=float)

    # Distribuidas internas
    dl_a = np.array([float(d.x1_mm) for d in data.dist_loads], dtype=float)
    dl_b = np.array([float(d.x2_mm) for d in data.dist_loads], dtype=float)
    dl_w = np.array([float(d.w_up_internal) for d in data.dist_loads], dtype=float)

    # Momentos internos
    pm_x = np.array([float(m.x_mm) for m in data.moments], dtype=float)
    pm_M = np.array([float(m.M_internal) for m in data.moments], dtype=float)

    return VMDiagram(
        x_start=float(x_start),
        x_end=float(x_end),
        pf_x=pf_x,
        pf_Fy=pf_Fy,
        dl_a=dl_a,
        dl_b=dl_b,
        dl_w=dl_w,
        pm_x=pm_x,
        pm_M=pm_M,
    )
