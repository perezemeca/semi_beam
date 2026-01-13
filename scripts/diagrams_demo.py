from semi_beam.domain.beam import Beam
from semi_beam.domain.loads import PointForce, DistUniform, PointMoment
from semi_beam.engine.diagrams import build_V_M

beam = Beam(L_mm=10000)

point_forces = [
    PointForce(label="P1", x_mm=2000, value_user=1000),   # down+
    PointForce(label="Rp1", x_mm=0, value_user=1500),     # up+
]
dist_loads = [
    DistUniform(label="q", x0_mm=0, Lq_mm=10000, q_user=0.05)  # down+
]
moments = [
    PointMoment(label="M1", x_mm=5000, M_user_kgmm=200000)     # CCW+
]

diag = build_V_M(
    beam_L_mm=beam.L_mm,
    point_forces=point_forces,
    dist_loads=dist_loads,
    moments=moments,
    x_start=0.0,
    x_end=beam.L_mm
)

x, V, M = diag.sample(n_per_segment=50)
print("V(0) =", diag.eval_V(0))
print("M(0) =", diag.eval_M(0))
print("V(L) =", diag.eval_V(beam.L_mm))
print("M(L) =", diag.eval_M(beam.L_mm))
print("M(8000) =", diag.eval_M(8000))
