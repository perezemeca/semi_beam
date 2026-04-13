from semi_beam.domain.beam import Beam
from semi_beam.domain.loads import PointForce, DistUniform, PointMoment
from semi_beam.domain.supports import FixedSupport, TandemSupport, DirectionalSupport
from semi_beam.domain.unknowns import UnknownUniformLoad
from semi_beam.domain.cases import BeamCase
from semi_beam.engine.equilibrium import solve_equilibrium


beam = Beam(L_mm=12000)

case = BeamCase(
    beam=beam,
    point_forces=[
        # Cargas al frente (ejemplo): P1=2000 kg down+ en x=2000 mm
        PointForce(label="P1", x_mm=2000, value_user=2000),
    ],
    dist_loads=[],
    moments=[],
    kingpin=FixedSupport(label="Rp1", x_mm=1500, reaction_user=3500),  # up+
    tandem=TandemSupport(label="Rt", reaction_user=5000, x_min_mm=5000, x_max_mm=11000),
    directional=DirectionalSupport(label="Rd", reaction_user=800, offset_mm=3075),
    hitch=None,
    unknown_uniform=UnknownUniformLoad(label="q", span_start_mm=0.0, span_len_mm=None),  # todo [0,L]
)

res = solve_equilibrium(case)
print("x_t [mm] =", res.x_t_mm)
print("q [kg/mm] =", res.q_user_kg_per_mm)
print("residual Fy =", res.residual_Fy)
print("residual M0 =", res.residual_M0)
print("\n".join(res.notes))
