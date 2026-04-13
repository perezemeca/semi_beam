from semi_beam.domain.loads import DistUniform
from semi_beam.engine.reactions import solve_reactions_3support


def test_three_support_uniform_load_is_symmetric():
    result = solve_reactions_3support(
        1200.0,
        0.0,
        600.0,
        1200.0,
        [DistUniform(label="q", x0_mm=0.0, Lq_mm=1200.0, q_user=0.1)],
    )

    assert abs(result.reacciones["R_k"] - result.reacciones["R_t"]) < 1e-6
    assert abs(result.reacciones["R_k"] - 22.5) < 0.2
    assert abs(result.reacciones["R_d"] - 75.0) < 0.3
    assert abs(result.Fy_total_residual) < 1e-6
    assert abs(result.M0_residual) < 1e-3


def test_three_support_reactions_stay_positive_for_symmetric_uniform_load():
    result = solve_reactions_3support(
        1200.0,
        100.0,
        600.0,
        1100.0,
        [DistUniform(label="q", x0_mm=0.0, Lq_mm=1200.0, q_user=0.1)],
    )

    assert result.reacciones["R_k"] > 0.0
    assert result.reacciones["R_d"] > 0.0
    assert result.reacciones["R_t"] > 0.0
    assert abs(result.Fy_total_residual) < 1e-6
    assert abs(result.M0_residual) < 1e-3

