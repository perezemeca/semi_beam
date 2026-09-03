from semi_beam.domain.loads import DistUniform, PointForce, PointMoment
from semi_beam.engine.reactions import solve_reactions_2support
import pytest


def test_two_support_point_load_matches_analytic_solution():
    result = solve_reactions_2support(
        1000.0,
        (0.0, 1000.0),
        [PointForce(label="P1", x_mm=250.0, value_user=100.0)],
    )

    assert abs(result.reacciones["R_A"] - 75.0) < 1e-6
    assert abs(result.reacciones["R_B"] - 25.0) < 1e-6
    assert abs(result.Fy_total_residual) < 1e-9
    assert abs(result.M0_residual) < 1e-6


def test_two_support_uniform_load_is_shared_symmetrically():
    result = solve_reactions_2support(
        1200.0,
        (0.0, 1200.0),
        [DistUniform(label="q", x0_mm=0.0, Lq_mm=1200.0, q_user=0.1)],
    )

    assert abs(result.reacciones["R_A"] - 60.0) < 1e-6
    assert abs(result.reacciones["R_B"] - 60.0) < 1e-6
    assert abs(result.Fy_total_residual) < 1e-9
    assert abs(result.M0_residual) < 1e-6


def test_two_support_point_moment_creates_opposite_reactions():
    result = solve_reactions_2support(
        1000.0,
        (0.0, 1000.0),
        [PointMoment(label="M1", x_mm=500.0, M_user_kgmm=1000.0)],
    )

    assert abs(result.reacciones["R_A"] - 1.0) < 1e-6
    assert abs(result.reacciones["R_B"] + 1.0) < 1e-6
    assert abs(result.Fy_total_residual) < 1e-9
    assert abs(result.M0_residual) < 1e-6


def test_two_support_allows_support_beyond_reference_length():
    result = solve_reactions_2support(
        1000.0,
        (0.0, 1200.0),
        [PointForce(label="P1", x_mm=600.0, value_user=100.0)],
    )

    assert abs(result.Fy_total_residual) < 1e-9
    assert abs(result.M0_residual) < 1e-6


def test_two_support_rejects_negative_support_position():
    with pytest.raises(ValueError, match="no pueden ser negativas"):
        solve_reactions_2support(
            1000.0,
            (-1.0, 1200.0),
            [PointForce(label="P1", x_mm=600.0, value_user=100.0)],
        )

