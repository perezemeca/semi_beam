import math

from semi_beam.sections.flex_check import M_ZERO_TOL_KGCM, compute_flex_row
from semi_beam.sections.i_section import ISection


def _section() -> ISection:
    return ISection(
        b_f_mm=120.0,
        t_top_mm=10.0,
        t_bot_mm=10.0,
        h_web_mm=300.0,
        t_web_mm=6.0,
    )


def _compute(moment_kgcm: float):
    return compute_flex_row(
        section=_section(),
        M_kgcm=moment_kgcm,
        sigma_adm_kgcm2=1400.0,
        sigma_adm_top_kgcm2=1000.0,
        sigma_adm_bot_kgcm2=2000.0,
    )


def _assert_no_flex_demand(result) -> None:
    assert result.Wreq_cm3 == 0.0
    assert result.sigma_max_kgcm2 == 0.0
    assert result.sigma_top_kgcm2 == 0.0
    assert result.sigma_bot_kgcm2 == 0.0
    assert math.isinf(result.FS)
    assert math.isinf(result.FS_top)
    assert math.isinf(result.FS_bot)
    assert result.govern_side == ""


def test_compute_flex_row_zero_moment_returns_infinite_fs():
    result = _compute(0.0)

    _assert_no_flex_demand(result)


def test_compute_flex_row_tiny_moment_inside_tolerance_returns_infinite_fs():
    result = _compute(M_ZERO_TOL_KGCM / 2.0)

    _assert_no_flex_demand(result)


def test_compute_flex_row_nonzero_moment_keeps_finite_flex_check():
    result = _compute(125000.0)

    assert math.isfinite(result.FS)
    assert result.sigma_max_kgcm2 > 0.0
    assert result.Wreq_cm3 > 0.0
    assert result.govern_side == "TOP"
