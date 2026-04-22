import numpy as np

from semi_beam.engine.deflection import (
    compute_deflection_from_moment,
    compute_total_deflection,
    precamber_profile,
)


def test_precamber_profile_midspan_is_30mm():
    x = np.linspace(0.0, 1000.0, 11)
    v0 = precamber_profile(x, L=1000.0, camber_mid_mm=30.0)
    assert abs(v0[0]) < 1e-9
    assert abs(v0[-1]) < 1e-9
    assert abs(v0[5] - 30.0) < 1e-9


def test_deflection_supports_zero_for_simple_case():
    x = np.linspace(0.0, 1000.0, 201)
    M = x * (1000.0 - x)
    v_load, _theta = compute_deflection_from_moment(
        x,
        M,
        E=2.1e4,
        I=1.0e8,
        supports=(0.0, 1000.0),
    )

    assert abs(v_load[0]) < 1e-6
    assert abs(v_load[-1]) < 1e-6


def test_total_deflection_criteria_ok_simple_case():
    x = np.linspace(0.0, 1000.0, 201)
    M = np.zeros_like(x)
    result = compute_total_deflection(
        x,
        M,
        E=2.1e4,
        I=1.0e8,
        supports=(0.0, 1000.0),
        camber_mid_mm=30.0,
    )

    assert result.ok is True
    assert result.vmin_mm >= -30.0
    assert result.utilized_mm <= 60.0


def test_deflection_accepts_variable_inertia_profile():
    x = np.linspace(0.0, 1000.0, 201)
    M = x * (1000.0 - x)
    I = np.linspace(8.0e7, 1.2e8, x.size)

    result = compute_total_deflection(
        x,
        M,
        E=2.1e4,
        I=I,
        supports=(0.0, 1000.0),
        camber_mid_mm=30.0,
    )

    assert result.v_total_mm.shape == x.shape
    assert np.isfinite(result.v_total_mm).all()
