from semi_beam.engine.constraints import check_no_overlap, dist_interval


def test_dist_interval_from_center_and_length():
    a, b = dist_interval(500.0, 200.0)
    assert a == 400.0
    assert b == 600.0


def test_check_no_overlap_detects_pairs():
    ok, pairs = check_no_overlap([(0.0, 200.0), (150.0, 300.0), (320.0, 400.0)])
    assert ok is False
    assert pairs == [(0, 1)]


def test_check_no_overlap_allows_touching_edges():
    ok, pairs = check_no_overlap([(0.0, 200.0), (200.0, 350.0)])
    assert ok is True
    assert pairs == []

