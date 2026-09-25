import pytest

from src.decision.decision_curve import (
    THRESHOLDS,
    decision_curve,
    expected_net_for_new,
    recommend,
)

PROBA = [0.9, 0.8, 0.3, 0.1]
ACTUAL = [True, False, True, False]


def _point(curve, t):
    return next(p for p in curve if p.threshold == t)


def test_curve_has_101_points_and_strict_threshold():
    curve = decision_curve(PROBA, ACTUAL, 10, 100, 0.5)
    assert len(curve) == 101 == len(THRESHOLDS)
    assert _point(curve, 1.0).flagged == 0
    assert _point(curve, 0.8).flagged == 1          # 0.8 is not > 0.8
    assert _point(curve, 0.0).flagged == 4


def test_counts_and_net_by_hand():
    curve = decision_curve(PROBA, ACTUAL, 10, 100, 0.5)
    p = _point(curve, 0.5)                           # flags 0.9, 0.8
    assert (p.flagged, p.tp, p.fp, p.fn, p.tn) == (2, 1, 1, 1, 1)
    assert p.precision == pytest.approx(0.5)
    assert p.recall == pytest.approx(0.5)
    assert p.net == pytest.approx(1 * 0.5 * 100 - 2 * 10)   # 30
    assert _point(curve, 0.2).net == pytest.approx(2 * 0.5 * 100 - 3 * 10)   # 70
    assert _point(curve, 0.05).net == pytest.approx(2 * 0.5 * 100 - 4 * 10)  # 60


def test_recommend_takes_max_net_and_prefers_higher_threshold_on_ties():
    best = recommend(decision_curve(PROBA, ACTUAL, 10, 100, 0.5))
    assert best.net == pytest.approx(70)
    assert best.threshold == pytest.approx(0.29)     # 0.10..0.29 all flag the same 3 rows


def test_recommend_flags_nobody_when_acting_never_pays():
    best = recommend(decision_curve(PROBA, ACTUAL, 1000, 100, 0.5))
    assert best.threshold == 1.0
    assert best.flagged == 0
    assert best.net == 0


def test_zero_division_cases():
    curve = decision_curve([0.2, 0.4], [False, False], 1, 10, 1.0)
    p = _point(curve, 1.0)
    assert p.precision == 0.0
    assert p.recall == 0.0


def test_expected_net_for_unlabeled_rows():
    flagged, net = expected_net_for_new([0.9, 0.5, 0.2], 0.4, 10, 100, 0.5)
    assert flagged == 2
    assert net == pytest.approx((0.9 * 50 - 10) + (0.5 * 50 - 10))  # 50
