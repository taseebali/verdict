"""Turn scores and business costs into a how-many-to-act-on decision.

A row is flagged when its probability is strictly above the threshold, so
threshold 1.00 always flags nobody.
"""
from dataclasses import dataclass

import numpy as np

THRESHOLDS = np.round(np.arange(101) / 100, 2)


@dataclass(frozen=True)
class CurvePoint:
    threshold: float
    flagged: int
    tp: int
    fp: int
    fn: int
    tn: int
    precision: float
    recall: float
    net: float


def decision_curve(proba, actual, action_cost: float, saved_value: float,
                   success_rate: float) -> list[CurvePoint]:
    proba = np.asarray(proba, dtype=float)
    actual = np.asarray(actual, dtype=bool)
    n, positives = len(actual), int(actual.sum())
    points = []
    for t in THRESHOLDS:
        flag = proba > t
        flagged = int(flag.sum())
        tp = int((flag & actual).sum())
        fp, fn = flagged - tp, positives - tp
        points.append(CurvePoint(
            threshold=float(t),
            flagged=flagged,
            tp=tp,
            fp=fp,
            fn=fn,
            tn=n - flagged - fn,
            precision=tp / flagged if flagged else 0.0,
            recall=tp / positives if positives else 0.0,
            net=float(tp * success_rate * saved_value - flagged * action_cost),
        ))
    return points


def recommend(curve: list[CurvePoint]) -> CurvePoint:
    """Highest net value (ties -> higher threshold). If nothing pays, act on nobody."""
    best = max(curve, key=lambda p: (p.net, p.threshold))
    if best.net <= 0:
        return next(p for p in curve if p.threshold == 1.0)
    return best


def expected_net_for_new(proba, threshold: float, action_cost: float, saved_value: float,
                         success_rate: float) -> tuple[int, float]:
    """For rows without known outcomes: expected saves come from the probabilities themselves."""
    p = np.asarray(proba, dtype=float)
    flag = p > threshold
    return int(flag.sum()), float((p[flag] * success_rate * saved_value - action_cost).sum())
