"""Unit tests for the iAUC module against hand-computed synthetic curves.

Run: PYTHONPATH=src python3 -m pytest tests/test_iauc.py -q
 or: PYTHONPATH=src python3 tests/test_iauc.py
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ppgr.iauc import (
    GRID_STEP_MIN,
    compute_baseline,
    compute_ppgr,
    resample_window,
    trapezoid_iauc_above,
    trapezoid_iauc_net,
)

MEAL = pd.Timestamp("2020-01-01 12:00:00")


def _series(rel_min, glucose):
    ts = [MEAL + pd.Timedelta(minutes=float(m)) for m in rel_min]
    return pd.Series(ts), pd.Series(glucose, dtype=float)


def test_triangle_known_iauc():
    """Triangle: baseline 100, rises +60 at t=60, back to baseline at t=120.

    Area above baseline of a triangle = 0.5 * base * height
      = 0.5 * 120 min * 60 mg/dL = 3600 mg/dL*min.
    Pre-meal flat at 100 for the baseline window.
    """
    rel = list(range(-15, 0, 5)) + list(range(0, 125, 5))
    glu = []
    for m in rel:
        if m <= 0:
            glu.append(100.0)
        elif m <= 60:
            glu.append(100.0 + 60.0 * (m / 60.0))
        else:
            glu.append(100.0 + 60.0 * (1 - (m - 60) / 60.0))
    ts, g = _series(rel, glu)
    out = compute_ppgr(ts, g, MEAL)
    assert abs(out.baseline - 100.0) < 1e-9
    assert abs(out.iauc_pos - 3600.0) < 1e-6, out.iauc_pos
    assert abs(out.iauc_net - 3600.0) < 1e-6, out.iauc_net
    assert abs(out.peak_rise - 60.0) < 1e-9
    assert abs(out.time_to_peak - 60.0) < 1e-9


def test_flat_curve_zero_iauc():
    """Flat glucose => zero iAUC, zero peak rise."""
    rel = list(range(-15, 125, 5))
    ts, g = _series(rel, [95.0] * len(rel))
    out = compute_ppgr(ts, g, MEAL)
    assert abs(out.iauc_pos) < 1e-9
    assert abs(out.iauc_net) < 1e-9
    assert abs(out.peak_rise) < 1e-9


def test_positive_only_truncates_below_baseline():
    """A symmetric dip below then equal rise above: iauc_pos counts only the rise.

    First half (0-60) dips to -40 and back; second half (60-120) rises to +40
    and back. Positive area = 0.5*60*40 = 1200; net area = 0 (symmetric).
    """
    rel = list(range(-15, 1, 5))
    glu = [100.0] * len(rel)
    # 0..60 a downward triangle to -40 at t=30
    for m in range(5, 65, 5):
        glu.append(100.0 + (-40.0) * (1 - abs(m - 30) / 30.0))
        rel.append(m)
    # 60..120 an upward triangle to +40 at t=90
    for m in range(65, 125, 5):
        glu.append(100.0 + (40.0) * (1 - abs(m - 90) / 30.0))
        rel.append(m)
    ts, g = _series(rel, glu)
    out = compute_ppgr(ts, g, MEAL)
    assert abs(out.iauc_pos - 1200.0) < 1e-6, out.iauc_pos
    assert abs(out.iauc_net - 0.0) < 1e-6, out.iauc_net


def test_baseline_window_mean():
    """Baseline = mean over [-15, 0]; values outside are ignored."""
    rel = [-20, -15, -10, -5, 0, 30]
    glu = [200.0, 90.0, 100.0, 110.0, 120.0, 150.0]  # [-15,0] mean = (90+100+110+120)/4=105
    ts, g = _series(rel, glu)
    assert abs(compute_baseline((g.index - g.index).to_numpy() * 0 + np.array(rel),
                                np.array(glu)) - 105.0) < 1e-9


def test_max_gap_detection():
    """A 35-min gap inside the window is detected; t0/t120 recoverable flags."""
    rel = [-5, 0, 5, 40, 45, 120]  # gap 5->40 = 35 min
    glu = [100.0, 100.0, 110.0, 120.0, 120.0, 100.0]
    ts, g = _series(rel, glu)
    out = compute_ppgr(ts, g, MEAL)
    assert out.max_gap_min >= 35.0 - 1e-9, out.max_gap_min
    assert out.has_t0 and out.has_t120


def test_no_t120_when_window_short():
    """If observations end before t=120, has_t120 is False and iAUC is NaN."""
    rel = list(range(-15, 95, 5))  # ends at t=90
    ts, g = _series(rel, [100.0 + 0.2 * m if m > 0 else 100.0 for m in rel])
    out = compute_ppgr(ts, g, MEAL)
    assert out.has_t120 is False
    assert np.isnan(out.iauc_pos)


def test_trapezoid_helpers_direct():
    grid = np.array([0.0, 60.0, 120.0])
    incr = np.array([0.0, 60.0, 0.0])
    assert abs(trapezoid_iauc_above(grid, incr) - 3600.0) < 1e-9
    assert abs(trapezoid_iauc_net(grid, incr) - 3600.0) < 1e-9
    # straddling segment
    grid2 = np.array([0.0, 10.0])
    incr2 = np.array([10.0, -10.0])  # crosses zero at midpoint; pos triangle 0.5*10*5=25
    assert abs(trapezoid_iauc_above(grid2, incr2) - 25.0) < 1e-9


def test_resample_grid_shape():
    rel = np.array(list(range(-15, 125, 5)), dtype=float)
    glu = np.full_like(rel, 100.0)
    grid, gg, _, has0, has120 = resample_window(rel, glu)
    assert grid[0] == 0.0 and grid[-1] == 120.0
    assert len(grid) == int(120 / GRID_STEP_MIN) + 1
    assert has0 and has120


if __name__ == "__main__":
    import traceback

    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    passed = 0
    for fn in fns:
        try:
            fn()
            print(f"PASS {fn.__name__}")
            passed += 1
        except Exception:
            print(f"FAIL {fn.__name__}")
            traceback.print_exc()
    print(f"\n{passed}/{len(fns)} tests passed")
    raise SystemExit(0 if passed == len(fns) else 1)
