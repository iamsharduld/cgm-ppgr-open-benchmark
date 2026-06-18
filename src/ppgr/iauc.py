"""Postprandial glucose-response (PPGR) outcomes per prereg.md §4.

Implements the field-standard iAUC over 0-120 min (Wolever 2004 / ISO 26642):

  - baseline = mean glucose in the pre-meal window [-15, 0] min  (prereg §4.1)
  - resample the meal window to a fixed 5-min grid by linear interpolation (§4.3)
  - iAUC = trapezoidal AREA ABOVE BASELINE (positive incremental area only) (§4.2)
  - also: net iAUC (signed), peak-rise, time-to-peak (§5.1)

The functions here are dataset-agnostic: they take a CGM series (timestamps +
glucose) and a meal time, and return the outcomes. Cohort adapters feed them.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

# Pre-registered constants (prereg.md §4)
BASELINE_WINDOW_MIN = 15.0   # pre-meal averaging window [-15, 0] min  (§4.1)
WINDOW_MIN = 120.0           # outcome window 0-120 min                 (§4)
GRID_STEP_MIN = 5.0          # 5-min resampling grid                    (§4.3)
MAX_GAP_MIN = 30.0           # inclusion: no interpolation gap > 30 min  (§3.2.1)


@dataclass
class PPGROutcome:
    """Outcomes for a single meal window (prereg §4-§5)."""

    iauc_pos: float          # primary: positive incremental AUC (mg/dL*min)
    iauc_net: float          # secondary: signed net AUC
    baseline: float          # pre-meal baseline glucose (mg/dL)
    peak_rise: float         # max(glucose) - baseline over 0-120 (mg/dL)
    time_to_peak: float      # minutes (5-min grid) of the (earliest) peak
    max_gap_min: float       # largest gap between observed readings in window
    has_t0: bool             # an observed reading at/within grid of t=0 recoverable
    has_t120: bool           # t=120 recoverable without extrapolation
    n_obs_window: int        # observed readings in [0, 120]


def _to_minutes(timestamps: pd.Series, meal_time: pd.Timestamp) -> np.ndarray:
    """Minutes relative to the meal anchor (t=0 at meal_time)."""
    delta = (pd.to_datetime(timestamps) - meal_time).dt.total_seconds().to_numpy()
    return delta / 60.0


def compute_baseline(rel_min: np.ndarray, glucose: np.ndarray) -> float:
    """Baseline = mean glucose in [-15, 0] min (prereg §4.1).

    Fallback (sparse window): the most recent reading at/just before t=0.
    Returns NaN if neither is recoverable.
    """
    in_win = (rel_min >= -BASELINE_WINDOW_MIN) & (rel_min <= 0.0)
    if in_win.any():
        return float(np.mean(glucose[in_win]))
    # fallback: most recent reading before t=0
    before = rel_min <= 0.0
    if before.any():
        idx = np.argmax(rel_min[before])  # largest (closest to 0) among <=0
        return float(glucose[before][idx])
    return float("nan")


def resample_window(
    rel_min: np.ndarray,
    glucose: np.ndarray,
    step: float = GRID_STEP_MIN,
    window: float = WINDOW_MIN,
) -> tuple[np.ndarray, np.ndarray, float, bool, bool]:
    """Linear-interpolate the 0..window window onto a fixed `step`-min grid.

    Returns (grid_minutes, grid_glucose, max_gap_min, has_t0, has_t120).
    No extrapolation: grid points outside [min(obs), max(obs)] become NaN.
    `max_gap_min` is the largest spacing between *observed* readings that the
    grid spans within [0, window] (used by the inclusion filter, prereg §3.2.1).
    """
    grid = np.arange(0.0, window + step / 2.0, step)

    order = np.argsort(rel_min)
    x = rel_min[order]
    y = glucose[order]
    # collapse exact-duplicate timestamps (mean) to keep interp monotone
    if np.any(np.diff(x) == 0):
        ux, inv = np.unique(x, return_inverse=True)
        uy = np.array([y[inv == i].mean() for i in range(len(ux))])
        x, y = ux, uy

    lo, hi = x.min(), x.max()
    grid_glucose = np.interp(grid, x, y, left=np.nan, right=np.nan)
    grid_glucose = np.where((grid >= lo) & (grid <= hi), grid_glucose, np.nan)

    has_t0 = bool(lo <= 0.0 <= hi)
    has_t120 = bool(lo <= window <= hi)

    # largest gap between consecutive observed readings that fall within or
    # bracket the [0, window] window.
    xr = x[(x >= 0.0) & (x <= window)]
    # bracketing points just outside the window matter for interpolation gaps
    below = x[x <= 0.0]
    above = x[x >= window]
    bracket = []
    if below.size:
        bracket.append(below.max())
    bracket.extend(xr.tolist())
    if above.size:
        bracket.append(above.min())
    bracket = np.array(sorted(set(bracket)))
    max_gap = float(np.max(np.diff(bracket))) if bracket.size >= 2 else float("inf")

    return grid, grid_glucose, max_gap, has_t0, has_t120


def trapezoid_iauc_above(grid: np.ndarray, incr: np.ndarray) -> float:
    """Trapezoidal area ABOVE baseline (positive only), Wolever 2004 / ISO 26642.

    `incr` = glucose - baseline on the grid. Segments below baseline contribute
    zero, not negative area. Sub-segment crossings of the baseline are handled
    exactly by splitting each trapezoid at its zero crossing.
    """
    area = 0.0
    for i in range(len(grid) - 1):
        y0, y1 = incr[i], incr[i + 1]
        dt = grid[i + 1] - grid[i]
        if np.isnan(y0) or np.isnan(y1):
            return float("nan")
        if y0 >= 0 and y1 >= 0:
            area += 0.5 * (y0 + y1) * dt
        elif y0 < 0 and y1 < 0:
            pass
        else:
            # one endpoint positive: add only the positive triangle
            if y0 >= 0:  # crosses down to negative
                frac = y0 / (y0 - y1)  # fraction of dt where it hits zero
                area += 0.5 * y0 * (frac * dt)
            else:  # crosses up from negative
                frac = -y0 / (y1 - y0)
                area += 0.5 * y1 * ((1 - frac) * dt)
    return float(area)


def trapezoid_iauc_net(grid: np.ndarray, incr: np.ndarray) -> float:
    """Signed (net) trapezoidal area; below-baseline subtracts (prereg §4.2)."""
    if np.isnan(incr).any():
        return float("nan")
    return float(np.trapezoid(incr, grid))


def compute_ppgr(
    timestamps: pd.Series | np.ndarray,
    glucose: pd.Series | np.ndarray,
    meal_time: pd.Timestamp,
) -> PPGROutcome:
    """Full PPGR outcome for one meal (prereg §4-§5).

    `timestamps`/`glucose` should already be restricted to a generous window
    around the meal (e.g. [-30, +150] min); extra rows are harmless.
    """
    timestamps = pd.to_datetime(pd.Series(timestamps).reset_index(drop=True))
    glucose = pd.Series(glucose).reset_index(drop=True).astype(float)
    keep = glucose.notna()
    timestamps, glucose = timestamps[keep], glucose[keep]

    rel = _to_minutes(timestamps, meal_time)
    g = glucose.to_numpy()

    baseline = compute_baseline(rel, g)
    grid, grid_g, max_gap, has_t0, has_t120 = resample_window(rel, g)

    n_obs_window = int(((rel >= 0.0) & (rel <= WINDOW_MIN)).sum())

    incr = grid_g - baseline
    iauc_pos = trapezoid_iauc_above(grid, incr)
    iauc_net = trapezoid_iauc_net(grid, incr)

    if np.all(np.isnan(incr)):
        peak_rise = float("nan")
        time_to_peak = float("nan")
    else:
        peak_idx = int(np.nanargmax(incr))  # earliest max (argmax tie-break)
        peak_rise = float(incr[peak_idx])
        time_to_peak = float(grid[peak_idx])

    return PPGROutcome(
        iauc_pos=iauc_pos,
        iauc_net=iauc_net,
        baseline=baseline,
        peak_rise=peak_rise,
        time_to_peak=time_to_peak,
        max_gap_min=max_gap,
        has_t0=has_t0,
        has_t120=has_t120,
        n_obs_window=n_obs_window,
    )
