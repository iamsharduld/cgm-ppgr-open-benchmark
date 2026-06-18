"""Per-meal inclusion filters (prereg.md §3.2) + attrition accounting.

A logged meal enters the analysis only if ALL hold:
  1. Valid CGM coverage 0-120 min: t=0 and t=120 recoverable, and no
     interpolation gap > 30 min (§3.2.1).
  2. Known carbohydrate (required) (§3.2.2).
  3. No overlapping meal in (0, 120] min AND no logged meal in the preceding
     120 min (prior-meal washout) (§3.2.3).

The filters are applied in this order; attrition is counted as a CONSORT-style
funnel (total -> passed each criterion) overall and per subject.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from ppgr.iauc import MAX_GAP_MIN, WINDOW_MIN, compute_ppgr
from ppgr.loader import Cohort

WASHOUT_MIN = 120.0   # prior-meal washout (prereg §3.2.3)
OVERLAP_MIN = 120.0   # no other meal in (0, 120] (prereg §3.2.3)
# generous CGM read window so baseline (-15) and bracketing points are present
READ_LO_MIN = -30.0
READ_HI_MIN = 150.0


@dataclass
class Attrition:
    """CONSORT-style funnel counts (prereg §3.3 reporting)."""

    total: int = 0
    pass_carb: int = 0
    pass_no_overlap: int = 0
    pass_washout: int = 0
    pass_cgm_coverage: int = 0           # t0 & t120 recoverable
    pass_gap: int = 0                    # max gap <= 30 min  (=> included)
    per_subject: dict = field(default_factory=dict)


def _neighbor_meal_flags(meal_times: pd.Series) -> tuple[np.ndarray, np.ndarray]:
    """For each meal (sorted within subject), whether a meal falls in (0,120]
    after it (overlap) and in [-120,0) before it (washout violation)."""
    n = len(meal_times)
    no_overlap = np.ones(n, dtype=bool)
    washout_ok = np.ones(n, dtype=bool)
    # unit-safe seconds-since-epoch (meal_time may be datetime64[us])
    secs = (
        pd.to_datetime(meal_times).astype("datetime64[ns]").astype("int64").to_numpy()
        / 1e9
    )
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            d = (secs[j] - secs[i]) / 60.0
            if 0 < d <= OVERLAP_MIN:
                no_overlap[i] = False
            if -WASHOUT_MIN <= d < 0:
                washout_ok[i] = False
    return no_overlap, washout_ok


def apply_inclusion(cohort: Cohort) -> tuple[pd.DataFrame, Attrition]:
    """Return (included_meals_with_outcomes, attrition).

    The returned frame has the meal columns plus iAUC outcomes and is restricted
    to meals passing all of §3.2.
    """
    att = Attrition()
    rows = []

    for sid, g in cohort.meals.groupby("subject_id", sort=True):
        g = g.sort_values("meal_time").reset_index(drop=True)
        no_overlap, washout_ok = _neighbor_meal_flags(g["meal_time"])

        ps = {
            "total": 0, "pass_carb": 0, "pass_no_overlap": 0, "pass_washout": 0,
            "pass_cgm_coverage": 0, "included": 0,
        }

        for i, meal in g.iterrows():
            att.total += 1
            ps["total"] += 1

            # (2) known carbs
            if pd.isna(meal["carbs"]):
                continue
            att.pass_carb += 1
            ps["pass_carb"] += 1

            # (3a) no overlapping meal in (0, 120]
            if not no_overlap[i]:
                continue
            att.pass_no_overlap += 1
            ps["pass_no_overlap"] += 1

            # (3b) prior-meal washout in preceding 120 min
            if not washout_ok[i]:
                continue
            att.pass_washout += 1
            ps["pass_washout"] += 1

            # (1) CGM coverage + gap
            win = cohort.cgm_window(sid, meal["meal_time"], READ_LO_MIN, READ_HI_MIN)
            if win.empty:
                continue
            out = compute_ppgr(win["timestamp"], win["glucose"], meal["meal_time"])
            if not (out.has_t0 and out.has_t120):
                continue
            att.pass_cgm_coverage += 1
            ps["pass_cgm_coverage"] += 1

            if not (out.max_gap_min <= MAX_GAP_MIN):
                continue
            if np.isnan(out.iauc_pos):
                continue
            att.pass_gap += 1
            ps["included"] += 1

            rows.append(
                {
                    **{k: meal[k] for k in cohort.meals.columns},
                    "iauc_pos": out.iauc_pos,
                    "iauc_net": out.iauc_net,
                    "baseline": out.baseline,
                    "peak_rise": out.peak_rise,
                    "time_to_peak": out.time_to_peak,
                    "max_gap_min": out.max_gap_min,
                    "n_obs_window": out.n_obs_window,
                }
            )

        att.per_subject[sid] = ps

    included = pd.DataFrame(rows)
    if not included.empty:
        included = included.sort_values(["subject_id", "meal_time"]).reset_index(drop=True)
    _ = WINDOW_MIN  # documented window constant
    return included, att
